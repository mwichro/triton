# Plan: closing the gap on the mixed MMA+FFMA p=4 Laplacian
> Teach the existing forward-propagation that arith.mulf(broadcast, broadcast) can carry MMA encoding when neither broadcast axis collides with the MMA K-axis


## Context

Operator `L u = (K⊗M⊗M + M⊗K⊗M + M⊗M⊗K) u` for p=4 (N=5), F64 on A100. The codegen-driven kernel ([dev_tmp/codegen_mixed_p4.py](dev_tmp/codegen_mixed_p4.py) → [dev_tmp/_generated_mixed_p4_kernel.py](dev_tmp/_generated_mixed_p4_kernel.py)) uses an 8-way **split-tensor** decomposition: every (5,5,5) cube is held as 8 tensors keyed by (k_tag, j_tag, i_tag) ∈ {c,t}³, eliminating all padding. The kernel runs at **2.23 TFLOPS / 24.3 GDoF/s**; the pure-FFMA-without-even-odd literature ceiling is 2.5 TFLOPS. The FP64+TC pipe is 70% idle, so there is substantial headroom.

Triton-side restructuring has been exhausted by the user; codegen is the right code shape. The previous review (`dev_tmp/plan_FMA.md`) targeted `CombineBroadcastMulReducePattern` / `AccelerateMatmul`, neither of which matches the actual bottleneck.

## Bottleneck (from profile in [dev_tmp/triton_BXYZE_mixed_p4_cg.py:20-131](dev_tmp/triton_BXYZE_mixed_p4_cg.py#L20-L131))

| Signal | Value | Reading |
|---|---|---|
| `stall_long_scoreboard` | **65.7%** | L1TEX/SMEM data dep — layout-convert traffic |
| `stall_barrier` | **26.4%** | `bar.sync` on those converts |
| `stall_wait` | 37.8% | DMMA latency not hidden |
| `selected_pct` | 19.4% | Warps issue 1 cycle in 5 |
| `shared_pipe_pct` (FP64+TC) | 30% | TC pipe 70% idle |
| register spill | **0 B** | FFMA codegen is tight |
| HBM SoL | 19% | Not memory-bound |

Inside each of the 24 slabs (4 αβ × 6 phase blocks) of [_generated_mixed_p4_kernel.py](dev_tmp/_generated_mixed_p4_kernel.py), the IR pattern is:

```
_acc      = tl.dot(in_c_flat, O_low4_T)                  # MmaEncoding
_acc      = _acc + in_t_flat[:, None] * O_col4[None, :]  # ← MMA→Blocked convert through smem
_acc_4d   = tl.reshape(_acc, ...)                        # Blocked
_acc_p    = tl.permute(_acc_4d, ...)                     # Blocked
_top,_bot = tl.split(_acc_p)                             # Blocked
_top      = tl.permute(_top, ...)                        # canonical axis order
# next phase consumes _top via tl.reshape → tl.dot operand → Blocked→DotOperand convert
```

≈ 48 smem round-trips + barriers per CTA per element-block. Quantitatively matches 65% `long_scoreboard` + 26% `barrier`.

## Approach: secondary first (zero compiler risk), compiler change only if needed

Reordered per user preference: try the no-compiler-touch wins first, escalate to the compiler patch only if they plateau below target.

### Step 1 — `BLOCK_SIZE_EXEC` bump (no compiler work)

Current `BLOCK_SIZE_EXEC = 8` in [dev_tmp/triton_BXYZE_mixed_p4_cg.py:9](dev_tmp/triton_BXYZE_mixed_p4_cg.py#L9). With **0 B register spill**, there is plain headroom to raise this. More independent MMAs in flight per warp → directly attacks `stall_wait` (38%) and `stall_long_scoreboard` to the extent it is HBM-latency-driven.

- Change: try `BLOCK_SIZE_EXEC ∈ {16, 24, 32}` with matching `BLOCK_SIZE_DATA`. Watch spill.
- Risk: pure-kernel knob, zero compiler risk. Worst case: register pressure rises → spill → instant rollback.

### Step 2 — Slab interleaving in [dev_tmp/codegen_mixed_p4.py](dev_tmp/codegen_mixed_p4.py)

Currently `_gen_phase_contract` emits four slabs (cc, ct, tc, tt) **sequentially** inside each phase block. Each slab's MMA dependency chain is independent, so they can be **interleaved** at the source level: issue all four `tl.dot`s first, then all four FFMA-tail adds, then all four reshape/split sequences, etc. This gives the compiler four independent dependency chains to schedule and hides DMMA latency without any compiler change.

- Change in [codegen_mixed_p4.py:439-657](dev_tmp/codegen_mixed_p4.py#L439-L657) (`_gen_phase_contract`): emit per-step lists across all 4 αβ slabs (loop-interchange the slab-loop and step-loop).
- Risk: kernel-source-level only. Worst case: register pressure spikes from holding 4 accumulators live → spill → rollback.

### Step 3 (gate on Step 1+2 plateauing) — Compiler change: narrow + flag-gated

Only proceed if Steps 1–2 don't reach the target. The compiler change extends `RemoveLayoutConversions` to propagate `NvidiaMmaEncodingAttr` forward through the **exact pattern**:

```
%dot  = tt.dot ... : tensor<...,#mma>
%bc1  = tt.broadcast %x : ... → tensor<...,#blocked>
%bc2  = tt.broadcast %y : ... → tensor<...,#blocked>
%mul  = arith.mulf %bc1, %bc2
%add  = arith.addf %dot, %mul             // ← convert(%dot) currently inserted here
%use  = tt.dot ... %add ...               // ← %add feeds another dot's accumulator chain
```

**Narrow gating, behind a build/runtime flag** (`TRITON_PROPAGATE_MMA_THROUGH_ACC_BCAST=1`, default off). The pattern only fires when:

1. The producer is `tt.dot` with `NvidiaMmaEncodingAttr` on the result.
2. The chain consuming the dot result is exactly `arith.addf(dotResult, arith.mulf(tt.broadcast, tt.broadcast))` (or the symmetric form).
3. The addf result eventually flows into another `tt.dot`'s accumulator operand (after any combination of `tt.reshape`, `tt.permute`, `tt.split` — all layout-preserving in principle).
4. The broadcast dimensions are realizable under the producer's `NvidiaMmaEncodingAttr` (no broadcast along the K reduction axis — that would require cross-lane shuffles).

Default off keeps blast radius zero for other Triton users / other kernels in the same fork. The user can build with `-DTRITON_PROPAGATE_MMA_THROUGH_ACC_BCAST=ON` or set the env var at runtime to opt in.

### Concrete files to modify (Step 3 only)

| File | Change |
|---|---|
| [lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) | Add narrow-gated pattern: when env flag set, propagate `NvidiaMmaEncodingAttr` through the `addf(dot, mulf(broadcast, broadcast))` chain described above. Reuse the existing MMA-anchor logic ([RemoveLayoutConversions.cpp:371](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L371), [:1259-1280](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1259-L1280)). |
| [lib/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.cpp](lib/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.cpp) | Helper deciding whether a `tt.broadcast` can carry `NvidiaMmaEncodingAttr` given the broadcast dimensions (reject broadcasts along the MMA K-axis). |
| [lib/Dialect/TritonGPU/Transforms/OptimizeAccumulatorInit.cpp](lib/Dialect/TritonGPU/Transforms/OptimizeAccumulatorInit.cpp) | Read first — may be the right entry point (smaller patch) instead of `RemoveLayoutConversions.cpp`. |
| `python/triton/runtime/build.py` or equivalent | Wire `TRITON_PROPAGATE_MMA_THROUGH_ACC_BCAST` env var into the pass options. |
| `test/TritonGPU/` (new lit file, gated) | Test that with the flag on, the toy IR pattern compiles with **zero** `convert_layout` between the two dots; with the flag off, IR is unchanged. |

### Risks (Step 3)

- **Correctness:** narrow gating + flag-off-by-default limits exposure to user's own kernels under the flag. Inside the gated path, the main hazard is broadcast-direction overlap with the MMA K-axis — handle in the `LayoutPropagationUtility` helper (reject; fall back to existing convert).
- **Performance regression elsewhere:** with flag default off and pattern restricted to "result feeds another `tt.dot` accumulator" (Condition 3), the pattern essentially only fires in fused multi-dot kernels. Attention / FFN / bias-add kernels do not match Condition 3 (softmax / activation / store breaks the chain). Estimated risk: ~10% of other kernels in the user's tree see any change at all when the flag is on.
- **Effort estimate:** plan_FMA.md's "~30 lines" is too low. Realistic: 150–300 lines once helpers and edge cases are written, 2–3 iterations to get correctness (silent miscompiles need `--verify` runs per iteration).

## Verification

Per step, run [dev_tmp/triton_BXYZE_mixed_p4_cg.py](dev_tmp/triton_BXYZE_mixed_p4_cg.py) `--verify --deep -e 20`. Targets:

| Metric | Baseline | After Step 1+2 | After Step 3 |
|---|---|---|---|
| `gdofs_per_s` | 24.3 | ≥ 30 (hide DMMA latency) | ≥ 36 (also kill convert traffic) |
| `stall_wait_pct` | 38 | < 20 | < 15 |
| `stall_long_scoreboard_pct` | 66 | 50–60 (some HBM component) | < 30 |
| `stall_barrier_pct` | 26 | 20–25 (still convert-bound) | < 10 |
| `shared_pipe_pct` | 30 | 40+ | 60+ |
| `selected_pct` | 19 | 25+ | 35+ |
| spill | 0 | must stay 0 | must stay 0 |
| `--verify` `rel_err` | < 1e-10 | < 1e-10 | < 1e-10 |

If Step 1+2 already reach 30+ GDoF/s and the user is satisfied: stop. Don't open the compiler patch.

Regression check (Step 3 only, with flag ON): re-run `bench/volume/triton_p3.py` and `triton_p7.py` to confirm no regression. With the narrow gating, these should be untouched (their kernels don't have the dot→bcast-mul-add→dot pattern), but verify.

## What I'm NOT planning

- No `Combine.cpp` / `CombineBroadcastMulReducePattern` changes (wrong leverage point).
- No `AccelerateMatmul.cpp` changes (K=4 MMA bulk is correct).
- No `FMADotUtility.cpp` changes (FFMA codegen is tight, 0 spill).
- No Triton-kernel restructure beyond Steps 1–2 (user has exhausted that path).
- No upstreaming of Step 3 in the initial implementation — local-fork patch behind a flag.

## Minimal reproducer (for development + upstream PR)

A standalone ~80-line script that isolates the pattern, demonstrates the slowdown vs an "ideal" baseline, and can be attached to a PR or bug report.

**File:** `dev_tmp/repro_mma_bcast_add.py` (new — to be created during implementation).

**Kernel A — "actual" (has the pattern):**

```python
@triton.jit
def kernel_actual(A, B, C, X, Y, OUT,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, N2: tl.constexpr):
    # All loads pre-tiled; shapes chosen so both dots hit the F64 MMA 8x8x4 tile.
    a = tl.load(A + ...)                  # (M, K)      K = 4
    b = tl.load(B + ...)                  # (K, 2*N)    contiguous "stacked" RHS
    c = tl.load(C + ...)                  # (N, N2)     second-dot RHS
    x = tl.load(X + ...)                  # (M,)
    y = tl.load(Y + ...)                  # (2*N,)

    # ----- The hot pattern: dot → broadcast-mul-add → reshape/split → next dot -----
    acc = tl.dot(a, b)                                # (M, 2*N), MmaEncoding
    acc = acc + x[:, None] * y[None, :]               # ← forces MMA→Blocked convert
    acc2 = tl.reshape(acc, (M, 2, N))
    acc_p = tl.permute(acc2, (0, 2, 1))               # (M, N, 2)
    top, _bot = tl.split(acc_p)                       # (M, N)
    out = tl.dot(top, c)                              # second MMA — chain consumer
    # -----------------------------------------------------------------------------

    tl.store(OUT + ..., out)
```

**Kernel B — "ideal" baseline (no bcast-mul-add):**
Identical to A except the line `acc = acc + x[:, None] * y[None, :]` is deleted. This is the lower bound — what the time would be if the layout convert weren't needed. Difference between A and B is the cost the compiler change is targeting.

**Kernel C — "FFMA reference" (optional):**
Same shapes but `tl.dot` is replaced with `tl.sum(broadcast*broadcast, axis=…)` — the pure-FFMA-no-MMA path, for context.

**Driver:** prints `tflops`, `ms`, and the count of `convert_layout` ops in the lowered TTGIR (via `kernel.asm["ttgir"]` + regex). Runs all three kernels on the same input shape and prints a small table.

**Shape choice:** `M ∈ {128, 256, 1024}`, `K=4`, `N=8`, `N2=4`. K=4 hits the F64 MMA tile exactly (no padding), so the entire delta between A and B is *cleanly* the layout-convert cost — nothing else moves.

**Expected baseline output (before the compiler change):**

| Kernel | TFLOPS | ms | # convert_layout | shared_pipe% |
|---|---|---|---|---|
| A (with bcast-mul-add) | ~3–5 | ~X | 4–6 per CTA | 30–40 |
| B (no add, baseline) | ~12–18 | ~X/3 | 0–1 | 70+ |
| C (FFMA reference) | ~2 | ~5X | 0 | 0 (no TC) |

The A-vs-B gap is the headroom. The compiler-change success criterion: A should approach B (within ~15%), with `convert_layout` count near zero.

**Why this is the right reproducer:**
- One file, two kernels, no external dependencies beyond Triton + PyTorch.
- Pattern is the irreducible core of what's in [_generated_mixed_p4_kernel.py](dev_tmp/_generated_mixed_p4_kernel.py) — same op chain, drastically smaller.
- Both kernels are F64 MMA-tile-aligned → no padding-waste confound, no FFMA-path confound.
- Easy to share on a Triton issue/PR: people can run it without needing the FE benchmark infrastructure.
- Doubles as a lit-test source: the IR dump from kernel A becomes the input to a `test/TritonGPU/*.mlir` lit case asserting "after pass: zero `convert_layout` between the two dots."

**Use during development:**
1. Run baseline (no patch) and record A's ms and `convert_layout` count.
2. Iterate on the patch; after each iteration, re-run and watch both numbers move toward B's values.
3. When A's count drops to ≤1 and ms approaches B's ms, run the full p=4 kernel (`triton_BXYZE_mixed_p4_cg.py --verify --deep`) to confirm the gain carries over to the real workload.

## Open question deferred to implementation

After reading [OptimizeAccumulatorInit.cpp](lib/Dialect/TritonGPU/Transforms/OptimizeAccumulatorInit.cpp): is that the right entry point (smaller, more localized patch) or does the propagation have to live in `RemoveLayoutConversions.cpp`?

## Decision: pause Step 3, file upstream issue

After source pass on [RemoveLayoutConversions.cpp](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp):

- `OptimizeAccumulatorInit.cpp` is the wrong scope (operand-side, init-only). Rule it out.
- `LayoutPropagationUtility.cpp` is 49 lines — only `inferSourceLoadLayout`. No big helper to extend.
- Infrastructure for forward MMA propagation through `addf`/`mulf`/`reshape`/`trans`/`split`/`broadcast` already exists ([:269-332](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L269), [:371](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L371)). Two likely blockers: permuting-reshape anchor at [:216-217](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L216) stops forward prop; broadcast cheap-to-convert heuristic at [:1299](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1299) / [:1377](lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L1377) actively pushes converts *below* broadcast.

### Why we are stopping

Both blockers are heuristics every kernel passes through. Touching them — even behind `TRITON_PROPAGATE_MMA_THROUGH_ACC_BCAST=1` — affects load-bearing decisions for attention / FFN / bias-add patterns. Original "~10% blast radius" was too optimistic; realistic is 30–50% of transformer kernels see *some* layout change when the flag is on. Flag-off-by-default contains it, but the discipline cost (every behavior change inside the flag check, lit tests with flag on/off both, per-kernel opt-in attribute) is high for a fork-local patch.

### Next step

File upstream Triton issue with [dev_tmp/repro_mma_bcast_add.py](dev_tmp/repro_mma_bcast_add.py) attached. Reproducer shows 1.17× slowdown at CHAIN=16 and +1 `convert_layout` per chain step, scaling linearly. Let upstream decide whether to relax the reshape anchor / broadcast cheap-convert heuristic in a way they're comfortable with — they have the cross-kernel visibility we don't.

In the meantime: `num_warps=2` already gave ~9% (24.2 → 26.4 GDoF/s). Steps 1–2 from this plan (BLOCK_SIZE_EXEC bump, slab interleaving) are still available as zero-compiler-risk wins.


