# FP64 residual perf gap — B-operand shared-layout investigation

> **Status as of 2026-05-15:** **Layer 1 swizzle retune landed.** kWidth=2
> path repaired (37 → 47 TFLOPS at size=2048, +27%). Autotuner still
> prefers BLOCK_K=16 (kWidth=1) at most sizes; the residual ~5–8% gap to
> cuBLAS lives in the BLOCK_K=16 path and is **not** in the B-shared-load
> layer. Next investigation: NCU profile of the BLOCK_K=16 winner.
>
> This document is the corrected record. The earlier draft proposed a
> Layer 3 (B-encoding transpose) fix; the TTGIR-flip experiment
> **disproved** that hypothesis — flipping B's `order` from `[1,0]` to
> `[0,1]` produced pure-vectorized PTX but ran 2× slower due to bank
> conflicts. The real lever was Layer 1, three numbers in the swizzle
> attr builder.

## TL;DR

- **Layer 1 fix landed:** for f64 + kWidth≥2 on the N-contig B microtile,
  the swizzle builder was producing `vec=8, perPhase=1, maxPhase=2`. This
  collided with the bank pattern of the m16n8k8 B load. Sweep of
  `(vec, perPhase, maxPhase)` on TTGIR-edited kernels identified
  `vec=4, perPhase=2, maxPhase=4` as conflict-free. Implemented in
  [TritonGPUAttrDefs.td:170-189](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L170-L189).
- **Layer 3 hypothesis (B `order=[0,1]`) is disproven.** The PTX-level
  diagnosis ("B loads are scalar because of N-contig shared encoding")
  was correct as a *description*. The proposed *fix* was wrong: a
  controlled TTGIR-flip showed pure-v2 loads can be emitted, but the
  swizzle params tuned for the old N-contig pattern produce bank
  conflicts under K-contig access. The B loads being scalar was not the
  bottleneck — bank-conflict-free wide loads on the existing N-contig
  layout beat them.
- **m16n8k8 is not faster than m16n8k4 on this kernel.** Even after the
  Layer 1 fix, best BLOCK_K=32 (47 TFLOPS) is still ~4 TF behind
  BLOCK_K=16 (51 TFLOPS) at size=2048. The Phase 2/3 implementation is
  mechanically correct but the autotuner consistently prefers
  BLOCK_K=16; this is now stable, not pathological.
- **The remaining cuBLAS gap is somewhere else.** ~5–8% at sizes ≥1024,
  on the BLOCK_K=16 / kWidth=1 path. Likely candidates: cp.async
  pipeline depth, register usage, warp scheduling. NCU profile needed.

## What changed in code

Single edit in `SwizzledSharedEncodingAttr` builder
([TritonGPUAttrDefs.td:170-189](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L170-L189)):

```cpp
int vec = std::min((int)(4 * kWidth), std::max(256 / (int)bitwidth, 1));
if (needTrans)
  std::swap(vec, mmaStride);
int rank = order.size();
int kDim = opIdx == 0 ? rank-1 : rank-2;
// For f64 with kWidth>=2 (m16n8k8/k16), the N-contig swap inflates vec
// from 4 to 8 which collapses maxPhase to 2 and causes bank conflicts
// in the m16n8k8 B-operand load (sweep showed ~+29% by skipping it).
bool skipNContigSwap = (bitwidth == 64 && kWidth >= 2);
if (order[0] != kDim && !skipNContigSwap)
  std::swap(vec, mmaStride);
if (skipNContigSwap && order[0] != kDim)
  perPhase = std::max(perPhase, 2);
int maxPhase = std::max(std::min<int>(mmaStride, 1024 / (vec * bitwidth)), 1);
// f64 N-contig microtile: don't divide maxPhase by the bumped perPhase
// (the maxPhase derivation already accounts for it via vec/mmaStride).
if (!skipNContigSwap || order[0] == kDim)
  maxPhase = std::max(maxPhase / perPhase, 1);
```

Effect on the emitted shared encoding for B at BLOCK_K=32:

| | vec | perPhase | maxPhase | order |
|---|---|---|---|---|
| Before | 8 | 1 | 2 | [1,0] |
| After  | 4 | 2 | 4 | [1,0] |

A's encoding (`#shared`, K-contig) is unchanged: `vec=4, perPhase=1, maxPhase=4, order=[1,0]`.

## How we got here (the disproven path matters for future work)

### Original hypothesis (Layer 3 in the previous draft)

PTX dump showed B-operand loads were **two scalar `ld.shared.b64`** per
thread instead of one `ld.shared.v2.b64`. TTGIR showed B's shared
encoding had `order=[1,0]` (N-innermost) — wrong axis for K-adjacent
fragment loads. Diagnosis seemed obvious: force B's shared encoding to
`order=[0,1]` (K-contiguous); v2 loads will emit; perf will jump.

### Why the diagnosis was right but the fix was wrong

A controlled TTGIR-edit experiment (`probe_ttgir_flip.py`):

1. Dumped the BLOCK_K=32 winner's TTGIR.
2. Patched only `#shared1` from `order=[1,0]` to `order=[0,1]`,
   leaving `vec=8, perPhase=1, maxPhase=2` unchanged.
3. Recompiled the edited TTGIR via `triton.compile(path)`.
4. Counted shared loads in the resulting PTX:

| Variant | B-side loads | Time @ 2048 | TFLOPS |
|---|---|---|---|
| Baseline `order=[1,0]` | 32 scalar + 32 v2 | 458 µs | 37.5 |
| Flipped `order=[0,1]` | 0 scalar + 48 v2 | 866 µs | 19.8 |

PTX shape changed exactly as predicted (scalar → vector). Runtime
**doubled.** Same MMA count, fewer total load instructions, much slower:
the signature of bank conflicts. The swizzle params (`vec=8,
maxPhase=2`) were tuned for the original N-contig access; under
K-contig access they produce a vectorized load that walks straight
through banks the wrong way.

### Layer 1+2+3 are coupled

Once that experiment ran, a fuller sweep of `(order, vec, perPhase,
maxPhase)` on the flipped TTGIR (`probe_swizzle_sweep.py`) revealed:

- **Best with `order=[0,1]`:** `vec=2, perPhase=2, maxPhase=8` → 37.3 TF
  (just matches the broken baseline).
- **Best with `order=[1,0]`:** `vec=4, perPhase=2, maxPhase=4` → 48.4 TF
  (the keeper).

Across the full grid, no `order=[0,1]` variant beat the best
`order=[1,0]` variant. The eliminated scalar B-loads were not on the
critical path. The actual bottleneck was bank conflicts on the
existing mixed-load pattern, fixable in the swizzle alone.

## Evidence summary

### Stale-binary footnote

A multi-hour confusion: the kWidth=2 / m16n8k8 path appeared to be
**unreachable** in the running binary — every config emitted m16n8k4 in
PTX, and `TRITON_FP64_MMA_K=8` had no effect. Root cause was a stale
`libtriton.so` from an older signature of `mmaVersionToInstrShape` that
ignored `operandK`. Rebuild fixed it. After rebuild:

```
TTGIR for BLOCK_K=32:
  instrShape = [16, 8, 8]
  kWidth = 2
PTX:
  mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64
```

Lesson for next session: **always check `nm -C python/triton/_C/libtriton.so | grep mmaVersionToInstrShape`** to confirm the binary has the 6-arg signature (with `operandK`) before drawing conclusions about kWidth=2 behavior.

### kWidth=2 pathology, reproduced and fixed

From `bench_fp64_layer1_probe.py`:

| Size | BLOCK_K≤16 | BLOCK_K≥32 before fix | BLOCK_K≥32 after fix |
|------|------------|------------------------|----------------------|
| 1280 | 39.95 (cu 43.3, gap 7.7%)  | 27.61 (gap 36.2%) | **36.35** (gap 16.7%) |
| 2048 | 51.84 (cu 55.07, gap 5.9%) | 37.25 (gap 33.3%) | **47.49** (gap 15.0%) |
| 4096 | 51.93 (cu 55.45, gap 6.3%) | 39.45 (gap 29.0%) | **47.56** (gap 12.7%) |

The original "BLOCK_K≥32 is ~30% slower" measurement was real (we
disproved the earlier hand-wave that it was a measurement artifact).
After the Layer 1 fix the kWidth=2 path closes most of the gap to
kWidth=1 but doesn't overtake.

### Full benchmark, post-fix

`python fp64mma_test.py` (autotuner picks across all configs):

| Size | cuBLAS | Triton | gap |
|---|---|---|---|
| 1024 | 47.87 | 45.87 | 4.2% |
| 1408 | 49.18 | 48.02 | 2.4% |
| 1792 | 55.74 | 51.43 | 7.7% |
| 2048 | 55.76 | 51.67 | 7.3% |
| 2432 | 56.15 | 51.07 | 9.1% |
| 4096 | 58.28 | 50.31 | 13.7% |

Numerical correctness: ✅ (`torch.allclose(atol=1e-8, rtol=1e-5)`).
Autotuner still selects BLOCK_K=16 at most sizes — m16n8k8 has no
inherent throughput advantage on this kernel once the swizzle is right.

## Layer accuracy (revised final)

### Layer 1 — swizzle params in `TritonGPUAttrDefs.td` ✅ FIXED

What was wrong: for `bitwidth=64 && kWidth≥2 && opIdx=1` (B operand with
N-innermost order), the `if (order[0] != kDim) swap(vec, mmaStride)`
branch inflated `vec` from 4 to 8, which then forced `maxPhase=2` via
the bank-coverage formula. The resulting `(vec=8, maxPhase=2)` collided
with the m16n8k8 access pattern.

The fix skips the swap and bumps `perPhase` to 2 for this specific
combo, giving `(vec=4, perPhase=2, maxPhase=4)`. Sweep-verified as the
optimum across `vec ∈ {2,4,8}`, `perPhase ∈ {1,2,4}`, `maxPhase ∈ {1,2,4,8}`.

### Layer 2 — `getSrcDstTiles` for f64 v2.b64 — NOT NEEDED

The previous draft proposed adding a `ld.shared.v2.b64` tile entry for
f64. The TTGIR-flip experiment shows that even with pure v2.b64
emission (48 v2 loads, 0 scalar), runtime is worse than the mixed
scalar+v2 pattern when swizzle is wrong. The compiler's existing
lowering can already emit v2.b64 when the layout supports it (A operand
does so today). The "missing tile" diagnosis was a red herring —
nothing was missing.

### Layer 3 — B `order=[0,1]` (force K-contig shared) — DISPROVEN

The empirical experiment falsifies this whole layer. No combination of
swizzle params with `order=[0,1]` beats the best `order=[1,0]` combo.
B's shared encoding being N-contig is **fine** — the m16n8k8 access
pattern doesn't actually require K-contig if the swizzle aligns bank
access across the warp lanes. cuBLAS may use a K-contig layout
internally; we don't, and we don't need to.

### Layer 4 — `ldmatrix.f64` — still out of scope

No PTX/SASS instruction for it. cuBLAS doesn't have it either and still
beats us by 5–8% on the kWidth=1 path. Not the lever.

## What's left

The 5–8% gap to cuBLAS at sizes 1792–2432 (and ~14% at 4096) is now
known to be on the **BLOCK_K=16 / kWidth=1 / m16n8k4** path — the one
the autotuner picks. The B-shared-load layer is no longer suspect.
Next-step candidates, in order of likelihood:

1. **cp.async pipeline depth.** cuBLAS may overlap more K-steps than
   Triton's pipeliner. NCU `smsp__inst_executed_pipe_dmma.avg.pct_of_peak_sustained_active`
   would show DMMA-pipe utilization; if it's <80%, the operand pipeline
   has slack.
2. **Register pressure / spilling.** Triton's lowering might spill
   accumulator registers at BM=BN=64; check `cuobjdump --dump-resource-usage`.
3. **Warp scheduling / occupancy.** numWarps=2 vs 4 vs 8 trade-off
   under different M·N tile sizes. Already sweep-covered by the
   autotuner, but the autotuner only sees runtime, not stall reasons.
4. **m8n8k4 epilogue/prologue?** Less likely — A100 path isn't on H100.

Recommended next action: NCU profile of the BLOCK_K=16 winner at
size=2048. Don't write code until the profile rules in/out the
cp.async hypothesis.

## Repro/diagnostic scripts

- `fp64mma_test.py` — main kernel + benchmark + correctness check.
- `bench_fp64_layer1_probe.py` — splits autotune list into
  BLOCK_K≤16 vs ≥32 subsets, benchmarks both at sizes [1280, 2048, 4096].
  Canonical regression probe.
- `dump_bk32_ptx.py` — restricts autotune to BLOCK_K=32 configs, runs
  once at size=2048 so `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=...` emits
  artifacts.
- `probe_ttgir_flip.py` — proves a hypothesis about shared-encoding
  effects by editing a dumped TTGIR and recompiling. The model script
  for "is the diagnosis right?" experiments — preserve it.
- `probe_swizzle_sweep.py` — sweeps `(order, vec, perPhase, maxPhase)`
  on B's `#shared1` via TTGIR edits + recompile. Identified the
  conflict-free combo before any C++ edit. Reuse pattern for future
  swizzle tuning.
- `bench_flipped.py` — times two compiled TTGIRs (baseline vs edited)
  via direct `k.run(...)` calls. Useful when a kernel must be timed
  outside the autotune harness.
- `bench_fp64_b_colmajor_probe.py` — user-space col-major B probe.
  Result: +0.5–2.5% (within noise). The TTGIR experiments superseded
  this; keep it for the negative result.

## Key files (verified 2026-05-15)

- [include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:160-200](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L160-L200) —
  `SwizzledSharedEncodingAttr` builder. **Layer 1 fix lives here.**
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:35-65](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L35-L65) —
  `pickFp64MmaK`. Threads operand K through `mmaVersionToInstrShape`.
  Sanity check `nm` for 6-arg signature after rebuild.
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:1225](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L1225) —
  `getSharedEncIfAllUsersAreDotEnc`. Where B's shared encoding is
  constructed via the dot-operand attr builder. **No edit needed here**
  — Layer 3 hypothesis was disproven.
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:307-309](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L307-L309) —
  `FP64_FP64_FP64_FP64{,_K8,_K16}` enum; routing in `getMmaTypeDot`
  works correctly post-rebuild.

## Verification (already done, record-keeping)

1. **PTX after fix:** B encoding `#shared1 = vec=4, perPhase=2, maxPhase=4, order=[1,0]`.
2. **Bench:** BLOCK_K=32 path 37.5 → 47.5 TFLOPS at size=2048 (+27%).
3. **Numerics:** `torch.allclose` ✅ end-to-end.
4. **No regression on BLOCK_K≤16 path:** stayed at ~51 TFLOPS (the
   skipNContigSwap branch only triggers for kWidth≥2).
5. **Autotuner:** still prefers BLOCK_K=16 at most sizes — expected;
   m16n8k8 is now competitive but not superior.

## Outstanding (future work, not for this branch)

- NCU profile of the BLOCK_K=16 winner at size=2048 — identify the
  dominant stall reason on the current path.
- Lit test pinning the new f64 kWidth≥2 swizzle params (vec=4,
  perPhase=2, maxPhase=4 on the N-contig microtile). Add under
  `test/Conversion/` so a future refactor of the swizzle formula can't
  silently regress it.
- A100 sanity rerun — confirm the `kWidth>=2` branch doesn't trigger
  on sm_80 (m8n8k4 fallback path uses kWidth=1, so the new branch is
  inert there, but verify).
