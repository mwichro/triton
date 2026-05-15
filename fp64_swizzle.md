# FP64 residual perf gap — B-operand shared-layout investigation

> **Status as of 2026-05-15:** investigation complete through PTX-level
> diagnosis. No code changes attempted yet. This document is the
> resumption-ready record — read it end-to-end before touching code.
>
> **Correction (2026-05-15):** the earlier "BLOCK_K≥32 is ~30% slower"
> figure from `bench_fp64_layer1_probe.py` did not reproduce on rerun.
> The kWidth≥2 path is **modestly behind** (1–3% at size=2048), not
> pathologically broken. The PTX/TTGIR root-cause diagnosis below
> (scalar B loads, N-contig shared encoding) still holds — the magnitude
> claim was the part that was wrong.

## TL;DR

- Phase 1 (`m16n8k4`) reaches ~50 TFLOPS on H100 — within 5–8% of cuBLAS.
- Phase 2/3 (`m16n8k8` / `m16n8k16`) are **numerically correct** but the
  autotuner picks BLOCK_K=16 anyway. Bigger MMAs *should* help compute
  throughput; they don't, which still points at the operand-feed pipeline
  — just with a smaller headroom than initially measured.
- **Root cause (PTX-confirmed):** B operand at `kWidth ≥ 2` is loaded as
  **two scalar `ld.shared.b64`** per thread (instead of one
  `ld.shared.v2.b64`). The two K-adjacent B elements per thread sit at
  **non-contiguous addresses** in shared memory because B's shared
  encoding has `order=[1,0]` (N-contiguous), inherited from global memory.
  PTX uses two *different base registers* for the two loads — not a
  vectorization choice the lowering can override, a layout fact.
- **The fix is not in `getSrcDstTiles`** as the earlier draft of this plan
  guessed. Even adding `ld.shared.v2.b64` as a tile option wouldn't help —
  the elements being loaded aren't adjacent. The real fix is to construct
  B's shared encoding with `order=[0,1]` (K-contiguous) when `kWidth≥2`
  for f64, which means physically transposing B during global→shared.
- **User-space workaround does not work.** Passing B with K-contig global
  strides (`b.T.contiguous().T`, stride `(1, K)`) gives +0.5–2.5% at
  size=2048, within noise. `getOrderForMemory` does follow global stride,
  but the dot-operand lowering at kWidth≥2 does not vectorize even when
  global is K-contig. See "User-space probe" section below.
- **This is not hardware-limited** (cuBLAS does it on the same H100). The
  expected win is modest though — closing a 5–8% gap, not a 30% one —
  so the cost/benefit of the ~150–400 line Layer 3 implementation is
  weaker than originally estimated.

## Evidence (don't re-derive — captured here)

### Bench probe: kWidth=2 path is modestly behind, not pathological

`bench_fp64_layer1_probe.py` splits the autotune list and benchmarks each.

**Original measurement (2026-05-14, did not reproduce on rerun):**

| Size | BLOCK_K≤16 (kWidth=1) | BLOCK_K≥32 (kWidth=2) | Δ |
|------|-----------------------|------------------------|---|
| 1280 | 40.40 (cu 43.5, gap 7%) | 27.64 (gap 37%) | **−32%** |
| 2048 | 51.61 (cu 54.5, gap 5%) | 37.37 (gap 33%) | **−28%** |
| 4096 | 50.72 (cu 54.4, gap 7%) | 39.45 (gap 35%) | **−22%** |

**Rerun (2026-05-15, size=2048 only, via `bench_fp64_b_colmajor_probe.py`
row-major arm):**

| Size | BLOCK_K≤16 | BLOCK_K≥32 | Δ |
|------|------------|------------|---|
| 2048 | 52.23 (cu 56.87, gap 8.1%) | 51.15 (cu 55.94, gap 8.6%) | **−2.1%** |

The original 28% collapse was likely a measurement artifact (stale
autotune cache or one-off thermal/sharing effect on the GPU). The
*real* gap is ~1–3% at size=2048, both BLOCK_K subsets sitting at
~52 TFLOPS with an ~8% gap to cuBLAS. **The cuBLAS gap is the actual
problem — not a kWidth=2-specific collapse.**

Winning BLOCK_K=32 config at 2048: `BM=64, BN=64, BK=32, nw=4, ns=3`
(was `nw=2` in the original; another sign the original autotune state
was unstable). m16n8k8 is being emitted (instrShape=[16,8,8]).

### User-space probe: B as K-contig global

`bench_fp64_b_colmajor_probe.py` re-runs both BLOCK_K subsets at
size=2048 with B passed as `torch.randn((N,K)).t()` — physical N×K
row-major, logical K×N with stride `(1, K)`. This makes
`getOrderForMemory(B)` return `[0, 1]` instead of `[1, 0]`.

| Subset | B row-major | B col-major (K-contig) | Δ |
|---|---|---|---|
| BLOCK_K≤16 | 52.23 | 53.55 | +2.5% |
| BLOCK_K≥32 | 51.15 | 51.42 | +0.5% |

Within noise. The compiler accepts the K-contig global layout (no
correctness failure, autotuner picks similar configs), but the
dot-operand lowering at kWidth≥2 doesn't visibly switch to vectorized
shared loads. This means the **"make transpose the user's responsibility"**
idea is not viable as a workaround — the compiler-side fix is required
to get any meaningful speedup.

### Forced-K sweep (`bench_fp64_focus.py`)

With autotune still picking BLOCK_K=16, forcing `TRITON_FP64_MMA_K∈{4,8,16}`:

| Forced K | size 2048 | size 1280 |
|---|---|---|
| 4  | 52.06 (cu 56.0) | 40.00 (cu 43.8) |
| 8  | 52.20 (cu 56.7) | 39.85 (cu 43.5) |
| 16 | 51.62 (cu 56.1) | 40.17 (cu 43.6) |

Identical to 1–2% across forced K values → the m16 MMA shape itself is
not the differentiator on this path. The winning config stays
BLOCK_K=16 in every case (env override forces the *MMA shape*, not
BLOCK_K), so this comparison is at kWidth=1 always for the autotuner.

### PTX dump (the load-bearing evidence)

Dumped via `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/fp64_bk32_dump`
running `dump_bk32_ptx.py` (BLOCK_K=32 configs only, size=2048).

K-loop body (`/tmp/fp64_bk32_dump/2SWHJWBQO56Z4JBWMUXBQMK73RIGB6VAZRUSLAZOYNHN5LB7JK7A/matmul_kernel_fp64.ptx`, lines 485–570):

```
ld.shared.v2.b64 {%rd74, %rd76}, [%r170];          // A load 1
ld.shared.v2.b64 {%rd75, %rd77}, [%r170+2048];     // A load 2
... 14 more A v2 loads ...
ld.shared.b64    %rd79, [%r175+49152];             // B load 1 (scalar)
ld.shared.b64    %rd78, [%r174+49152];             // B load 2 (scalar)
... 14 more B scalar loads ...
mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64
  { %rd169-%rd172 },                                // C
  { %rd74, %rd75, %rd76, %rd77 },                   // A (4 regs, v2 path)
  { %rd78, %rd79 },                                 // B (2 regs, scalar path)
  { %rd169-%rd172 };
```

**Key:** B loads use `%r174` and `%r175` (two different base registers)
to fetch the two K-adjacent B elements for the same mma's B fragment.
They are not adjacent in shared memory — there is no v2 to emit here.

### TTGIR shared encodings (the root cause)

Same dump, `matmul_kernel_fp64.ttgir`:

```
#shared  = swizzled_shared<vec=4, perPhase=1, maxPhase=4, order=[1, 0]>
#shared1 = swizzled_shared<vec=8, perPhase=1, maxPhase=2, order=[1, 0]>

%a = ttg.local_alloc : memdesc<3x64x32xf64, #shared,  #smem, mutable>
%b = ttg.local_alloc : memdesc<3x32x32xf64, #shared1, #smem, mutable>
```

`order=[1, 0]` means axis-1 is innermost. For A (M×K) axis-1 is K → A is
K-contiguous, good. For B (K×N) axis-1 is N → **B is N-contiguous**,
which is the wrong axis for the m16n8k8 B-operand fragment (PTX wants
K-adjacent register pairs). Two K-adjacent elements are at stride-N=32
in shared memory → cannot vectorize → scalar loads. The structural
issue is real; the empirical magnitude is the residual ~5–8% cuBLAS gap,
not the 28% claimed earlier.

### Swizzle-math sanity (Layer 1 is inert here)

f64 cap: `vec = min(4·kWidth, max(256/64, 1)) = min(4·kWidth, 4)`.

| kWidth | vec | maxPhase |
|---|---|---|
| 1 | 4 | min(8, 1024/(4·64)) = 4 |
| 2 | 4 | 4 |
| 4 | 4 | 4 |

All kWidths produce identical swizzle params (the `vec=8, maxPhase=2`
seen in `#shared1` above comes from the `if (order[0] != kDim) swap(vec,
mmaStride)` branch, not the cap; the swap is the standard handling for B
with N-innermost order). The cap itself does not differentiate kWidths
for f64, so relaxing it changes nothing visible until something else
unsplits the loads.

## Layer-by-layer accuracy update (corrects earlier draft)

### Layer 1 — Swizzle `vec` cap in `TritonGPUAttrDefs.td:170`

- **Code:** [include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:170](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L170)
  `vec = std::min((int)(4 * kWidth), std::max(256 / (int)bitwidth, 1));`
- **What it does today:** caps `vec` at 4 for f64 — keeps `maxPhase ≥ 4`
  for f64 instead of collapsing to 1 (the existing fix from plan_fp64.md
  "Root cause 1").
- **Effect on the residual gap:** **none in isolation**. Math + PTX both
  show kWidth=1 and kWidth=2 produce the same swizzle params; the
  pathology is elsewhere. Relaxing or tightening the cap with no other
  change will leave the kWidth=2 perf collapse intact.
- **When it might matter:** *after* Layer 3 lands. If B's shared layout
  becomes K-contiguous and v2.b64 actually emits, vec=4 (32 B per vec
  line) is smaller than the load width (16 B/thread × 32 threads = 512 B
  per wave; per-thread load = 16 B). A vec=8 variant (64-B vec line)
  might align better with the new access pattern. Re-evaluate then.
- **Depth:** 1-line tweak, but conditional on Layer 3.

### Layer 2 — Add `ld.shared.v2.b64` to `getSrcDstTiles`

- **Code:** [lib/Conversion/TritonGPUToLLVM/Utility.cpp:43-81](lib/Conversion/TritonGPUToLLVM/Utility.cpp#L43-L81).
- **What it does today:** `bitwidth ≤ 32` gate excludes f64 from
  ldmatrix; only scalar `ld.shared` tiles are offered for bitwidth=64.
- **Why I previously thought this was the fix:** v2.b64 exists in PTX
  and A uses it today. Adding a v2.b64 tile for f64 *seemed* sufficient
  to vectorize B loads.
- **Why it isn't (PTX evidence):** the two K-adjacent B elements per
  thread live at *different base registers* (e.g. `[%r174+offset]` vs
  `[%r175+offset]`). Even if Triton's lowering offered v2.b64, it could
  not coalesce two loads from disjoint base registers into one. The
  layout makes vectorization impossible.
- **Where it would help:** *after* B's shared layout becomes K-contig
  (Layer 3), the two elements land at adjacent addresses and v2.b64
  becomes legal. So Layer 2 is a **paired follow-on** to Layer 3, not a
  standalone fix.
- **Depth (when paired with Layer 3):** ~20–40 lines.

### Layer 3 — B's shared encoding `order` (the real fix)

- **Construction site:** [lib/Dialect/TritonGPU/Transforms/Utility.cpp:1225-1229](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L1225-L1229)

  ```cpp
  auto order = getOrderForMemory(srcTy);
  unsigned bitWidth = srcTy.getElementTypeBitWidth();
  tempAttr = ttg::SwizzledSharedEncodingAttr::get(
      val.getContext(), dot, srcTy.getShape(), order, CGALayout, bitWidth,
      /*needTrans=*/false);
  ```
- **What it does:** picks `order` from global memory layout. For B
  (K×N row-major in fp64mma_test), `getOrderForMemory` → `[1, 0]` (N
  innermost). This propagates into the shared encoding as we see in the
  TTGIR.
- **What `needTrans` actually does:** [TritonGPUAttrDefs.td:172-173](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L172-L173)
  only swaps `vec` and `mmaStride` for the swizzle computation. **It
  does not transpose the data placement.** It exists for the f16/bf16
  case where `ldmatrix.trans` performs the transpose at load time. f64
  has no `ldmatrix.trans` equivalent, so `needTrans=true` would only
  fiddle the swizzle params, not solve the access pattern.
- **What's actually required:** B's shared layout must be K-contiguous
  (`order=[K-dim, N-dim]`) when `kWidth ≥ 2` for f64. That implies B is
  physically transposed during the global→shared copy. cuBLAS does
  exactly this.
- **Constraints / complications:**
  - The global B is K×N row-major in the kernel. A K-contig shared
    layout requires either:
    1. **Cross-lane reads at cp.async time** — each warp's lanes read N
       elements per K row, but write into shared memory as a column of
       K-contig data. This is what `cp.async` with the right destination
       swizzle does for f16; the swizzle has to align with the access
       so warp lanes hit different banks. Whether the existing
       `swizzled_shared` attribute can express the needed permutation
       for 64-bit elements without a new variant is open.
    2. **Two-phase staging** — copy to shared as N-contig, then run a
       separate shuffle pass to re-layout into K-contig. Expensive,
       likely worse than the current scalar-load cost.
  - The pipeliner (`lib/Dialect/TritonGPU/Transforms/Pipeliner/`) inserts
    `async_copy_global_to_local` with a specific destination shared
    layout. Changing the layout shape here may require coordinated edits
    in the pipeliner.
  - Conditional logic must trigger only for f64 + kWidth≥2 — the f16
    path with `ldmatrix.trans` must remain unchanged.
- **Files likely touched:**
  - `lib/Dialect/TritonGPU/Transforms/Utility.cpp:1225` (and
    `getOrderForMemory` callers).
  - `lib/Dialect/TritonGPU/IR/Dialect.cpp` —
    `getOrderForDotOperand`, `getSharedEncoding` paths.
  - `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:170-184` —
    swizzle attr builder may need a new branch for "f64 + K-contig B".
  - `lib/Dialect/TritonGPU/Transforms/Pipeliner/` — `async_copy` lowering
    that uses the new shared layout.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`
    (around line 1455 — `SwizzledSharedEncodingAttr` consumer).
- **Depth:** ~150–400 lines, 3–5 files. Highest-impact and highest-risk
  layer in this plan.

### Layer 4 — `ldmatrix.f64` (out of scope, kept for completeness)

`ldmatrix` has no f64 variant in PTX/SASS. Replicating its cross-thread
effect at f64 would be a structural change. cuBLAS achieves the same
result through (swizzled K-contig shared + vectorized v2.b64), so this
is not the path to chase.

## What cuBLAS likely does

- B in shared memory is laid out **K-contig** (transposed from global
  K×N row-major during the copy).
- Reads from shared memory via vectorized `ld.shared.v2.b64`.
- Swizzling chosen so K-strided reads land on different banks across
  warp lanes — zero bank conflicts.
- Async copies (`cp.async`) pipeline B fetch with the previous K-step's
  MMA. Triton's pipeliner already does this; not in scope.

## Why the current state is *not* a regression

- The autotuner in `fp64mma_test.py` selects `BLOCK_K=16` (kWidth=1) at
  most sizes (BLOCK_K=32 wins at some). Both subsets land near
  ~52 TFLOPS at size=2048, both ~8% behind cuBLAS.
- End-to-end Triton TFLOPS at size 4096: ~50, vs cuBLAS ~57 (~12% gap).
- Phase 2/3 are correct but don't materially shift the headline number
  — the bigger MMAs don't help because the operand-feed pipeline is
  the bottleneck, not MMA throughput.
- "Doing nothing" leaves us at the Phase 1 result with a 5–12% cuBLAS
  gap (size-dependent). Fixing Layer 3 *should* close some of this,
  but the upside is bounded by the gap itself — there isn't a hidden
  30% to recover.

## Resumption checklist (start here next session)

1. **Re-read this document.** Sanity-check by re-running
   `bench_fp64_layer1_probe.py` and `bench_fp64_b_colmajor_probe.py`.
   Expect BLOCK_K≤16 and BLOCK_K≥32 to both land ~52 TFLOPS at size=2048
   with ~8% gap to cuBLAS, and the col-major arm to give ≤+3%. If the
   numbers diverge wildly, the autotune cache is unstable — clear it and
   rerun before drawing conclusions.
2. **Decide path:**
   - **(a) Implement Layer 3 (then 2, then 1).** Highest reward, deepest
     work. Start by reading the construction site at Utility.cpp:1225,
     understanding all callers of `getSharedEncIfAllUsersAreDotEnc`, and
     prototyping a conditional that forces K-contig shared for f64
     dotOp opIdx=1 with kWidth≥2. Validate first with a forced
     small-size matmul, then sweep.
   - **(b) Document and stop.** Update plan_fp64.md with the diagnosis,
     leave Phase 2/3 unwired from the autotune fast path, accept the
     5–8% cuBLAS gap as the cost of f64's missing `ldmatrix.trans`.
   - **(c) Discuss with upstream Triton maintainers** — the K-contig
     shared layout for f64 dotOps is a generic gap, not specific to this
     kernel. May be worth a design discussion before implementation to
     avoid stepping on planned refactors.
3. **NCU profile before committing to (a).** Run on the BLOCK_K=16
   winner at size=2048 — confirm whether shared-load throughput is
   actually the dominant stall reason for the *current* path. If it
   isn't, even fixing Layer 3 won't close the cuBLAS gap (the autotuner
   would still prefer BLOCK_K=16 for some unrelated reason).

## Key files (with current line numbers verified 2026-05-14)

- [include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:160-184](include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L160-L184) —
  `SwizzledSharedEncodingAttr` builder (Layer 1 + part of Layer 3).
- [lib/Conversion/TritonGPUToLLVM/Utility.cpp:43-81](lib/Conversion/TritonGPUToLLVM/Utility.cpp#L43-L81) —
  `getSrcDstTiles` (Layer 2 — paired follow-up only).
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:1150-1186](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L1150-L1186) —
  `swizzleDotOperandLike`.
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:1192-1240](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L1192-L1240) —
  `getSharedEncIfAllUsersAreDotEnc` — **construction site of B's shared
  encoding, the place to start for Layer 3**.
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:35-65](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L35-L65) —
  `pickFp64MmaK` (current K-selector with `repK≥4` guard).
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:99-131](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L99-L131) —
  `getValuesFromDotOperandLayoutStruct`, register unpacking
  (`numElemsPerVec = kWidth` for f64 — already correct, no change
  expected).
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:307-309](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L307-L309) —
  `FP64_FP64_FP64_FP64{,_K8,_K16}` enum (Phase 2/3 in tree).
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1455](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1455) —
  consumes `SwizzledSharedEncodingAttr` during lowering.

## Repro/diagnostic scripts (committed/uncommitted, all reusable)

- `fp64mma_test.py` — main kernel + benchmark + correctness check.
- `bench_fp64_focus.py` — single-size benchmark with autotune config
  printout (uses module-import trick to avoid running top-level bench).
- `bench_fp64_layer1_probe.py` — splits autotune list into
  BLOCK_K≤16 vs ≥32 subsets, benchmarks both at sizes
  [1280, 2048, 4096]. Use this as the canonical regression probe.
- `dump_bk32_ptx.py` — restricts autotune to BLOCK_K=32 configs,
  runs once at size=2048 so `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=...`
  emits the .ptx/.ttgir/.sass artifacts to inspect.
- `bench_fp64_b_colmajor_probe.py` — runs both BLOCK_K subsets with B
  in row-major *and* col-major (K-contig global) global layouts.
  Confirms the user-space transpose workaround is within noise.

## Artifacts to preserve (git add list)

The diagnosis above quotes specific addresses and shared-encoding lines
from one particular PTX/TTGIR pair. Capture it into the repo so the
evidence survives `/tmp` cleanup; everything else here is regenerable.

```bash
# 1. PTX/TTGIR/TTIR/LLIR for the BLOCK_K=32 winner at size=2048
#    (the dump cited above; ~250 KB total).
mkdir -p fp64_dumps/bk32_winner_size2048
cp /tmp/fp64_bk32_dump/2SWHJWBQO56Z4JBWMUXBQMK73RIGB6VAZRUSLAZOYNHN5LB7JK7A/matmul_kernel_fp64.{ptx,ttgir,ttir,llir} \
   fp64_dumps/bk32_winner_size2048/

# 2. The plan and the scripts cited above.
git add fp64_swizzle.md \
        fp64_dumps/ \
        fp64mma_test.py \
        bench_fp64_focus.py \
        bench_fp64_layer1_probe.py \
        dump_bk32_ptx.py
```

Skip the `.cubin` and `.sass` in the dump dir (binary blob and
regenerable via `ptxas` / `nvdisasm` from `.ptx`). The other untracked
files in the working tree (`bench_fp64_quick.py`, `repro_k*.py`,
`probe_b2_swizzle.py`, `dump_k*.py`, `test_*.py`, `ptx*.txt`, `MMAv3.md`,
`wgmma_fp64.md`, etc.) are dead-end probes from earlier sessions, not
referenced from this plan — `.gitignore` or leave untracked.

## Numerical correctness still ✅

`fp64mma_test.py` runs `torch.allclose(triton, torch, atol=1e-8,
rtol=1e-5)` and prints ✅. Phase 2/3 are functionally correct. Any
edits in Layer 2/3 must preserve this — re-run after every rebuild.

## Verification (when work resumes)

1. **PTX re-inspection:** after Layer 3 lands, dump the BLOCK_K=32
   kernel — B-side loads must be `ld.shared.v2.b64` (single vector per
   thread per K-step), not two scalars. The two B-element addresses
   must share a base register (or differ by ≤8 B).
2. **TTGIR check:** `#shared1` for B should show `order=[0, 1]` (or
   whatever K-inner equivalent) when kWidth≥2.
3. **Probe sanity:** `bench_fp64_layer1_probe.py` BLOCK_K≥32 path must
   beat the current ~51 TFLOPS at size=2048 by a margin clearly outside
   noise (target ≥54 TFLOPS, matching cuBLAS within 3%). If it merely
   ties the BLOCK_K=16 path again, the fix didn't change the load
   pattern — re-diagnose via PTX dump before claiming success.
4. **Numerical correctness:** `torch.allclose` ✅; full
   `pytest python/test/unit/language/test_core.py -k dot` on A100 + H100.
5. **NCU:** DMMA-pipe utilization ≥80% at size=2048; long-scoreboard
   stalls subordinate; bank-conflict rate near zero.
6. **End-to-end:** rerun `python fp64mma_test.py`. Target at M=N=K=4096:
   Triton ≥ 54 TFLOPS (current 50.2, cuBLAS ~56.7).
7. **No regression on sm_80:** rerun on A100 — Phase 1 `m8n8k4`
   fallback path must be unchanged. The Layer 3 change must be gated on
   `sm ≥ 90` and `f64` and `kWidth ≥ 2`.
8. **Lit tests:** `test/Conversion/` assertion that f64 B-operand at
   kWidth≥2 lowers to `ld.shared.v2.b64` (not scalar).
