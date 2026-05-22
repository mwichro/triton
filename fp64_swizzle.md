# FP64 residual perf gap — B-operand shared-layout investigation

> **Status as of 2026-05-15:** **Layer 1 swizzle retune landed; root cause
> of residual gap identified.** kWidth=2 path repaired (37 → 47 TFLOPS,
> +27%). Residual gap root cause: **autotune instability** — true best config
> (BM=64,BN=64,nw=2,ns=3) gives 54.4 TFLOPS (−4.4% vs cuBLAS 56.9 on GPU3)
> but autotune picks BM=128,BN=128,nw=8 (52 TFLOPS) due to single-pass timing
> noise. Static analysis (ELF+SASS, HW counters blocked) ruled out: cp.async
> depth, register spilling, occupancy, BK>16 (all spill). Residual ~4.4% is in
> DMMA/LDGSTS scheduling — needs HW counters or wgmma to close further.
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

## Static analysis results (2026-05-15, GPU3 H100, ncu blocked by RmProfilingAdminOnly)

HW counters unavailable (`ERR_NVGPUCTRPERM`). Used ELF resource analysis
(`cuobjdump` segfaults on sm90, parsed `.nv.info` ELF section directly)
and `nvdisasm -c` SASS disassembly instead.

### Hypotheses ruled out

**cp.async pipeline depth** — ruled out. SASS shows `cp.async.wait_group 2`
in the main loop body: correct 3-stage prefetch. The pipeline is not the
bottleneck.

**Register spilling at BM=64,BN=64** — ruled out for the winning config.
EIATTR_FRAME_SIZE = 0, zero STL/LDL in SASS. 254 registers used, nothing
spilled for BM=64,BN=64,BK=16,nw=2.

**Occupancy** — ruled out as the lever. Smaller tiles (BM=32,BN=64 → 7–8
CTAs/SM vs 4 for BM=64,BN=64) are consistently *slower*, not faster. The
kernel is compute-bound; more CTAs just means each CTA does less useful
work before the SM fills up.

**BK > 16** — blocked by register pressure. BK=32 spills (19 STL+LDL,
frame=40B), BK=64 is catastrophic (329 spills, frame=656B). BK=16 is the
only viable value on the nw=2 path.

### BK sweep at size=2048 on GPU3 (direct probe, BM=64,BN=64,nw=2)

| BK | ns | regs | frame | spills | TFLOPS | vs cuBLAS |
|----|----|------|-------|--------|--------|-----------|
| 16 | 3  | 254  |   0   |   0    | 54.40  | −4.4%     |
| 16 | 4  | 250  |   0   |   0    | 54.09  | −4.9%     |
| 32 | 3  | 255  |  40   |  19    | 39.28  | −30.9%    |
| 32 | 4  | 255  |  48   |  23    | 34.86  | −38.7%    |
| 64 | 3  | 255  | 656   | 329    |  9.48  | −83.3%    |

cuBLAS reference: **56.87 TFLOPS** (GPU3, size=2048).

### Tile size comparison at size=2048

| Config              | regs | spills | CTAs/SM(reg) | TFLOPS |
|---------------------|------|--------|--------------|--------|
| BM64,BN64,BK16,nw2,ns3  | 254  | 0  | ≤4   | **54.40** |
| BM64,BN64,BK16,nw2,ns4  | 250  | 0  | ≤4   | 54.09 |
| BM128,BN128,BK16,nw8,ns3 | 240 | 0  | ≤1   | 52.07 |
| BM128,BN128,BK16,nw8,ns4 | 236 | 0  | ≤1   | 51.98 |
| BM32,BN64,BK16,nw2,ns3  | 255  | 0  | ≤4   | 45.66 |
| BM32,BN64,BK16,nw2,ns4  | 255  | 0  | ≤4   | 47.64 |

### Autotune instability finding

The autotune consistently picks **BM=128,BN=128,nw=8** (52 TFLOPS) instead
of the true best **BM=64,BN=64,nw=2** (54.4 TFLOPS). Both are in the config
list. Likely cause: the autotune's single-pass timing is noisy at close
configurations; BM=128 has only 256 CTAs (one wave) so its timing variance
is lower, making it look better in the argmin. The real per-config gap is
~2 TFLOPS but autotune noise is ~1 TFLOPS.

### Residual gap analysis

The direct-probe best (BM=64,BN=64,BK=16,nw=2,ns=3) is **−4.4% vs cuBLAS**.
SASS instruction mix for this config:
- 64 DMMA, 89 LDS, 48 LDGSTS, 1136 total instructions per warp
- Shmem = 48 KB, 4 CTAs/SM (reg-limited), 8 active warps/SM
- No spilling, no excess memory traffic

The ~4.4% gap is most likely in **instruction scheduling**: cuBLAS has
hand-optimized SASS with tighter LDGSTS/DMMA interleaving. Not addressable
through tile-size tuning or Triton-level changes. Would require either
(a) HW counters to confirm DMMA stall fraction, or (b) wgmma (Hopper
warp-group MMA) support in Triton's fp64 path.

**Actionable fix:** Fix the autotune to reliably select BM=64,BN=64,nw=2.
This closes ~1.6 TFLOPS of the apparent gap without any compiler change.
Options: (1) add a `prune_configs_by_size` hint or (2) increase autotune
`rep` count so variance is lower, or (3) hard-code the winner for this
arch/dtype combo.

Previous candidates (now resolved):
1. ~~cp.async pipeline depth~~ — RULED OUT (wait_group 2, pipeline correct)
2. ~~Register spilling at BM=64~~ — RULED OUT (0 frame, 0 STL/LDL)
3. ~~Occupancy~~ — RULED OUT (smaller tiles are slower)
4. ~~BK > 16~~ — BLOCKED by register pressure; BK=32 spills immediately

Remaining open question: autotune noise source (why BM=128 beats BM=64 in
the timed autotune pass but not in isolated `do_bench`). Probably single-pass
variance; fix is to increase rep count or use a more stable comparison metric.

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

- **Autotune instability fix** — BM=64,BN=64,nw=2 is the true best config
  (54.4 TFLOPS direct probe) but autotune picks BM=128,BN=128,nw=8 (52 TFLOPS)
  due to single-pass noise. Options: increase autotune `rep` count, or add a
  size-aware config pruner that down-weights tiles with ≤1 CTA/SM register
  budget. Closing this alone recovers ~1.6 TFLOPS of the apparent gap.
- **NCU profile with HW counters** — needs `RmProfilingAdminOnly=0` (root or
  sysctl). Once accessible: collect `sm__pipe_tensor_op_dmma_cycles_active`
  and `smsp__average_warps_issue_stalled_*` to confirm the residual ~4.4% is
  in DMMA stalls from imperfect LDGSTS/DMMA interleaving, not L2 or occupancy.
- **Lit test** pinning the f64 kWidth≥2 swizzle params (vec=4, perPhase=2,
  maxPhase=4 on the N-contig microtile). Add under `test/Conversion/` so a
  future refactor of the swizzle formula can't silently regress it.
- **A100 sanity rerun** — confirm the `kWidth>=2` branch doesn't trigger on
  sm_80 (m8n8k4 fallback uses kWidth=1, so the new branch is inert, but verify).
