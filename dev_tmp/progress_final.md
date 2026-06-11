# FP64 GEMM on H100 — Session Summary (2026-06-11)

**Branches:** `h100_fp64_optimizing_WIP` → `h100_fp64_non_pow2` (checkpoint before prefetch work)
**Goal:** close the ~5% gap between Triton fp64 `tl.dot` GEMM and cuBLAS on H100 (sm_90, GPU 1).

## Where we started

The handout claimed the gap was a shared-memory bank-conflict (swizzle) bug for fp64 kWidth≥2,
with a failed vec-cap fix pending revert, and the repo in a mess (unresolved merge-conflict
markers sitting in MMAv2.cpp, a harmful uncommitted .td edit, ~25 scratch scripts in the root).

## What was done, in order

### 1. Cleanup
- Restored `MMAv2.cpp` from HEAD (working tree had `<<<<<<<` conflict markers — uncompilable).
- Reverted the failed vec-cap edit in `TritonGPUAttrDefs.td`.
- Moved root scratch scripts to `dev_tmp/scratch/`. Rebuilt, re-verified baseline
  (k4 54.4T, k8 48.4T at N=2048).

### 2. Swizzle fix (landed, but theory partly disproven)
- Built a TTGIR-patching sweep (`dev_tmp/probe_swizzle_ab.py`): regex-replace the `#shared`
  encodings in dumped TTGIR, recompile via `triton.compile(path.ttgir)` — no C++ rebuild.
- Found the 16.8M bank conflicts are on the **B operand** (not A); fix = B encoding
  vec=2/perPhase=1/maxPhase=8. Landed in `computeSwizzling` (TritonGPUAttrDefs.td), scoped to
  `bitwidth==64 && kWidth>=2` on the N-contiguous path. Conflicts 16.8M → 43K (400×).
- **BUT** k8 only went 48.4 → 49.3T. Bank conflicts were NOT the main bottleneck.

### 3. Real root cause found
ncu pipe census (Triton k8 vs cuBLAS, identical DMMA and smem-load counts): Triton issues
52.4M FMA-pipe instructions vs cuBLAS's 1.4M — ~216 register moves per K-iter in SASS
(ptxas shuffling loaded fragments into the consecutive register tuples DMMA requires, under
~199-reg pressure). cuBLAS does addressing on the uniform pipe and its LDS land in operand
regs directly. DMMA-util ratio exactly predicts the gap (k4: 90.5/94.5 × 56.7 = 54.3T ≈ measured).

### 4. Wave quantization at large sizes
The "+9–13% at N≥3072" was tile/wave fit, not kernel quality (64×64 → 4.36 waves at 3072 = 87%
tail efficiency). Fixed 4096 (+1.7% paired) by adding 64×128/128×64 nw4 configs to the
`fp64mma_test.py` autotune list (also removed previous session's junk: nw1 config + duplicates).

### 5. Split-tile kernel (96 = 64+32, 80 = 64+16 — no masking, no padding)
`bench_fp64_showcase.py`: two stacked accumulators/dots sharing one B load per K-step; only
the epilogue store is masked. Wins at 1536 (+12.8%→+6.4%), 2304, 2560. Doesn't win at 3072
(two-dot structure costs ~5% per-SM). Also verified: warpsPerCTA=[1,4] from the hasChainedDot
fallback is genuinely right for this shape (patching [2,2] measured slower).



----

## BRANCH `non_pow2` started


### 6. Prefetch pass enabled on sm_90 (the "ptxas register-move project")
- `tritongpu-prefetch` (per-k-slice operand staging) was gated to **Ampere only** in
  `third_party/nvidia/backend/compiler.py`. Enabled for sm_90 — safe because the pass itself
  rejects non-MMAv2 loops, so only fp64 is affected on Hopper.
- Mechanism confirmed: k8 loop moves 216 → 128/iter, addressing migrated to the uniform pipe.
  Net: k8 +2T; small tiles improved (1024 flipped to a 2.9% Triton WIN).
- Ungated it destroys reg-tight configs (nw2 winner −3.4T; 128-wide BK32 tiles → 6–19T with
  hundreds of spills). Added a **register-pressure gate** to `Prefetch.cpp`: estimate
  acc + 3×operand-slice regs/thread ((elems×bits+31)/32), skip above 260 (empirical boundary:
  256 = net win, 272 = net loss). Rank-2 dots only. FileCheck on `test/TritonGPU/prefetch.mlir`
  passes; 60 fp64 dot unit tests pass.

### 7. Benchmark methodology (matters on this box!)
Neighboring GPUs swing clocks ±10%; fresh-process numbers are boost-inflated ~+2T. Only trust
paired same-process rounds (`bench_fp64_showcase.py`, `dev_tmp/head2head.py`). The showcase
also alternates cuBLAS-first/last per round to cancel the first-slot thermal bias.


---


## Final state (paired rounds, median gap vs cuBLAS)

```
1024 -2.9% (win)  1280 +3.3%  1536 +6.4%  1792 +9.5%  2048 +1.2%
2304 +5.9%        2560 +0.4%  3072 +8.2%  3584 +2.0%  4096 -0.1% (tie)
```
Median +2.7% (from ~5% everywhere + 9–13% at large sizes at session start).

## What remains

- The four wave-trap sizes (1536/1792/2304/3072, 6–9.5%): residue = split-kernel per-SM
  overhead + the ptxas-move floor at high-pressure configs. 1792 is the purest case: a
  0.99-wave config exists and still loses — that's per-SM floor, needs ptxas-level control
  we don't have (uniform-pipe addressing is below PTX).
- Possible follow-ups: reduce split-kernel overhead; revisit the prefetch gate threshold if
  ptxas behavior changes; k8/k16 instruction regimes stay non-competitive (capped ~51T).

## Files changed (committed on the branches)
- `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` — fp64 B-operand swizzle fix.
- `third_party/nvidia/backend/compiler.py` — prefetch enabled for sm_90.
- `lib/Dialect/TritonGPU/Transforms/Prefetch.cpp` — register-pressure gate.
- `fp64mma_test.py` — autotune list: wave-fit nw4 configs in, junk out.
- `bench_fp64_showcase.py` — new paired benchmark, pow2 + split kernels, results in docstring.
- `handout.md` — full continuation doc (§0 has the running state).
- Probe tooling in `dev_tmp/`: probe_swizzle_ab.py, probe_bfix_configs.py, ncu_variant.py,
  head2head.py, check_pick.py, wrapper_overhead.py, probe_split*.py.
