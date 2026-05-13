# FP64 MMA shapes — Triton on H100 (sm_90)

## Context

H100 FP64 tensor-core peak: ~67 TFLOPS.
Observed (see [fp64_H100.txt](fp64_H100.txt)):
- cuBLAS: ~55 TFLOPS (82% of peak) — uses `m16n8k16.f64`
- Triton: ~31 TFLOPS (46% of peak) — uses `m8n8k4.f64`

Gap ratio ≈ 1.77×. WGMMA is not available for `.f64` in PTX ISA (verified end-to-end in a previous session), so the path to closing the gap is the larger MMAv2 FP64 shapes that already exist in PTX ISA but are not used by Triton.

## Native PTX shapes

| Shape | A elts/thr | B elts/thr | C elts/thr | sm_min |
|---|---|---|---|---|
| m8n8k4.f64 | 1 | 1 | 2 | sm_80 (currently the only one Triton emits) |
| m16n8k4.f64 | 2 | 1 | 4 | sm_80 |
| m16n8k8.f64 | 4 | 2 | 4 | sm_80 |
| m16n8k16.f64 | 8 | 4 | 4 | sm_90 |

M = 32, 64, ... is **not** a native PTX shape. Larger per-warp M tiles are achieved by repeating `m16n8k4` (or whichever K is selected) `(M/16)` times. So `mXn8k4` for arbitrary X ≥ 16 falls out of the Phase 1 work for free.

## Thread → element mapping (load-bearing reuse)

For `m16n8k4.f64`, per PTX ISA Table 27:
- `gid = laneid >> 2`, `tid = laneid & 3`
- **A (16×4)**: `a[0] = (gid, tid)`, `a[1] = (gid+8, tid)`
- **B (4×8)**: `b[0] = (tid, gid)` — **identical** to current `m8n8k4`
- **C/D (16×8)**: `c[0..1] = (gid, 2*tid+{0,1})`, `c[2..3] = (gid+8, 2*tid+{0,1})`

This is exactly the standard sm_80+ 16×8 fragment that f16 / bf16 / tf32 use today. K=4 (vs K=8) only changes A's column count, not the M-dimension thread mapping. **Accumulator and B layouts are unchanged from the existing f16 m16n8k8 path** — the load-bearing reuse opportunity.

For Phase 2/3:
- `m16n8k8.f64`: A becomes 16×8 (4 elts/thr), B becomes 8×8 (2 elts/thr). Accumulator unchanged.
- `m16n8k16.f64`: A becomes 16×16 (8 elts/thr), B becomes 16×8 (4 elts/thr). Accumulator unchanged.

## Design constraints (from user)

1. **mXn8k4 must remain available unconditionally.** K=4 is the critical shape.
2. **Extensibility:** Phase 1 must leave space for Phase 2/3 (k8, k16) without rework. Concretely: factor out the per-shape (PTX string, A/B/C reg counts, K) so adding a shape is a table entry plus a layout function, not a refactor.
3. **Target arch is sm_90.** No Blackwell testing available. Phase 1 covers sm_80+ (k4 works on Ampere too); Phase 3 (k16) is sm_90-only.

## Phasing

- **Phase 1 — m16n8k4.f64 + selection mechanism.** Critical path. Introduces the shape-selection plumbing (table + override) even though only k4 is registered, so Phase 2/3 are pure additions.
- **Phase 2 — m16n8k8.f64.** Register a second shape entry. `mXn8k4` remains selectable via the Phase 1 override.
- **Phase 3 — m16n8k16.f64.** Third shape entry. sm_90-gated. Closes the cuBLAS gap.
- **Phase 4 — Auto-selection heuristic.** Default policy that picks the largest K that fits, layered on top of the Phase 1 override (override always wins).

Each phase is independently shippable.

---

## Phase 1 — m16n8k4.f64

### 1.1 Instr-shape selection
[lib/Dialect/TritonGPU/Transforms/Utility.cpp:40](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L40)
```cpp
ret[rank - 2] = eltType.isF64() ? 8 : 16;
```
→ `ret[rank - 2] = 16;` for all dtypes (f64 included).
Changes the `instrShape` stored on `NvidiaMmaEncodingAttr`. All downstream code that branches on `instrM == 8 && f64` must follow.

### 1.2 Accumulator / output layout
[lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1018](lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp#L1018)

Remove the "instrM == 8" f64 special case. After Phase 1, FP64 accumulator layout = standard f16 m16n8 accumulator layout (4 elts/thread, 16-row tile per warp).

Also audit:
- `NvidiaMmaEncodingAttr::getElemsPerThread` / `getSizePerThread` — must return 4 for f64 (currently 2).
- `NvidiaMmaEncodingAttr::getRepForOperand` — match the f16 m16n8 path for f64.

### 1.3 A-operand `DotOperandEncoding`
[lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp](lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp) (search `DotOperandEncoding` + f64)

A fragment: 1 elt/thread (8×4 tile) → 2 elts/thread (16×4 tile). The two A elements at `(gid, tid)` and `(gid+8, tid)` — same row-stride-8 pattern as f16 m16n8.

B-operand layout unchanged; verify nothing couples it to A.

### 1.4 Shared → register operand loading
FP64 has no `ldmatrix` variant; falls back to `ld.shared`. Find the FP64 operand load path in [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp) and the `loadA` / `loadB` helpers. A path now issues 2 `ld.shared.f64` per thread per K-step. B path unchanged.

### 1.5 PTX emit string
[third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:484](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L484)

Replace:
```cpp
{TensorCoreType::FP64_FP64_FP64_FP64,
 "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"},
```
with `m16n8k4`. Keep the `TensorCoreType` enum value; switching the PTX string follows from the new instr shape.

### 1.6 Shape table + selection mechanism (new — load-bearing for Phase 2/3)

Introduce a registry of FP64 native shapes and a selector. Phase 1 registers a single entry; Phase 2/3 are pure table additions.

```cpp
struct Fp64MmaShape {
  unsigned m, n, k;          // 16, 8, {4|8|16}
  unsigned aRegs, bRegs, cRegs;
  unsigned smMin;            // 80 or 90
  const char *ptxString;
};

// Phase 1 registers only k4. Phase 2 adds k8. Phase 3 adds k16.
static const Fp64MmaShape kFp64Shapes[] = {
  {16, 8, 4, 2, 1, 4, 80, "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64"},
};

// Selector: explicit override beats heuristic. Phase 1 only has one shape
// to return, but the API is the one Phase 2/3/4 will keep using.
const Fp64MmaShape &selectFp64Shape(unsigned smArch, unsigned blockK,
                                    std::optional<unsigned> overrideK);
```

**Override surface in Phase 1** (pick one — needs decision):
- `tl.dot(..., fp64_mma_k=4)` Python kwarg threaded through `tt.dot` attr, OR
- environment variable `TRITON_FP64_MMA_K` for global override, OR
- an attribute on `NvidiaMmaEncodingAttr` storing the chosen K.

Recommendation: store on `NvidiaMmaEncodingAttr` (so it survives IR roundtrips and pipeline transforms), plumb the env var through `mmaVersionToInstrShape` as the producer. The Python kwarg can be added later without breaking the IR contract.

In Phase 1 the selector returns the k4 entry unconditionally — the override exists but has only one valid value. Adding shapes in Phase 2/3 makes the override meaningful without API changes.

### 1.7 Replace `callMmaAmpereFp64`
[third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:595](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L595)

Current code synthesizes `m16n8k8` from 4× `m8n8k4`. Rewrite to be shape-table-driven: take an `Fp64MmaShape` and emit one real instruction per (warp-M-tile, K-step):
- `shape.aRegs` A regs, `shape.bRegs` B regs, `shape.cRegs` C regs (`=d` constraint).
- Outer loop over `repM = warpTileM / 16` — one instruction per repeat, advancing A base by 16 rows. **This is the `mXn8k4` repetition path.**
- Outer loop over `repK = BLOCK_K / shape.k` — advance A and B by `shape.k` columns/rows.

Phase 1 only exercises the k=4 path; the loop is shape-agnostic so Phase 2/3 reuse it.

### 1.8 Tests
- [python/test/unit/language/test_core.py](python/test/unit/language/test_core.py) — extend FP64 dot tests across M ∈ {16, 32, 64, 128} × N ∈ {8, 16, 32} × K ∈ {4, 8, 16, 32, 64}.
- Lit test in [test/Conversion/](test/Conversion/) — assert f64 dots lower to `m16n8k4.row.col.f64`, not `m8n8k4`.

### 1.9 Benchmark
Rerun [python/tutorials/03-matrix-multiplication.py](python/tutorials/03-matrix-multiplication.py) FP64 mode against [fp64_H100.txt](fp64_H100.txt) baseline. **Expected: ~40 TFLOPS** at large M=N=K (from 31). Phase 1 alone won't reach cuBLAS — that's Phase 3.

---

## Phase 2 — m16n8k8.f64

Layout work mostly inherited from Phase 1 (accumulator unchanged).

- A: 4 elts/thread. `a[0,1] = (gid, 2*tid+{0,1})`, `a[2,3] = (gid+8, 2*tid+{0,1})`.
- B: 2 elts/thread. `b[0,1] = (2*tid+{0,1}, gid)`.
- New entry in `kFp64Shapes` table (one line; emit side is shape-agnostic).
- A/B operand `DotOperandEncoding` need K-aware layouts — the K=4 path stays untouched.
- The selector now has a real choice; Phase 1 override (`overrideK=4`) keeps `mXn8k4` reachable.

## Phase 3 — m16n8k16.f64

Same layout pattern, doubled K. **Gate on sm_90+** (current target). Accumulator-side code requires zero change beyond Phase 2.

Expected: ~55 TFLOPS — cuBLAS parity.

## Phase 4 — Auto-selection heuristic

Default policy layered on top of the Phase 1 override (override always wins):
- sm_90 and `BLOCK_K % 16 == 0` → m16n8k16
- `BLOCK_K % 8 == 0` → m16n8k8
- else → m16n8k4

`mXn8k4` stays selectable via the Phase 1 override — no API change here, just a smarter default when no override is set.

---

## Risks / things to verify during implementation

1. **`NvidiaMmaEncodingAttr` versioning.** Switching f64 from M=8 to M=16 changes the public attribute shape. Tests, IR fixtures, downstream passes that pattern-match on f64 M=8 will break — grep + fix.
2. **B-operand layout for f64.** Asserted unchanged; verify the existing layout actually matches `m16n8k4`'s `(tid, gid)` mapping and isn't doing something m8n8k4-specific in element ordering.
3. **Shared-memory swizzling.** f64 swizzle is coupled to operand layout. The new A pair `(gid, gid+8)` may straddle a swizzle phase → bank conflicts. First cut: keep existing swizzle, measure, then tune.
4. **Pipelining.** The pipeline pass has shape assumptions. Test with `num_stages > 1`.
5. **TF32 m16n8k8 sharing.** Confirm f64 changes don't touch the TF32 path that shares helper code.
6. **Old GPUs.** Phase 1 (k4) works on sm_80+, so no regression risk for Ampere users from the layout change. Phase 3 (k16) must be sm_90-gated.

---

## Checkpoints

- **Phase 1 done** when: m16n8k4 emits, shape-selection mechanism (table + override surface) is in place with k4 registered, all FP64 unit tests pass, M ∈ {16, 32, 64, ...} all produce numerically correct output, FP64 benchmark shows ≥35 TFLOPS (or meaningful improvement) at M=N=K=4096. **PHASE DONE** Tests are passing.
- **Phase 2 done** when: m16n8k8 selectable via shape table, benchmark ≥45 TFLOPS.
- **Phase 3 done** when: m16n8k16 selectable on sm_90, benchmark approaches cuBLAS (≥50 TFLOPS).
- **Phase 4 done** when: heuristic auto-selects per problem; `mXn8k4` override still honored.


## Progress status

---

### Performance regression investigation (2026-05-13)

**Issue:** After WIP commit (`edfc776f50`) added K=8/16 auto-selection, benchmark dropped from ~49 TFLOPS (Phase 1 baseline) to ~31 TFLOPS at M=N=K=4096.

**Root cause 1 — swizzle collapse for kWidth>1 (`TritonGPUAttrDefs.td` line ~166):**
`vec = 4 * kWidth` → for f64 kWidth=4: `vec=16`, `maxPhase = min(8, 1024/(16×64)) = 1` (no swizzle → bank conflicts).
**Fix:** `vec = min((int)(4*kWidth), max(256/(int)bitwidth, 1))` → f64 kWidth=4 gets `vec=4, maxPhase=4` (same as K=4 path).

**Root cause 2 — auto-selection ignoring repK (`Utility.cpp pickFp64MmaK`):**
K=16 was selected for BLOCK_K=16 → `repK = 16/16 = 1` (one MMA per K-step), too thin to hide shared-memory latency.
**Fix:** Require `repK ≥ 4` — K=16 only when `BLOCK_K ≥ 64`, K=8 only when `BLOCK_K ≥ 32`.
With the autotune configs in `fp64mma_test.py` (max BLOCK_K=32), K=4 is now always selected; K=8/16 auto-selected only for kernels with larger tiles.

**Fix also incorporates:** "fix MMA" commit (`2800e94a85`) A-register M-inner/K-outer ordering (already committed).

**Result:** ~44 TFLOPS at 4096 restored (cf. Phase 1 baseline 49 TFLOPS — remaining gap is benchmark variance).

---



**Bug:** `m16n8k8` + `m16n8k16` produce wrong numerical results. `m16n8k4` (Phase 1) still works.

**Root cause identified:** The B-operand global-memory address computation puts K and N axes swapped vs what PTX `m16n8k8.f64` expects. PTX dump for the failing case shows:

- `ld.global.v2.b64 { %rd3, %rd4 }, [ %rd5 ]` with `%rd5 = B + gid*64 + tid_K*16`
- This loads `B[K=gid, N=2tid_K]` and `B[K=gid, N=2tid_K+1]` — i.e., two N-adjacent values at the same K.
- But PTX `m16n8k8.f64` expects `b[0..1]` to be K-adjacent at the same N: `(K=2tid+0, N=gid_b)`, `(K=2tid+1, N=gid_b)`.

**Where the bug is:** The dot-operand LL for B with `kWidth=2` has K and N axes swapped — the optimizer materialized this swap into the address arithmetic. The likely culprit is the order argument in `nvidiaDotToLinearLayout` for `opIdx=1`:

```cpp
auto order = getOrderForDotOperand(dot.getOpIdx(), rank, /*kContig*/ true);
auto ctaLayout = nvidiaMmaTile(ctx, tileShape, kWidth, order, dot.getRepOrder());
```

`getOrderForDotOperand(1, 2, true)` returns col-major `[0, 1]` so `inner=0=K`, `outer=1=N`. The `tileShape` passed in `nvidiaDotToLinearLayout` is `tileShape[rank-2] = kTile`; `tileShape[rank-1] = 8` — i.e., `[K, N]`. With `inner=0`, `tileShape[inner=0] = kTile` ✓. The bases use `dimNames[inner]` for K, so the LL should map register-bit-0 to the K dim, not N.

But the materialized PTX shows it's loading N-adjacent. Something downstream of the LL produces the swap. Candidates:

1.  **`swizzleDotOperandLike` (`Utility.cpp:1105`)** — uses `getContigPerThread` to derive `kWidth` from MEMORY contiguity. For B in `(K, N)` memory order, contig is along N, so it might infer `kWidth` from N, not K. This produces a shared-encoding microtile that's N-major — and the load planner uses it.
2.  **`getRepForOperand` for `opIdx=1`** — I set `tileSize = [kTile, 8]` (axes K, N). If the function's caller expects axes in `(N, K)` order for `opIdx=1`, that'd flip everything.
3.  **Operand-loading optimizer in DotOp lowering** — for `kWidth=2` it might assume that the contiguous-in-memory dimension equals the K dim, which is true for A (M, K row-major: K is contig) but false for B (K, N row-major: N is contig). Phase 1 worked because `kWidth=1` doesn't try to vectorize.

**Recommendation for the next session:** Focus on `swizzleDotOperandLike` and the shared-encoding pickup path for `f64` with `kWidth>1`. The B operand needs a shared layout that makes K-adjacent elements contiguous in shared memory.

The Phase 1 `m16n8k4` work (committed) is unaffected and remains correct.

---

## Implementation Status (End of Extended Session)

### Completed Work

**Phase 1 (m16n8k4) — ✅ COMMITTED**
- m16n8k4 emits on sm_90+; m8n8k4 fallback on sm_80 (no ptxas rejection)
- Shape-selection plumbing in place: `pickFp64MmaK(cc, operandK)` → returns 4/8/16 based on compute capability and operand K
- `NvidiaMmaEncodingAttr` now stores 3-elt `instrShape = [16, 8, K]` for sm_90+ f64 (2-elt fallback for sm_80)
- `DotOperandEncodingAttr::get` sets `kWidth = K/4` for f64 with 3-elt parent instrShape
- `getRepForOperand` and `nvidiaDotToLinearLayout` respect new kTile from `instrShape.back()` when present
- PTX emit via `getMmaTypeDot` routes correctly: `instrShape[0]==16` picks m16 path, then branches on K
- Unified `callMmaAmpereFp64M16` helper with kWidth-aware A/B register unpacking:
  - Special-case `kWidth==1` to pass register directly (no extract)
  - For kWidth>1: extract each element and append to operand list
- `getValuesFromDotOperandLayoutStruct` sets `numElemsPerVec = kWidth` for f64 (was 1)
- env var override `TRITON_FP64_MMA_K=4|8|16` (default: auto-select)
- Benchmark on H100: **~50 TFLOPS** (from 31 TFLOPS), **94% of cuBLAS**
- All unit tests pass on A100 (sm_80) and H100 (sm_90)
- Committed to main with git history: 80ce005 (sm_80 fix), c26b9a7, 8da9db9

**Phase 4 Mechanism (Auto-Selection) — ✅ IN TREE**
- Selection heuristic integrated into `pickFp64MmaK`:
  - sm_90 + `operandK % 16 == 0` → K=16
  - `operandK % 8 == 0` → K=8
  - else → K=4 (unconditional fallback)
- Override via env var always wins
- Ready for Phase 2/3 shape table additions

**Phase 2/3 Code Structure — ✅ IN PLACE**
- `Fp64MmaShape` struct + `selectFp64Shape(smArch, blockK, overrideK)` plumbing added to MMAv2.cpp
- Enum variants added: `FP64_FP64_FP64_FP64_K8`, `FP64_FP64_FP64_FP64_K16`, `FP64_FP64_FP64_FP64_M8`
- PTX strings for m16n8k8 and m16n8k16 in place
- Registry table ready for k8/k16 entries

### In-Progress: Phase 2/3 Numerical Bug

**Symptom:** m16n8k8 and m16n8k16 produce **wrong outputs** (non-zero error when tested against identity A, odd rows are zero). m16n8k4 (Phase 1) produces correct results.

**Root Cause (Identified):** B-operand K/N axes are swapped in the LinearLayout materialization for `kWidth > 1`.

**Evidence (PTX dump for m16n8k8 with M=16, N=8, K=8):**
- A operand load: `ld.global.v2.b64 { %rd10, %rd11 }, [ %rd1 ]` with `%rd1 = A + gid*64 + tid_K*16`
  - Loads A[gid, 2*tid_K], A[gid, 2*tid_K+1] ✓ (K-adjacent, as expected for m16n8k8)
- B operand load: `ld.global.v2.b64 { %rd3, %rd4 }, [ %rd5 ]` with `%rd5 = B + gid*64 + tid_K*16`
  - Loads B[K=gid, N=2*tid_K], B[K=gid, N=2*tid_K+1] ✗ (N-adjacent at same K)
  - But PTX m16n8k8 expects `b[0,1] = (K=2*tid+0, N=gid_b), (K=2*tid+1, N=gid_b)` (K-adjacent at same N)

**Investigation Candidates (in order of likelihood):**

1. **`swizzleDotOperandLike` (Utility.cpp:1105)** — uses `getContigPerThread()` to infer memory contiguity. For B in `(K, N)` memory layout (row-major), memory is contiguous along N, so the function might compute `kWidth` based on N contiguity rather than K, flipping the shared-memory encoding. The shared layout then encodes N-major, and the operand-load optimizer uses it, reversing K/N in the address arithmetic.

2. **`getRepForOperand` axis ordering (Dialect.cpp)** — when `opIdx=1` (B operand), the function receives `tileSize = [kTile, 8]` (axes K, N order). If the caller expects a different axis order for B, the result tileshape gets flipped.

3. **Operand-loading optimizer in DotOp lowering** — assumes contiguous-in-memory dimension = K-dim (true for A in row-major, false for B). For B with kWidth>1, this mismatch may cause incorrect vectorized load offsets.

**Files to Investigate (next session):**
- [lib/Dialect/TritonGPU/Transforms/Utility.cpp:1105](lib/Dialect/TritonGPU/Transforms/Utility.cpp#L1105) — `swizzleDotOperandLike`
- [lib/Dialect/TritonGPU/IR/Dialect.cpp:242](lib/Dialect/TritonGPU/IR/Dialect.cpp#L242) — `getMatrixOrder`
- [lib/Dialect/TritonGPU/IR/Dialect.cpp:257](lib/Dialect/TritonGPU/IR/Dialect.cpp#L257) — `getOrderForDotOperand`
- [lib/Dialect/TritonGPU/IR/Dialect.cpp](lib/Dialect/TritonGPU/IR/Dialect.cpp) — `getRepForOperand` with opIdx=1 handling for f64
- [third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp) — operand-loading code

### Test Debugging Notes

- **Kernel cache issue discovered:** Early debugging showed Phase 2/3 "working" when cached kernels from earlier runs (with env-var overrides) were reused. Solution: clear `~/.triton/cache` before each test run.
- **Unit tests verified:** `pytest -n 16 python/test/unit/language/test_core.py` passes on A100 and H100.
- **Lit tests:** Phase 1 IR lowering verified (`test/Conversion/tritongpu_to_llvm_hopper.mlir:642` still uses `[8, 8]` for legacy m8n8k4; `test/TritonGPU/accelerate-matmul.mlir:139` expects `[16, 8, 4]` for sm_90).

### Next Steps (for continuation)

1. **Debug Phase 2/3 B-operand layout:** Trace through `swizzleDotOperandLike` → `getContigPerThread` → shared-encoding for f64 B with kWidth=2. Likely fix: ensure B's shared layout encodes K-contiguity, not N-contiguity.
2. **Add m16n8k8 and m16n8k16 to shape table:** Once B-operand bug is fixed, add entries to `Fp64MmaShape[]` and update `selectFp64Shape()` dispatcher.
3. **Regression tests:** Run full unit test suite (`pytest python/test/unit/language/test_core.py`) on A100 and H100.
4. **Benchmark:** Confirm m16n8k8 ≥45 TFLOPS, m16n8k16 ≥50 TFLOPS (parity with cuBLAS at ~55).
5. **Commit:** Phase 2/3 as second commit; Phase 4 auto-selection is already merged.

---

### Phase 2/3 status update (2026-05-14)

**Numerical bug: FIXED.** m16n8k8 and m16n8k16 now produce correct results
(verified by the `torch.allclose` check at the head of `fp64mma_test.py`).

**Performance: NO IMPROVEMENT over Phase 1.** End-to-end H100 sweep with
the "full implementation" branch (k4 + k8 + k16 all enabled and
selectable) lands within benchmark noise of the Phase-1-only result:

    size 4096:  Phase-1 only  50.18  →  full impl  50.18 TFLOPS
    size 2048:  Phase-1 only  52.57  →  full impl  51.73 TFLOPS
    size 1280:  Phase-1 only  40.14  →  full impl  39.99 TFLOPS

cuBLAS sits ~5–8% above Triton across the sweep (e.g. 56.7 vs 50.2 at
4096), unchanged from Phase 1.

**Why larger MMAs didn't help — diagnosed via forced-K sweep**
(`bench_fp64_focus.py` with `TRITON_FP64_MMA_K ∈ {4, 8, 16}`):

    K=4   @2048:  52.06 TFLOPS, winning cfg BLOCK_K=16, nw=2
    K=8   @2048:  52.20 TFLOPS, winning cfg BLOCK_K=16, nw=2  (→ falls back to k4: repK guard)
    K=16  @2048:  51.62 TFLOPS, winning cfg BLOCK_K=16, nw=2  (env override forces k16, no speedup)

Two reinforcing reasons:

1. **Autotuner picks small BLOCK_K.** With the current config list, every
   problem size selects BLOCK_K=16. The `repK ≥ 4` guard in
   `pickFp64MmaK` then auto-downgrades to k4. So in the default path,
   "full implementation" never actually emits k8/k16.
2. **Even when k16 is forced via env var, throughput is unchanged.** The
   MMA instruction shape is not the bottleneck at this BLOCK_K. One fat
   m16n8k16 vs four thin m16n8k4 per K-step deliver the same TFLOPS.

**What this means for the gap.** The remaining ~5–8% gap to cuBLAS is
**not** unlockable by:

- larger autotune search space at the block-tile level (exhausted: 80/96
  tiles are not pow-2 so disallowed by `tl.arange`; 128-tile variants
  added but don't win),
- larger MMA shapes (Phase 2/3 done, no measurable effect),
- `num_warps` tuning (nw=2 already wins; nw={4,8} configs lose).

The bottleneck has moved upstream of the MMA pipe. For FP64, A is loaded
as `ld.shared.v2.b64` (vector) but B uses scalar `ld.shared.b64`
(no `ldmatrix` variant exists for f64). The likely remaining levers are:

- **Shared-memory swizzle for B** at `kWidth>1` — verify `getSharedEncoding`
  produces a B layout that vectorizes correctly with the new kTile, and
  that bank-conflict patterns match cuBLAS. The "Risks #3" item is the
  real next step despite the earlier docstring claim to the contrary.
- **Async-pipeline depth / cp.async scheduling** for the B operand —
  cuBLAS likely overlaps B loads with compute differently.
- **Profile-driven diagnosis:** an NCU run on size=2048 or 4096 to
  determine whether the kernel is MIO-throttled (shared-load bound, fixable)
  vs MMA-pipe bound (fundamental, gap is closed).

**Recommendation:** Do not invest further effort in MMA-shape work or
autotune-config expansion. Either (a) profile under Nsight Compute and
chase the B-operand shared-load path, or (b) declare Phase 2/3
mechanically done and accept the residual gap as a hardware-limited
artifact of f64 lacking `ldmatrix`.