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

- **Phase 1 — m16n8k4.f64.** Critical path. Biggest layout change, smallest emit change. Checkpoint here.
- **Phase 2 — m16n8k8.f64.** K-extension on top of Phase 1.
- **Phase 3 — m16n8k16.f64.** Closes the cuBLAS gap. sm_90-gated.
- **Phase 4 — Shape-selection heuristic.** Pick the right native shape per problem.

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

### 1.6 Replace `callMmaAmpereFp64`
[third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp:595](third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L595)

Current code synthesizes `m16n8k8` from 4× `m8n8k4`. Rewrite to emit one real `m16n8k4` per (warp-M-tile, K-step):
- 2 A regs, 1 B reg, 4 C regs per instruction (`=d` constraint).
- Outer loop over `repM = warpTileM / 16` — one instruction per repeat, advancing A base by 16 rows. **This is the `mXn8k4` repetition path.**
- Outer loop over `repK = BLOCK_K / 4` — advance A and B by 4 columns/rows.

**Extensibility hook:** structure the helper around a small struct describing the native shape:
```cpp
struct Fp64MmaShape {
  unsigned m, n, k;          // 16, 8, {4|8|16}
  unsigned aRegs, bRegs, cRegs;
  const char *ptxString;
};
```
Phase 2/3 plug in new entries; the emit loop is shape-agnostic.

### 1.7 Tests
- [python/test/unit/language/test_core.py](python/test/unit/language/test_core.py) — extend FP64 dot tests across M ∈ {16, 32, 64, 128} × N ∈ {8, 16, 32} × K ∈ {4, 8, 16, 32, 64}.
- Lit test in [test/Conversion/](test/Conversion/) — assert f64 dots lower to `m16n8k4.row.col.f64`, not `m8n8k4`.

### 1.8 Benchmark
Rerun [python/tutorials/03-matrix-multiplication.py](python/tutorials/03-matrix-multiplication.py) FP64 mode against [fp64_H100.txt](fp64_H100.txt) baseline. **Expected: ~40 TFLOPS** at large M=N=K (from 31). Phase 1 alone won't reach cuBLAS — that's Phase 3.

---

## Phase 2 — m16n8k8.f64

Layout work mostly inherited from Phase 1 (accumulator unchanged).

- A: 4 elts/thread. `a[0,1] = (gid, 2*tid+{0,1})`, `a[2,3] = (gid+8, 2*tid+{0,1})`.
- B: 2 elts/thread. `b[0,1] = (2*tid+{0,1}, gid)`.
- New PTX string `mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64`.
- `mmaVersionToInstrShape` needs to carry a K value for f64 — currently the v2 path returns 2D. Either add a K dim or thread the chosen K through the lowering separately.

If the Phase 1 `Fp64MmaShape` table is in place, the emit side is a one-line addition.

## Phase 3 — m16n8k16.f64

Same layout pattern, doubled K. **Gate on sm_90+** (current target). Accumulator-side code requires zero change beyond Phase 2.

Expected: ~55 TFLOPS — cuBLAS parity.

## Phase 4 — Shape selection

Pick the largest native K that divides `BLOCK_K`, subject to register budget:
- sm_90 and `BLOCK_K % 16 == 0` → m16n8k16
- `BLOCK_K % 8 == 0` → m16n8k8
- else → m16n8k4

`mXn8k4` stays selectable via an option / builtin override for the user's critical path.

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

- **Phase 1 done** when: m16n8k4 emits, all FP64 unit tests pass, M ∈ {16, 32, 64, ...} all produce numerically correct output, FP64 benchmark shows ≥35 TFLOPS at M=N=K=4096.
- **Phase 2 done** when: m16n8k8 selectable via shape table, benchmark ≥45 TFLOPS.
- **Phase 3 done** when: m16n8k16 selectable on sm_90, benchmark approaches cuBLAS (≥50 TFLOPS).
- **Phase 4 done** when: heuristic auto-selects per problem; `mXn8k4` override still honored.
