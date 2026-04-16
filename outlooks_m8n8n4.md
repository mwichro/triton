# Triton FP64 `m8n8k4` Limitations and Outlook

## Overview
Recent PRs in Triton added support for Double Precision (`fp64`) Matrix Multiply-Accumulate via the `m8n8k4` PTX instruction for Ampere (SM80) and Hopper (SM90).

However, while the native hardware instruction calculates an `8x8x4` tile, Triton artificially forces a minimum block size of `16x8x16` during compilation, causing segmentation faults if smaller shapes are provided.

## Why does it segfault?
The segmentation fault occurs during MLIR to LLVM IR lowering in `TritonGPUToLLVM/DotOpToLLVM/MMAv2.cpp` and the TritonGPU Dialect layout calculation:

1. **Simulated Macro-blocks**: To maximize pipeline throughput and register reuse, the compiler groups exactly 8 `m8n8k4` hardware calls logically together: `(2 in M) x (1 in N) x (4 in K) = 8 instructions`. It explicitly iterates over hardcoded lengths (e.g., `int kRegs = 4`) and unconditionally issues multiple `mma` operations (`retArgs1` and `retArgs2`) across the M dimension.
2. **Tile Size Enforcement**: In `TritonGPU/IR/Dialect.cpp` (`NvidiaMmaEncodingAttr::getRepForOperand`), the `tileBitWidthK` for Ampere `fp64` is overridden to `4 * 256 = 1024`, forcing `tileSize[K] = 16`. `tileSize[M]` is strictly hardcoded to `16`.
3. **Array Bounds**: If an operation specifies a shape like `8x8x8`, integer division computes `repM = 8 // 16 = 0` and `repK = 8 // 16 = 0`. The generation loops in `callMmaAmpereFp64` still assume the full 16x8x16 layout, indexing out-of-bounds `ValueTable` elements and violently crashing the compiler.

## Can it be fixed inside Triton?
Yes, but it requires uncoupling the `m16n8k16` simulation logic.

To fully support arbitrarily small `m8n8k4` blocks:
- **`NvidiaMmaEncodingAttr::getRepForOperand`** would need to relax `tileSize` to `M=8` and `K=4` for `fp64` on SM80.
- **`MMAv2.cpp (callMmaAmpereFp64)`** would need to dynamically determine `numMmaRets` and `kRegs` loops based on the actual tile shape, rather than hardcoding `kRegs=4` and statically dual-issuing `cArgs1` and `cArgs2`. 

Until this macro-block unrolling logic is made elastic, the constraint must be caught at the frontend compiler level (e.g. JAX `pallas`) to prevent backend crashes.


# Plan to Extend FP64 MMA to Support Small Block Sizes (8x8x4)

## Problem Context
PR #7310 introduced `fp64` MMA support using the `m8n8k4` PTX instruction for Ampere (SM80) and Hopper (SM90). However, the implementation forces a simulated macro-block of `16x8x16`. If Triton encounters smaller shapes (e.g., `8x8x8` or `8x8x4`), it experiences array out-of-bounds errors (segfaults) during the LLVM IR lowering phase because the layout engine forces minimum shape requirements and the emitting loop unconditionally indexes into the `ValueTable` arrays assuming a `16x8x16` footprint.

## Proposed Action Plan based on PR7310

### 1. Dynamic Layout Tile Sizing in TritonGPU Dialect
The primary issue stems from hardcoded assumptions in `NvidiaMmaEncodingAttr::getRepForOperand`. We need to un-hardcode the macro-block scaling for `fp64`.

*   **File:** `lib/Dialect/TritonGPU/IR/Dialect.cpp`
*   **Changes:**
    *   In `getRepForOperand`, rather than forcing `tileBitWidthK = (isAmpere() && bitwidth == 64) ? (4 * 256) : (4 * 64)`, check if the provided operation/block size allows for smaller multiples. 
    *   Currently, the logic overrides `tileSize[M] = 16` and computes `tileSize[K]` from `tileBitWidthK`. Update this section to permit `tileSize[M] = 8` and `tileSize[K] = 4` when `fp64` small shapes (like `8x8x4`) are detected or requested by the tensor shape.
    *   Ensure that the macro-block multipliers (the simulated `2` in `M` and `4` in `K`) are only applied if the tensor's dimensions are large enough to support them.

### 2. Parameterize Loop Execution in the LLVM Lowering
The PTX emission phase unconditionally tries to group `m8n8k4` instructions into a `16x8x16` grid (8 PTX instructions per call).

*   **File:** `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`
*   **Function:** `callMmaAmpereFp64`
*   **Changes:**
    *   Modify the macro-block loops. Instead of hardcoding `for (int vk = 0; vk < 4; ++vk)` and directly accessing `ha[{b, m + 1, k + vk}]`, pass down the actual active repetitions or check if `m + 1` and `k + vk` fall within the `repM` and `repK` boundaries.
    *   Calculate the required bounds for `M` (e.g., `numMmaM = min(2, available_M_tiles)`) and `K` (e.g., `numMmaK = min(4, available_K_tiles)`).
    *   If the shape is exactly `8x8x4`, emit exactly *one* `mma.sync.aligned.m8n8k4...` instruction instead of 8.

### 3. Adjust Packing in `getValuesFromDotOperandLayoutStruct`
The initial data packing loops for operands `A` and `B` assume a uniform grid of `numVecM=2` and `numVecK=4` for `fp64`.

*   **File:** `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`
*   **Function:** `getValuesFromDotOperandLayoutStruct`
*   **Changes:**
    *   Currently, `numVecM = 2;` and `numVecK = bitwidth == 64 ? 4 : 2;` are hardcoded.
    *   Calculate these bounds based on the *actual* tensor dimensions (`repM` and `repK`) divided by the base hardware instruction sizes (`8` for `M`, `4` for `K`).
    *   Guard the packing loops (`for (auto vm = 0; vm < numVecM; ++vm)`) so they only pack the slices of the layout that actually exist in the `ValueTableV2`, avoiding the out-of-bounds exception that leads to the segfault.

### 4. Enable Small Block Tests
Add specific test cases targeting `8x8x4` and `8x8x8` matrix multiplications to ensure stability.

*   **File:** `python/test/unit/language/test_matmul.py` and/or `test_core.py`
*   **Changes:**
    *   In the `@pytest.mark.parametrize` matrices for `fp64`, add `BLOCK_M=8, BLOCK_N=8, BLOCK_K=4` (and `8x8x8`).
    *   Run tests explicitly enforcing exact `fp64` tile execution boundaries. 

---

# Progress Report: Implementation Complete

## Summary

All changes from the plan above have been implemented and tested on branch `smallMMA`. The fp64 MMA path now operates at native `m8n8k4` granularity, supporting any shape that is a multiple of 8×8×4, including the minimal 8×8×4 case.

## Files Changed

### `lib/Dialect/TritonGPU/IR/Dialect.cpp`
- `getRepForOperand`: Changed `tileBitWidthK` from `2 * 256` to `1 * 256` for fp64 (K-tile = 4). Changed `tileSize[M]` from hardcoded `16` to `8` for fp64.

### `lib/Dialect/TritonGPU/Transforms/Utility.cpp`
- `mmaVersionToInstrShape`: Returns `instrShape[M] = 8` for fp64 (was always 16). This ensures the MMA encoding attribute matches the native instruction shape.

### `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`
- `nvidiaDotToLinearLayout`: Uses `instrShape` from the MMA encoding for tile shape computation. K tile multiplier is 4 (not 8) when `instrM == 8`. This keeps LinearLayout data packing consistent with the changed rep computation.

### `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`
- `getMmaRetType`: fp64 returns `struct{f64, f64}` (2 elements) instead of `struct{f64, f64, f64, f64}` (4 elements).
- `callMmaAmpereFp64`: Rewritten to emit exactly one `m8n8k4` instruction per call (single retArgs(2), aArgs(1), bArgs(1), cArgs(2)).
- `numRegisters`: `{1, 1, 1}` for fp64 (was effectively `{2, 1, 2}`).
- `numMmaRets`: 2 for fp64 (was 4).
- `numCPackedElem`: 1 for fp64 (was incorrectly computed).
- fc indexing formula: Uses `numMmaRets * numCPackedElem` instead of hardcoded `4`.

### `third_party/nvidia/backend/compiler.py`
- `min_dot_size`: Added `elif lhs_bitwidth == 64: return (1, 1, 4)` to allow K=4 for fp64.

### `python/test/unit/language/test_core.py`
- Added small fp64 test cases: `(8,8,4)`, `(8,8,8)`, `(16,8,4)`, `(8,8,16)` with `num_warps=1`.

### `test/Conversion/tritongpu_to_llvm.mlir`
- Updated `f64_mma_cvt` test to use `instrShape = [8, 8]` matching the new fp64 encoding.

## Test Results

All tests pass on A100 (SM80):

- **Existing fp64 dot tests**: 60 passed, 89 skipped (all skips are for non-applicable configs like HIP or non-fp64 types)
- **New small-shape tests**: 8×8×4, 8×8×8, 16×8×4, 8×8×16 all pass
- **Larger shapes**: 16×16×16, 32×32×32, 64×64×64 all produce correct results
- Identity matrix tests and random matrix tests both verified