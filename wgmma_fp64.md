# Roadmap: WGMMA FP64 (`wgmma.mma_async.sync.aligned.m64nNk4.f64.f64.f64`) Support on Hopper

## Hardware Reality Check

The FP64 `wgmma.mma_async` instruction on H100 has these fixed constraints (different from the `mma.sync` used on A100):

*   **M = 64** (always; the warp group collectively covers 64 rows)
*   **N = 8, 16, 24, …, 256** (multiples of 8)
*   **K = 4** (always; 4 FP64 elements per warp group K-slice)
*   **Memory:** Both A and B must come from shared memory (no register-A path; this is unlike FP16/BF16 where A can be in registers)
*   **No Transpose:** Modifiers (`trans-a`, `trans-b`) are not supported.
*   **No Scaling:** No `imm-scale-a`/`imm-scale-b` modifiers (those are only for float-type WGMMA with FP16/BF16/TF32/FP8).
*   **Accumulator:** Each thread holds $2 \times (N/8)$ FP64 registers.

> **Note:** This means the minimum WGMMA FP64 tile is $64 \times 8 \times 4$, and M must always be a multiple of 64.

---

## Touch Points (7 files)

### 1. `third_party/nvidia/include/Dialect/NVGPU/IR/NVGPUOps.td`
**Add f64 enum case**

`WGMMA_EltTypeAttr` currently lists: `s8`, `s32`, `e4m3`, `e5m2`, `f16`, `bf16`, `tf32`, `f32`. Add:

```tablegen
I32EnumAttrCase<"f64", 8>
```

Also update the doc string. This makes `WGMMAEltType::f64` available throughout the `NVGPU` dialect.

---

### 2. `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp`
**PTX emission for FP64 wgmma**

This is where the PTX string `wgmma.mma_async.sync.aligned.m64nNk4.f64.f64.f64 ...` is assembled.

*   Change `assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0)` to allow `k=4` for `f64`.
*   Validate shapes only for currently-known types; add the FP64 case:

```cpp
supported |= (eltTypeA == f64) && (eltTypeB == f64) && (eltTypeC == f64) &&
             (m == 64 && 8 <= n && n <= 256 && k == 4);
```

*   Set `floatTypeWGMMA = false` for FP64 (no `imm-scale-a/b` modifiers).
*   Set `needTransArgs = false` for FP64 (no transpose support).
*   The `scale-d` argument (`useC`) is still needed.

**Important:** The register constraint for operand A changes. For other types, if A is in registers, `structTypeA` is an `LLVM::LLVMStructType`. For FP64, A must always be a shared memory descriptor (`i64`), so `structTypeA` will always be null.

---

### 3. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp`
**Core WGMMA lowering**

Five sub-changes:

*   **a. `getMmaRetType`:** Add `dTy.isF64() -> WGMMAEltType::f64`.
*   **b. `getMmaOperandType`:** Add `aTy.isF64() -> WGMMAEltType::f64`.
*   **c. `accSize` calculation:** `accSize = instrMNK[1] / 4` (N/4 f64 values per thread). Audit this to ensure it matches `getElemsPerThread` for the FP64 accumulator tensor.
*   **d. Disable Register-A path:** For FP64, operand A must be in shared memory. Add an assertion: `if (!aInShared && aTy.isF64()) return failure();`.
*   **e. `regASize` formula:** Block the register-A path entirely for FP64 to avoid incorrect `mov.b32` generation.

---

### 4. `lib/Analysis/Utility.cpp`
**`supportMMA(DotOp, version=3)` gate**

The current `version==3` check excludes FP64. Add `aElemTy.isF64()` to the allowed list:

```cpp
if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 64 == 0 &&
      retShapePerCTA[rank - 1] % 8 == 0 &&   // Relaxed from % 16 to % 8 for FP64
      (... || aElemTy.isF64()))) {
  return false;
}
```

The existing `if (k < 256 / bitwidth)` check remains correct as $256 / 64 = 4$.

---

### 5. `lib/Dialect/TritonGPU/Transforms/Utility.cpp`
**`mmaVersionToInstrShape` for version 3**

Ensure `version==3` produces a valid `instrShape` for FP64:

```cpp
if (eltType.isF64()) {
  // N must be a multiple of 8, up to 256
  validN.assign({256, 248, 240, ..., 8});  
  k = 4; 
}
```

The warp-tiling $M=16$ per warp (64 total for 4 warps) remains constant across dtypes.

---

### 6. `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`
**`BlockedToMMA` rewrite**

*   **a. Remove SM90 F64 exclusion:** Remove the guard blocking FP64 from reaching version 3 on SM90.
*   **b. `allowTranspose = false`:** Ensure FP64 does not attempt transposes in the version 3 path.
*   **c. Force SMEM for A:** 
    ```cpp
    if (!aFromLoad && eltType.isF64()) {
        // FP64 wgmma requires A in shared memory
        return failure(); 
    }
    ```

---

### 7. `third_party/nvidia/backend/compiler.py`
**`min_dot_size` for WGMMA FP64**

Update the minimum constraints for SM90:

```python
elif lhs_bitwidth == 64:
    return (1, 1, 4)  # K constraint; M=64 is enforced by the MMA pass
```

---

## Additional Concerns

### Shared Memory Swizzling
The `NVMMASharedEncodingAttr` and `getCoreMatrixLinearLayout` are tuned for bitwidths $\le 32$. For 64-bit elements:
*   A core matrix tile is $8 \text{ rows} \times 4 \text{ cols} = 256$ bytes.
*   `getDescriptor()` in `MMAHelpers.h` needs verification to ensure swizzling doesn't cause assertion failures when `swizzling < bitwidth`.

### Accumulator Register Count
Per PTX spec, each thread holds $2 \times (N/8)$ `f64` registers. 
*   $N=8 \rightarrow 2$ `f64` values per thread.
*   $N=128 \rightarrow 32$ `f64` values per thread.
This matches the `accSize` logic if audited carefully for 64-bit element sizes.

### WGMMA Operand Type
`WGMMA_OperandType` (`NVGPUOps.td`) is `AnyTypeOf<[LLVM_AnyStruct, I64]>`. Since FP64 operands arrive as descriptors (`i64`), this is already correct.

---

## Testing Plan

1.  **Lit Test:** Create `test/TritonNvidiaGPU/wgmma_fp64.mlir`. Verify that WGMMA FP64 ops with `instrShape [64, N, 4]` lower to the correct PTX instruction.
2.  **Python E2E Test:** Update `python/test/unit/language/test_core.py`. Add FP64 `dot` test cases with `num_warps=4` and $M=64$, covering shapes like $(64, 8, 4)$, $(64, 64, 4)$, and $(128, 64, 8)$ on SM90 hardware.


## Progress 



Critical finding: the roadmap in wgmma_fp64.md is based on a false premise. The PTX instruction wgmma.mma_async.sync.aligned.m64nNk{4,16}.f64.f64.f64 does not exist. wgmma.mma_async is not defined for .f64 element types in PTX ISA.

What I verified
I implemented the roadmap end‑to‑end (all 7 touchpoints) and got compilation as far as PTX emission. The Triton-emitted PTX is structurally what the doc described:

wgmma.mma_async.sync.aligned.m64n64k4.f64.f64.f64 {%rd...32...}, %rd60, %rd61, %p8;
ptxas 12.9 rejects it: Unknown modifier '.m64n64k4' and Unexpected instruction types specified for 'wgmma.mma_async'.
I tried hand-written PTX with .f64 registers (%fd) and tried k=4, k=8, k=16, with/without imm-scale-a,b and trans args, with N=8 and N=64. Every variant fails with Unexpected instruction types. Only k=16 gets past the modifier parse — meaning mNk16 is the only recognized shape — but the .f64 dtype combo is still rejected.
CUTLASS's mma_sm90_gmma.hpp contains 646 wgmma instances; none reference f64.
NVIDIA's PTX ISA docs do not list .f64 among wgmma.mma_async supported dtypes (they list f16/bf16/tf32/fp8/int8).
Conclusion
FP64 matmul on H100 uses the legacy synchronous mma.sync (MMA v2) instruction — which the codebase already supports and the benchmark already exercises (the ~31 TFLOPS in fp64_H100.txt is the v2 path running on SM90). There is no v3/async FP64 path on Hopper to enable.

State of the working tree
The implementation changes are still in place on wgmma_fp64 branch:

NVGPUOps.td — added f64 enum case
NVGPUToLLVMPass.cpp — =d constraint and f64 support gate
WGMMA.cpp — f64 in getMmaRetType/getMmaOperandType, smem-A guard
Utility.cpp (Analysis) — supportMMA v3 gate relaxed for f64 (% 8 on N)
Utility.cpp (Transforms) — validN for f64
AccelerateMatmul.cpp — f64 forced to smem-A path
compiler.py — already (1,1,4) for 64‑bit
These changes are dead code given the hardware reality. Recommend: revert these changes. Want me to revert? If improving FP64 throughput on H100 is the actual goal, the lever would be optimizing the existing MMAv2 path (better software pipelining / smem layouts) rather than enabling a v3 instruction that doesn't exist