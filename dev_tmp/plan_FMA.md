

## Analysis: Avoiding Intermediate Tensor Materialization

### Root Cause

The Python code in question generates the following Triton IR chain:

#### Code

```mlir
%u_exp   = tt.expand_dims %u {axis=3}           : (BE,A,B,NP)      → (BE,A,B,1,NP)
%u_bcast = tt.broadcast %u_exp                   : (BE,A,B,1,NP)   → (BE,A,B,2NP,NP)
%o_bcast = tt.broadcast (reshape %O_MD)          : (1,1,1,2NP,NP)  → (BE,A,B,2NP,NP)
%prod    = arith.mulf %u_bcast, %o_bcast         : (BE,A,B,2NP,NP)
%res     = tt.reduce %prod {axis=4, addf}        : → (BE,A,B,2NP)
```

Every element of `%prod` — a `(BE × A × B × 2·NP × NP)` tensor — must be live in registers simultaneously before the reduce can consume them. For even modest values (e.g., `BE=A=B=8, NP=5`) that is 25,600 `f64` values ≈ 200 KB, far exceeding the A100 register file → register spill → memory traffic.

This is exactly the `sum(x[:, :, None] * y[None, :, :], 1) → dot(x, y)` pattern that `CombineBroadcastMulReducePattern` in `lib/Dialect/Triton/Transforms/Combine.cpp` was designed to handle. But the pattern has four hard-wired restrictions that block it for the user's case, listed below with the fixes needed.

---

## Required Changes

### Change 1 — Lift the "fp32 only" restriction in `CombineBroadcastMulReducePattern`

`Combine.cpp` lines 113–134 gate the entire pattern on:

#### C++

```cpp
static bool isAddF32(const Operation *op) {
  if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
    return addf.getType().getIntOrFloatBitWidth() <= 32;
  return false;
}
...
bool isReduceAdd = ... && isAddF32(&*combineOp...);
if (!isReduceAdd) return failure();
```

This silently rejects every `fp64` (and `fp16`, `bf16`) reduction. The check should be replaced with one that accepts any float `arith::AddFOp` (and optionally integer `arith::AddIOp` for completeness).

### Change 2 — Lift the "shape ≥ 16" guard

`Combine.cpp` lines 163–164:

#### C++

```cpp
if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
    return failure();
```

This was put in to avoid feeding tiny matrices into MMA (which would pad to 16 and waste compute). But:

*   The FMA dot lowering (`FMADotUtility.cpp`) works for any size — it is a pure scalar FFMA loop over the K dimension with no alignment requirement.
*   The decision to use MMA vs FMA is made later, in the `AccelerateMatmul` pass, not here.

The guard should be removed entirely. Let the downstream pass decide the lowering strategy.

### Change 3 — Generalise the shape/axis checks to handle batch dimensions and different reduce axes

`Combine.cpp` lines 155–162:

#### C++

```cpp
int expandLhsAxis = expandLhsOp.getAxis();
int expandRhsAxis = expandRhsOp.getAxis();
if (expandLhsAxis != 2 || expandRhsAxis != 0)
    return failure();
```

This hard-codes the 3-D pattern: LHS is `(M, K)` expanded on axis 2 → `(M, K, 1)`, RHS is `(K, N)` expanded on axis 0 → `(1, K, N)`, reduce on axis 1.

The user's 5-D pattern has:

*   LHS `u`: `(BE, A, B, NP)` expanded on axis 3 → `(BE, A, B, 1, NP)` broadcast to `(BE, A, B, 2NP, NP)`
*   RHS `O_MD`: `(2NP, NP)` — shape `(N, K)` — broadcast over the three batch dims to `(BE, A, B, 2NP, NP)`
*   Reduce over the last axis (axis 4 = K = NP)

The generalised detection logic must:

1.  Identify the reduce axis `R` from the `tt.reduce` op (not assume it is axis 1).
2.  For the LHS: verify the `expand_dims` was on axis `R-1` (i.e., the LHS had 1 inserted just before the K dim). The resulting pre-expand shape is `(batch..., N, K)` — wait, actually for the user case expand is on axis rank-2 before the K dim — specifically axis 3 in the 4D `(BE,A,B,NP)` tensor, yielding `(BE,A,B,1,NP)`. The "N" dimension in this case comes from the other operand (`2NP`), not from `u`. So the correct detection is: after stripping the `expand_dims`, LHS has shape `(batch..., K)` and the broadcast added an `N`-sized dim between the batch dims and `K`.
3.  For the RHS: the broadcast source (after tracing through reshape/broadcast) is a 2-D tensor of shape `(N, K)` — i.e., the contraction matrix is stored transposed relative to what `tt.dot` expects.
4.  The batch dims are any leading dimensions that are identical on both sides and are not the `N` or `K` dim.

### Change 4 — Emit `tt.reshape` (to flatten batch dims) and `tt.trans` (for the transposed weight)

`tt.dot` is at most 3-D: `(batch, M, K) × (batch, K, N) → (batch, M, N)`. The generalised rewrite must:

1.  Flatten the multiple batch dims of `u` into one: `reshape (BE, A, B, NP) → (BE·A·B, NP)`. This is `M = BE·A·B, K = NP`.
2.  Transpose `O_MD` from `(N, K) = (2NP, NP)` to `(K, N) = (NP, 2NP)` using `tt.trans`.
3.  Emit `tt.dot(u_flat, O_MD_T, zeros)` of shape `(BE·A·B, NP) × (NP, 2NP) → (BE·A·B, 2NP)`.
4.  Reshape the result back to `(BE, A, B, 2NP)`.

With this transformation the `(BE, A, B, 2NP, NP)` intermediate tensor never exists in the IR, and the FMA dot lowering in `FMADotUtility.cpp` will generate a tight scalar FFMA loop over `K=NP`, accumulating directly into the `(BE·A·B, 2NP)` result — exactly what is needed.

### Change 5 (secondary) — Let small F64 dots fall through to FMA in `AccelerateMatmul`

`AccelerateMatmul.cpp` lines 404–411 enable F64 MMA on SM80 (A100) for any F64 dot:

#### C++

```cpp
if ((... isF64 ...) && !(computeCapability == 80 || computeCapability == 90))
    return failure();   // ← falls back to FMA only off SM80/SM90
```

On A100, a 5×5 F64 dot will be given an MMA v2 layout (8×8×4 tile), wasting 39 of the 64 compute slots. To make the 5×5 case use FFMA, a shape guard is needed:

#### C++

```cpp
// Only apply F64 MMA when the shapes are multiples of the MMA tile (8×8).
// For smaller or irregular shapes, let FMA handle it.
if (isF64 && (M % 8 != 0 || N % 8 != 0))
    return failure();
```

This does not affect the `p=7 (8×8)` case, which will still get optimal MMA; it causes the `p=4 (5×5)` case to keep `BlockedEncoding` and use the FMA lowering path.

---

## Summary of Files to Change

| File | Change |
| :--- | :--- |
| `lib/Dialect/Triton/Transforms/Combine.cpp` | Generalise `CombineBroadcastMulReducePattern`: lift `fp32`-only and ≥16 size restrictions; generalise axis/rank detection; emit `tt.reshape` + `tt.trans` + `tt.dot` for batched, transposed-weight patterns |
| `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` | Add shape guard for `F64` MMA: only use MMA when `M` and `N` are multiples of the 8×8 `F64` MMA tile size, so irregular shapes fall back to the FMA path |

The FMA lowering itself (`FMADotUtility.cpp`) already does the right thing — a compile-time scalar FFMA loop over `K` — so no changes are needed there. The entire problem is upstream: the broadcast-mul-reduce pattern is not being converted to `tt.dot` in the first place.