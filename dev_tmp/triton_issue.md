# MMA→Blocked layout convert inserted for `dot + broadcast*broadcast` even when result feeds another `tl.dot`

## Summary

When a `tl.dot` result is consumed by `acc + x[:, None] * y[None, :]` and the result later feeds another `tl.dot` accumulator chain, the compiler inserts an `MMA → Blocked` `convert_layout` (round-trip through shared memory + `bar.sync`) at the `addf`. This happens even though every op in the chain (`broadcast`, `mulf`, `addf`, `reshape`, `permute`, `split`) is layout-realizable under the producer's `NvidiaMmaEncodingAttr` (broadcast axes do not collide with the MMA K-axis).

In a real F64 finite-element kernel (24 such patterns per CTA), this is the dominant bottleneck: 65.7% `stall_long_scoreboard` + 26.4% `stall_barrier`, with the FP64+TC pipe 70% idle and 0 B register spill.

## Reproducer

Minimal standalone script attached: `repro_mma_bcast_add.py` (Triton + PyTorch only).

Two kernels with identical HBM traffic and an MMA-tile-aligned shape (M=128, K=4, N=8 — exactly fits the F64 8×8×4 tile, no padding):

- **Kernel A** — with the pattern: `acc = tl.dot(a, b); acc = acc + x[:, None] * y[None, :]; reshape/permute/split; tl.dot(top, c)` in a `tl.static_range(CHAIN)` loop.
- **Kernel B** — same chain without the `acc + x[:, None] * y[None, :]` line (tensor sinks prevent DCE of `x`, `y`).

Result on A100, F64, CHAIN=16, NUM_CTAS=2048:

| Kernel | μs | TFLOPS | `convert_layout` count |
|---|---|---|---|
| A (with bcast-mul-add) | 42.7 | 14.33 | 34 |
| B (baseline)           | 36.5 | 14.92 | 18 |

**Δ = +16 `convert_layout` (one per CHAIN step, linear in CHAIN — confirmed at CHAIN=32), 1.17× slowdown.**

## What the IR looks like

```mlir
%dot  = tt.dot ... : tensor<MxN2, #mma>
%bc1  = tt.broadcast %x : tensor<Mx1, #blocked> → tensor<MxN2, #blocked>
%bc2  = tt.broadcast %y : tensor<1xN2, #blocked> → tensor<MxN2, #blocked>
%mul  = arith.mulf %bc1, %bc2
%cvt  = ttg.convert_layout %dot : #mma → #blocked       // ← inserted here
%add  = arith.addf %cvt, %mul
// ... reshape / permute / split, all in #blocked
%use  = tt.dot ... %top : (..., #blocked) → (..., #mma) // → another DotOperand convert
```

The forward-propagation infrastructure in [`RemoveLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) already handles all of these ops (Elementwise / `ReshapeOp` / `TransOp` / `SplitOp` / `BroadcastOp`) and already prefers `MmaEncodingTrait` over Blocked in conflict resolution. Two heuristics appear to block propagation in this case:

1. The permuting-reshape anchor at `RemoveLayoutConversions.cpp:216-217` (`reshape.getAllowReorder()`) stops forward propagation at the first reshape after the `addf`.
2. The "cheap to convert below broadcast" handling at `RemoveLayoutConversions.cpp:1299` / `:1377` prefers to convert *down* through the broadcast (toward the small load) rather than carry MMA *up* through the broadcast into the `mulf`/`addf`.

## Real-world impact

In an F64 sum-factorization Laplacian kernel (`L u = (K⊗M⊗M + M⊗K⊗M + M⊗M⊗K) u`, p=4, N=5), this pattern appears 24× per CTA. Kernel runs at **2.23 TFLOPS / 24.3 GDoF/s** on A100 with profile signature:

- `stall_long_scoreboard` 65.7%, `stall_barrier` 26.4% — smem-convert round-trips
- `shared_pipe_pct` (FP64+TC) 30% — TC pipe 70% idle
- `selected_pct` 19.4% — warps issue 1 cycle in 5
- register spill 0 B, HBM SoL 19% — not register/memory bound

Removing the converts should let the TC pipe approach ~60% utilization, projecting ~33–38 GDoF/s (~1.5×).

## Question

Is the "MMA encoding can be propagated through `addf(dot, mulf(broadcast, broadcast))` when broadcast axes don't collide with the MMA K-axis" rewrite something the project would accept upstream, or are the two heuristics above intentionally load-bearing for attention/FFN/bias-add patterns? Happy to prototype if there's interest, but wanted to surface the pattern + reproducer first since touching either heuristic has high cross-kernel blast radius.

## Environment

- Triton: (fill in commit hash)
- GPU: NVIDIA A100
- CUDA: (fill in)
- PyTorch: (fill in)
