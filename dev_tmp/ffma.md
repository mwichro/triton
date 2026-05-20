# Triton outlook: making sum-factorization at odd-N fast

**Context.** Working on the volumetric Laplace operator for p=4 (N=5) in Triton
on A100 (FP64). Two kernels exist in `bench/volume/`:

| Kernel | Path | Approach | GDoF/s |
|---|---|---|---|
| MMA 8×4 packing | `triton_p4.py` | `tl.dot` with packed `(8,4)` operator + K=4 FMA tail | ~22 |
| CUDA-core sum | `triton_p4_cuda.py` | `tl.sum(broadcast_mul(op, u))` | **0.07** |





For comparison, the same kernel structure at p=3 (N=4) reaches 97 GDoF/s;
at p=7 (N=8) reaches 78. The N=5 case sits at 22 because 
**every Triton tensor axis must be a power of 2**, so spatial
dims get padded 5→8 (2.56× M-axis compute waste) and the K=5 contraction
either gets padded to K=8 or uses the 8×4 packing trick (extra peel
overhead + bookkeeping). Neither path is close to memory-bound.

The variant was an attempt to bypass MMA entirely and use the
register-blocked sum-factorization pattern from the deal.II / CUDA-C++
template (one thread per (row,col), `pval[z]` register accumulator). It
catastrophically underperformed for reasons explored below.

The core of that can be shown on the example provided below:



```python
@triton.jit
def _contract_stacked(
    u,           # (BE, A, B, N_PAD)  contract over last dim
    O_MD,        # (2*N_PAD, N_PAD)   stacked [M; D]
    BE: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    N_PAD: tl.constexpr,
):
    """Returns (out_M, out_D) each (BE, A, B, N_PAD).

    out[..., n] = sum_k O[n, k] * u[..., k]
    """
    # Broadcast-multiply + reduce. Triton  **does not** fuse this into a register-blocked
    # FFMA loop (so the full product tensor is materialized!).
    prod = u[:, :, :, None, :] * O_MD[None, None, None, :, :]   # (BE, A, B, 2*NP, NP)
    res = tl.sum(prod, axis=4)                                  # (BE, A, B, 2*NP)
    res_2d = tl.reshape(res, (BE, A, B, 2, N_PAD))              # (BE, A, B, type, n)
    res_p  = tl.permute(res_2d, (0, 1, 2, 4, 3))                # (BE, A, B, n, type)
    out_M, out_D = tl.split(res_p)                              # each (BE, A, B, N_PAD)
    return out_M, out_D
```

or a simpler one, that suffers from exatly the same problem.

```python
@triton.jit
def _contract_single(
    u,           # (BE, A, B, N_PAD)
    O,           # (N_PAD, N_PAD)
    BE: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    N_PAD: tl.constexpr,
):
    """out[..., n] = sum_k O[n, k] * u[..., k]. Returns (BE, A, B, N_PAD)."""
    prod = u[:, :, :, None, :] * O[None, None, None, :, :]
    return tl.sum(prod, axis=4)
```


---

## Root cause of the slow `tl.sum(a*b)` path

Three orthogonal problems stack on top of each other:

### 1. `tt.dot` is a first-class IR op; `sum(mul(broadcast, broadcast))` is not

`tl.dot` lowers through a dedicated pipeline (`TritonGPUDialect` →
`TritonGPUToLLVM`) that:
- Picks `mma`-layout for the output and `dot_operand`-layout for the inputs.
- Inserts swizzled `ldmatrix` / `cp.async.bulk` for shared-memory loads.
- Software-pipelines HBM→shared→MMA via the `num_stages` knob.
- Encodes the operand-reuse pattern inside the MMA instruction itself
  (each A element is broadcast across the n-direction, each B element
  across the m-direction).

`tl.sum(a[..., None, :] * b[None, ..., :], axis=-1)` looks like two
unrelated ops to the compiler. There is **no canonicalization pass that
recognizes this as a matmul** and reroutes it through the dot pipeline.
The generic elementwise+reduce path uses the default "blocked" register
layout, has no MMA awareness, no swizzling, and no software pipelining.

### 2. Materialization of the broadcasted product

The IR literally constructs the rank-(r+1) product tensor before reducing.
For our case shape `(BE, 8, 8, 16, 8) = 32768` FP64 elements = 256 KB. A100
has a 256 KB register file *per SM*; per-thread that's 256 regs = 2 KB FP64.
With ~128 threads/CTA you have ~256 KB total — right at the edge.

When it overflows, intermediates **spill to local memory** (HBM-backed). A
single spilling reduction can produce a 100× slowdown. The 224 ms timing on
`triton_p4_cuda.py` is almost certainly mostly spill-fill traffic.

The 135 s compile time is the second symptom: the register allocator and
layout planner are wrestling with enormous tensors they were never
designed to handle.

### 3. No software pipelining on the elementwise+reduce path

Even without spilling, the reduce path runs synchronously with the HBM
load — no `cp.async` overlap, no multi-stage pipelining. With `tl.dot` and
`num_stages=3` you get HBM↔shared overlapped with MMA. The reduce path
doesn't.