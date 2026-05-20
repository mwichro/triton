"""Volumetric Laplace operator for p=4 (N=5) — mixed MMA + CUDA-core kernel.

Operation
---------
  L*u = (K⊗M⊗M + M⊗K⊗M + M⊗M⊗K) * u

Strategy (per concepts/MMA.md, p=4 mixed approach)
--------------------------------------------------
N=5 is not power-of-2. Triton FP64 MMA tile is M×8×4 (K=4 contracted, output=8).
Each contraction is implemented as:

  * MMA portion (tensor cores): stacked operator [M[0:4,0:4]; D[0:4,0:4]] of
    shape (8, 4), one tl.dot covers the 4×4 sub-tile of the (5,5) operator.

  * K-tail (CUDA cores): the K=5 column (index 4) handled by an outer-product
    add into the same accumulator.

  * Output-row tail (CUDA cores): the row=4 of M and D operators computed as
    a scalar-vector dot over the full K=5 input.

Phase 1 contracts i, phase 2 contracts j, phase 3 contracts k. Phases 2 and 3
follow the MMA.md "branching" pattern: phase 2a stacks [M; D] applied to ux_M
to yield (uy_MM, uy_MD); phase 2b stacks [M; D] applied to ux_D to yield uy_DM
(D side discarded). Phase 3 similarly produces z1 from u_sum and z2 from uy_MM.

Data layout: BXYZE (n_blocks, N, N, N, block_size_data).
All tl.arange sizes are powers of two (N_PAD = 8); valid data sits in [:5] of
each spatial dim, with out-of-range lanes zero on load and masked on store.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import triton
import triton.language as tl

_triton_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".triton_cache_vol")
os.environ.setdefault("TRITON_CACHE_DIR", _triton_cache_dir)


_N = 5
_N_PAD = 8
_N_MMA = 4


def math_flops(num_elements: int) -> float:
    N = _N
    return float((18 * N**4 + 2 * N**3) * num_elements)


def theory_bytes(num_elements: int, dtype_size: int = 8) -> float:
    N = _N
    return float(2 * N**3 * num_elements * dtype_size)


# ---------------------------------------------------------------------------
# Helper @triton.jit functions (inlined by compiler)
# ---------------------------------------------------------------------------

@triton.jit
def _mixed_contract(
    a_lo_flat,           # (M, 4)        leading 4 cols of input
    a_tail_flat,         # (M,)          col 4 of input
    a_full,              # (BE, A, B, 8) full input (for output-row tail), perm = same as result
    out_row4_M,          # (8,)          M[4, :], zero in [5:]
    out_row4_D,          # (8,)          D[4, :], zero in [5:]
    O_low4_T,            # (4, 8)        stacked [M[0:4,0:4]; D[0:4,0:4]] transposed
    O_col4,              # (8,)          stacked [M[0:4,4]; D[0:4,4]]
    contract_axis: tl.constexpr,         # axis in a_full to contract (3 means innermost)
    BE: tl.constexpr,
    A: tl.constexpr,                     # outer dims (powers of 2 ≥ N)
    B: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    N_MMA: tl.constexpr,
):
    """Generic stacked-operator contraction: returns (out_top, out_bot) each of
    shape (BE, A, B, N_PAD) where N_PAD's first 5 cols are valid.
    """
    # MMA: A_low (M, 4) × (4, 8) → (M, 8)
    acc = tl.dot(a_lo_flat, O_low4_T)
    # K-tail (col 4 of input) outer-product into accumulator
    acc = acc + a_tail_flat[:, None] * O_col4[None, :]
    # Reshape and split into top (M-side) and bottom (D-side) blocks of width 4.
    acc_4d = tl.reshape(acc, (BE, A, B, 2, N_MMA))
    acc_p = tl.permute(acc_4d, (0, 1, 2, 4, 3))   # (BE, A, B, 4, 2)
    out_top_lo, out_bot_lo = tl.split(acc_p)      # each (BE, A, B, 4)

    # Output-row 4: full contraction of a_full over `contract_axis` with row4 vectors.
    # We pass a_full already permuted with the contract axis last.
    out_top_hi = tl.sum(a_full * out_row4_M[None, None, None, :], axis=3)  # (BE, A, B)
    out_bot_hi = tl.sum(a_full * out_row4_D[None, None, None, :], axis=3)

    # Assemble width-8 result (cols 0..3 from MMA, col 4 from hi, 5..7 zero).
    ip_lo = tl.arange(0, N_MMA)
    is_hi = ip_lo == 0
    hi_top = tl.where(is_hi[None, None, None, :],
                      out_top_hi[:, :, :, None].broadcast_to(BE, A, B, N_MMA),
                      tl.zeros_like(out_top_lo))
    hi_bot = tl.where(is_hi[None, None, None, :],
                      out_bot_hi[:, :, :, None].broadcast_to(BE, A, B, N_MMA),
                      tl.zeros_like(out_bot_lo))
    out_top = tl.cat(out_top_lo, hi_top, dim=3)
    out_bot = tl.cat(out_bot_lo, hi_bot, dim=3)
    return out_top, out_bot


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

@triton.jit
def _laplace_kernel_mixed_p4(
    u_ptr,
    O_low4_ptr,        # (8, 4)
    O_col4_ptr,        # (8,)
    M_row4_ptr,        # (8,) — M[4, :] padded
    D_row4_ptr,        # (8,) — D[4, :] padded
    out_ptr,
    num_elements,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    N_MMA: tl.constexpr,
    BLOCK_SIZE_DATA: tl.constexpr,
    BLOCK_SIZE_EXEC: tl.constexpr,
):
    pid = tl.program_id(0)
    e_off = tl.arange(0, BLOCK_SIZE_EXEC)
    e_global = pid * BLOCK_SIZE_EXEC + e_off
    mask_e = e_global < num_elements

    block_b = e_global // BLOCK_SIZE_DATA
    e_inblock = e_global % BLOCK_SIZE_DATA
    block_base = block_b * N * N * N * BLOCK_SIZE_DATA

    p_idx = tl.arange(0, N_PAD)
    mask_p = p_idx < N
    safe_p = tl.where(mask_p, p_idx, 0)

    # Full load offsets and mask
    offs_4d = (block_base[None, None, None, :] +
               safe_p[:, None, None, None] * N * N * BLOCK_SIZE_DATA +
               safe_p[None, :, None, None] * N * BLOCK_SIZE_DATA +
               safe_p[None, None, :, None] * BLOCK_SIZE_DATA +
               e_inblock[None, None, None, :])
    spatial_mask = (mask_p[:, None, None, None]
                    & mask_p[None, :, None, None]
                    & mask_p[None, None, :, None])
    full_mask = spatial_mask & mask_e[None, None, None, :]
    u_kjie = tl.load(u_ptr + offs_4d, mask=full_mask, other=0.0)    # (8,8,8,BE)
    u_ekji = tl.permute(u_kjie, (3, 0, 1, 2))                       # (BE,8,8,8) = (BE,k,j,i)

    # Operator tiles (constant across phases — all three axes use same M, D).
    rO = tl.arange(0, 2 * N_MMA)
    cO = tl.arange(0, N_MMA)
    O_low4 = tl.load(O_low4_ptr + rO[:, None] * N_MMA + cO[None, :])
    O_low4_T = tl.trans(O_low4)
    O_col4 = tl.load(O_col4_ptr + rO)
    M_row4 = tl.load(M_row4_ptr + p_idx)
    D_row4 = tl.load(D_row4_ptr + p_idx)

    # =================================================================
    # Phase 1: contract over i. u_ekji is already (BE, k, j, i_contract).
    # =================================================================
    i_lo = tl.arange(0, N_MMA)
    offs_lo = (block_base[None, None, None, :] +
               safe_p[:, None, None, None] * N * N * BLOCK_SIZE_DATA +
               safe_p[None, :, None, None] * N * BLOCK_SIZE_DATA +
               i_lo[None, None, :, None] * BLOCK_SIZE_DATA +
               e_inblock[None, None, None, :])
    mask_lo = (mask_p[:, None, None, None]
               & mask_p[None, :, None, None]
               & mask_e[None, None, None, :])
    u_lo_kjie = tl.load(u_ptr + offs_lo, mask=mask_lo, other=0.0)
    u_lo_ekji = tl.permute(u_lo_kjie, (3, 0, 1, 2))                 # (BE,8,8,4)
    u_lo_flat = tl.reshape(u_lo_ekji, (BLOCK_SIZE_EXEC * N_PAD * N_PAD, N_MMA))

    offs_t = (block_base[None, None, :] +
              safe_p[:, None, None] * N * N * BLOCK_SIZE_DATA +
              safe_p[None, :, None] * N * BLOCK_SIZE_DATA +
              4 * BLOCK_SIZE_DATA +
              e_inblock[None, None, :])
    mask_t = mask_p[:, None, None] & mask_p[None, :, None] & mask_e[None, None, :]
    u_tail_kje = tl.load(u_ptr + offs_t, mask=mask_t, other=0.0)
    u_tail_ekj = tl.permute(u_tail_kje, (2, 0, 1))                   # (BE,8,8)
    u_tail_flat = tl.reshape(u_tail_ekj, (BLOCK_SIZE_EXEC * N_PAD * N_PAD,))

    # For Phase 1, _mixed_contract returns ux_M, ux_D each (BE, k, j, i')
    ux_M, ux_D = _mixed_contract(
        u_lo_flat, u_tail_flat, u_ekji,
        M_row4, D_row4, O_low4_T, O_col4,
        contract_axis=3,
        BE=BLOCK_SIZE_EXEC, A=N_PAD, B=N_PAD,
        N=N, N_PAD=N_PAD, N_MMA=N_MMA,
    )

    # =================================================================
    # Phase 2a: contract ux_M over j → uy_MM, uy_MD. Shape (BE, k, j, i').
    # Need j as the inner (contracted) axis: permute to (BE, k, i', j).
    # =================================================================
    ux_M_kij = tl.permute(ux_M, (0, 1, 3, 2))                       # (BE, k, i', j)
    # MMA input: leading 4 cols of j
    ux_M_lo4 = tl.reshape(
        # Slice j ∈ {0..3}: we use cat-style trick — we can't slice arbitrary;
        # but j here is the inner dim of size 8. Reshape (BE,k,i',8) → split into two halves of 4.
        ux_M_kij, (BLOCK_SIZE_EXEC, N_PAD, N_PAD, 2, N_MMA)
    )
    # halves[..., 0, :] = j∈{0..3}; halves[..., 1, :] = j∈{4..7}. j=4 is the K-tail; j∈{5..7} are zero.
    ux_M_p = tl.permute(ux_M_lo4, (0, 1, 2, 4, 3))                   # (BE, k, i', 4, 2)
    ux_M_first, ux_M_second = tl.split(ux_M_p)                       # each (BE, k, i', 4)
    ux_M_a_lo_flat = tl.reshape(ux_M_first, (BLOCK_SIZE_EXEC * N_PAD * N_PAD, N_MMA))
    # Tail (j=4) = lane 0 of the second half; extract via masked sum.
    lane0_sel = (tl.arange(0, N_MMA) == 0).to(ux_M.dtype)            # (4,)
    ux_M_tail = tl.sum(ux_M_second * lane0_sel[None, None, None, :], axis=3)
    ux_M_tail_flat = tl.reshape(ux_M_tail, (BLOCK_SIZE_EXEC * N_PAD * N_PAD,))
    # Full input for output-row tail (axis 3 = j).
    uy_MM, uy_MD = _mixed_contract(
        ux_M_a_lo_flat, ux_M_tail_flat, ux_M_kij,
        M_row4, D_row4, O_low4_T, O_col4,
        contract_axis=3,
        BE=BLOCK_SIZE_EXEC, A=N_PAD, B=N_PAD,
        N=N, N_PAD=N_PAD, N_MMA=N_MMA,
    )
    # uy_MM, uy_MD shape: (BE, k, i', j')

    # =================================================================
    # Phase 2b: contract ux_D over j → uy_DM (discard DD half).
    # =================================================================
    ux_D_kij = tl.permute(ux_D, (0, 1, 3, 2))
    ux_D_lo4 = tl.reshape(ux_D_kij, (BLOCK_SIZE_EXEC, N_PAD, N_PAD, 2, N_MMA))
    ux_D_p = tl.permute(ux_D_lo4, (0, 1, 2, 4, 3))
    ux_D_first, ux_D_second = tl.split(ux_D_p)
    ux_D_a_lo_flat = tl.reshape(ux_D_first, (BLOCK_SIZE_EXEC * N_PAD * N_PAD, N_MMA))
    ux_D_tail = tl.sum(ux_D_second * lane0_sel[None, None, None, :], axis=3)
    ux_D_tail_flat = tl.reshape(ux_D_tail, (BLOCK_SIZE_EXEC * N_PAD * N_PAD,))
    uy_DM, _uy_DD = _mixed_contract(
        ux_D_a_lo_flat, ux_D_tail_flat, ux_D_kij,
        M_row4, D_row4, O_low4_T, O_col4,
        contract_axis=3,
        BE=BLOCK_SIZE_EXEC, A=N_PAD, B=N_PAD,
        N=N, N_PAD=N_PAD, N_MMA=N_MMA,
    )

    # =================================================================
    # Phase 3: contract over k.  We have (BE, k, i', j').
    # Permute to (BE, i', j', k) to make k the inner contracted dim.
    # u_sum + uy_MM are processed in two _mixed_contract calls, but each call
    # gives M-out and D-out; we only need: from u_sum → z1_M (top); from uy_MM → z2_D (bot).
    # =================================================================
    u_sum = uy_MD + uy_DM                                            # (BE, k, i', j')

    u_sum_ijk = tl.permute(u_sum, (0, 2, 3, 1))                      # (BE, i', j', k)
    u_sum_lo4 = tl.reshape(u_sum_ijk, (BLOCK_SIZE_EXEC, N_PAD, N_PAD, 2, N_MMA))
    u_sum_p = tl.permute(u_sum_lo4, (0, 1, 2, 4, 3))
    u_sum_first, u_sum_second = tl.split(u_sum_p)
    u_sum_a_lo_flat = tl.reshape(u_sum_first, (BLOCK_SIZE_EXEC * N_PAD * N_PAD, N_MMA))
    u_sum_tail = tl.sum(u_sum_second * lane0_sel[None, None, None, :], axis=3)
    u_sum_tail_flat = tl.reshape(u_sum_tail, (BLOCK_SIZE_EXEC * N_PAD * N_PAD,))
    z1_M, _z1_D = _mixed_contract(
        u_sum_a_lo_flat, u_sum_tail_flat, u_sum_ijk,
        M_row4, D_row4, O_low4_T, O_col4,
        contract_axis=3,
        BE=BLOCK_SIZE_EXEC, A=N_PAD, B=N_PAD,
        N=N, N_PAD=N_PAD, N_MMA=N_MMA,
    )

    uy_MM_ijk = tl.permute(uy_MM, (0, 2, 3, 1))
    uy_MM_lo4 = tl.reshape(uy_MM_ijk, (BLOCK_SIZE_EXEC, N_PAD, N_PAD, 2, N_MMA))
    uy_MM_p = tl.permute(uy_MM_lo4, (0, 1, 2, 4, 3))
    uy_MM_first, uy_MM_second = tl.split(uy_MM_p)
    uy_MM_a_lo_flat = tl.reshape(uy_MM_first, (BLOCK_SIZE_EXEC * N_PAD * N_PAD, N_MMA))
    uy_MM_tail = tl.sum(uy_MM_second * lane0_sel[None, None, None, :], axis=3)
    uy_MM_tail_flat = tl.reshape(uy_MM_tail, (BLOCK_SIZE_EXEC * N_PAD * N_PAD,))
    _z2_M, z2_D = _mixed_contract(
        uy_MM_a_lo_flat, uy_MM_tail_flat, uy_MM_ijk,
        M_row4, D_row4, O_low4_T, O_col4,
        contract_axis=3,
        BE=BLOCK_SIZE_EXEC, A=N_PAD, B=N_PAD,
        N=N, N_PAD=N_PAD, N_MMA=N_MMA,
    )

    # z1_M, z2_D shape (BE, i', j', k'). Sum and permute to BXYZE store order.
    res_eijk = z1_M + z2_D                                           # (BE, i', j', k')
    # Want (k, j, i, BE) for the BXYZE store using offs_4d (k,j,i,e).
    res_kjie = tl.permute(res_eijk, (3, 2, 1, 0))
    tl.store(out_ptr + offs_4d, res_kjie, mask=full_mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def _laplace_apply_mixed(u, stiffness_1d, mass_1d, block_size_data, block_size_exec):
    n_blocks = u.shape[0]
    N = u.shape[1]
    assert N == _N
    num_elements = n_blocks * block_size_data
    grid = num_elements // block_size_exec

    O_low4 = torch.empty(2 * _N_MMA, _N_MMA, dtype=u.dtype, device=u.device)
    O_low4[:_N_MMA, :] = mass_1d[:_N_MMA, :_N_MMA]
    O_low4[_N_MMA:, :] = stiffness_1d[:_N_MMA, :_N_MMA]
    O_col4 = torch.empty(2 * _N_MMA, dtype=u.dtype, device=u.device)
    O_col4[:_N_MMA] = mass_1d[:_N_MMA, 4]
    O_col4[_N_MMA:] = stiffness_1d[:_N_MMA, 4]
    M_row4 = torch.zeros(_N_PAD, dtype=u.dtype, device=u.device)
    D_row4 = torch.zeros(_N_PAD, dtype=u.dtype, device=u.device)
    M_row4[:N] = mass_1d[4, :]
    D_row4[:N] = stiffness_1d[4, :]

    out = torch.empty_like(u)
    _laplace_kernel_mixed_p4[(grid,)](
        u, O_low4, O_col4, M_row4, D_row4, out,
        num_elements,
        N=_N,
        N_PAD=_N_PAD,
        N_MMA=_N_MMA,
        BLOCK_SIZE_DATA=block_size_data,
        BLOCK_SIZE_EXEC=block_size_exec,
    )
    return out


def gpu_timeit(fn, *, warmup=10, iters=100, pipeline_warmup=50):
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(stream)
    for _ in range(pipeline_warmup):
        fn()
    torch.cuda.synchronize(stream)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        fn()
    end.record(stream)
    torch.cuda.synchronize(stream)
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--log2-elements", type=int, default=20)
    parser.add_argument("--data-block", type=int, default=8)
    parser.add_argument("--exec-block", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    N = _N
    P = 4
    num_elements = 2 ** args.log2_elements
    BD, BE = args.data_block, args.exec_block
    if num_elements % BD != 0:
        raise ValueError(f"num_elements={num_elements} not divisible by data-block={BD}")
    n_blocks = num_elements // BD

    device = torch.cuda.current_device()
    print(f"Device:      {torch.cuda.get_device_name(device)}")
    print(f"P={P} N={N}  E=2^{args.log2_elements}={num_elements}")
    print(f"data_block={BD}  exec_block={BE}  n_blocks={n_blocks}  kernel=mixed_p4")
    print()
    mfl = math_flops(num_elements)
    tb = theory_bytes(num_elements)
    print(f"Math FLOPs: {mfl:.3e}")
    print(f"Theory DRAM: {tb:.3e} bytes ({tb/1e9:.3f} GB)")
    print()

    u = torch.randn(n_blocks, N, N, N, BD, dtype=torch.float64, device="cuda")
    stiffness = torch.randn(N, N, dtype=torch.float64, device="cuda")
    mass = torch.randn(N, N, dtype=torch.float64, device="cuda")

    t0 = time.perf_counter()
    _laplace_apply_mixed(u, stiffness, mass, BD, BE)
    torch.cuda.synchronize()
    print(f"Compilation time: {(time.perf_counter()-t0)*1000:.1f} ms")

    if args.verify:
        out_triton = _laplace_apply_mixed(u, stiffness, mass, BD, BE)
        u_ref = u.permute(0, 4, 1, 2, 3).reshape(num_elements, N, N, N).contiguous()
        out_ref = out_triton.permute(0, 4, 1, 2, 3).reshape(num_elements, N, N, N).contiguous()
        expected = (
            torch.einsum("kK,jJ,iI,eKJI->ekji", stiffness, mass, mass, u_ref) +
            torch.einsum("kK,jJ,iI,eKJI->ekji", mass, stiffness, mass, u_ref) +
            torch.einsum("kK,jJ,iI,eKJI->ekji", mass, mass, stiffness, u_ref)
        )
        max_err = float(torch.max(torch.abs(out_ref - expected)))
        ref_scale = float(torch.max(torch.abs(expected)))
        rel_err = max_err / ref_scale
        status = "PASS" if rel_err < 1e-10 else "FAIL"
        print(f"verify: max_abs_err={max_err:.3e}  rel_err={rel_err:.3e}  [{status}]")
        if status == "FAIL":
            raise SystemExit(1)

    ms = gpu_timeit(lambda: _laplace_apply_mixed(u, stiffness, mass, BD, BE),
                    warmup=args.warmup, iters=args.iters)
    tflops_math = mfl / (ms * 1e-3) / 1e12
    bw = tb / (ms * 1e-3) / 1e9
    dofs = N**3 * num_elements
    gdofs_s = dofs / (ms * 1e-3) / 1e9
    print(f"Timing ({args.iters} iters): avg = {ms:.3f} ms")
    print(f"  TFLOPS (math):     {tflops_math:.3f}")
    print(f"  Bandwidth:         {bw:.1f} GB/s")
    print(f"  Throughput:        {gdofs_s:.3f} GDoF/s")


if __name__ == "__main__":
    main()
