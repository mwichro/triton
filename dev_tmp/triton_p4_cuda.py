"""Volumetric Laplace operator for p=4 (N=5) — CUDA-core sum-factorization.

Mirrors the per-thread register-blocked CUDA pattern (pval[z] accumulator,
no tensor cores). In Triton terms, every contraction is expressed as

    out[..., n] = sum_k op[n, k] * u[..., k]

via broadcast-multiply + tl.sum (which Triton lowers to FFMA loops, not MMA).

Layout: BXYZE (n_blocks, 5, 5, 5, block_size_data), N_PAD=8 spatial padding.
Stacked [mass; stiff] operator (16, 8) so each contraction yields both M and
D outputs in one reduction, mirroring triton_BXYZE.py's batched structure.

NOTE: padding cost. The reduction runs over the padded K=8 instead of the
useful K=5, and over M = 8×8 = 64 instead of 25. Total compute waste vs the
ideal CUDA C++ template is ~6.4×. This is structural to Triton's
power-of-two `tl.arange` requirement.
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


def _pad_op(A: torch.Tensor) -> torch.Tensor:
    P = torch.zeros(_N_PAD, _N_PAD, dtype=A.dtype, device=A.device)
    P[0:_N, 0:_N] = A
    return P


def _stacked_MD(mass: torch.Tensor, stiff: torch.Tensor) -> torch.Tensor:
    """(2*NP, NP) padded stack of [mass; stiff]."""
    return torch.cat([_pad_op(mass), _pad_op(stiff)], dim=0).contiguous()


def math_flops(num_elements: int) -> float:
    N = _N
    return float((18 * N**4 + 2 * N**3) * num_elements)


def theory_bytes(num_elements: int, dtype_size: int = 8) -> float:
    N = _N
    return float(2 * N**3 * num_elements * dtype_size)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

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
    # Broadcast-multiply + reduce. Triton fuses this into a register-blocked
    # FFMA loop (no MMA, no materialization of the full product tensor).
    prod = u[:, :, :, None, :] * O_MD[None, None, None, :, :]   # (BE, A, B, 2*NP, NP)
    res = tl.sum(prod, axis=4)                                  # (BE, A, B, 2*NP)
    res_2d = tl.reshape(res, (BE, A, B, 2, N_PAD))              # (BE, A, B, type, n)
    res_p  = tl.permute(res_2d, (0, 1, 2, 4, 3))                # (BE, A, B, n, type)
    out_M, out_D = tl.split(res_p)                              # each (BE, A, B, N_PAD)
    return out_M, out_D


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


@triton.jit
def _laplace_kernel_p4_cuda(
    u_ptr,
    O_MD_ptr,            # (16, 8) stacked
    M_ptr,               # (8, 8)  padded mass
    D_ptr,               # (8, 8)  padded stiff
    out_ptr,
    num_elements,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
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

    # Operators (constant across phases).
    r2 = tl.arange(0, 2 * N_PAD)
    O_MD = tl.load(O_MD_ptr + r2[:, None] * N_PAD + p_idx[None, :])           # (16, 8)
    M_op = tl.load(M_ptr + p_idx[:, None] * N_PAD + p_idx[None, :])           # (8, 8)
    D_op = tl.load(D_ptr + p_idx[:, None] * N_PAD + p_idx[None, :])           # (8, 8)

    # Load u with BXYZE → permute to (BE, k, j, i) with E outermost (good for smem layout).
    offs_4d = (block_base[None, None, None, :] +
               safe_p[:, None, None, None] * N * N * BLOCK_SIZE_DATA +
               safe_p[None, :, None, None] * N * BLOCK_SIZE_DATA +
               safe_p[None, None, :, None] * BLOCK_SIZE_DATA +
               e_inblock[None, None, None, :])
    full_mask = (mask_p[:, None, None, None]
                 & mask_p[None, :, None, None]
                 & mask_p[None, None, :, None]
                 & mask_e[None, None, None, :])
    u_kjie = tl.load(u_ptr + offs_4d, mask=full_mask, other=0.0)              # (NP,NP,NP,BE)
    u_ekji = tl.permute(u_kjie, (3, 0, 1, 2))                                 # (BE,k,j,i)

    # ===== PHASE 1 — contract over i (last dim) =====
    ux_M, ux_D = _contract_stacked(
        u_ekji, O_MD, BLOCK_SIZE_EXEC, N_PAD, N_PAD, N_PAD)                   # (BE,k,j,i')
    # Permute (BE,k,j,i') → (BE,k,i',j) so next contraction is over j (last).
    ux_M_kij = tl.permute(ux_M, (0, 1, 3, 2))
    ux_D_kij = tl.permute(ux_D, (0, 1, 3, 2))

    # ===== PHASE 2a — ux_M contracted over j → uy_MM, uy_MD =====
    uy_MM_kij, uy_MD_kij = _contract_stacked(
        ux_M_kij, O_MD, BLOCK_SIZE_EXEC, N_PAD, N_PAD, N_PAD)                 # (BE,k,i',j')

    # ===== PHASE 2b — ux_D contracted over j → uy_DM (DD discarded) =====
    uy_DM_kij = _contract_single(
        ux_D_kij, M_op, BLOCK_SIZE_EXEC, N_PAD, N_PAD, N_PAD)                 # (BE,k,i',j')

    u_sum_kij = uy_MD_kij + uy_DM_kij                                         # (BE,k,i',j')

    # ===== PHASE 3 — contract over k. Permute (BE,k,i',j') → (BE,i',j',k). =====
    u_sum_ijk = tl.permute(u_sum_kij, (0, 2, 3, 1))
    uy_MM_ijk = tl.permute(uy_MM_kij, (0, 2, 3, 1))

    z1_M = _contract_single(
        u_sum_ijk, M_op, BLOCK_SIZE_EXEC, N_PAD, N_PAD, N_PAD)                # (BE,i',j',k')
    z2_D = _contract_single(
        uy_MM_ijk, D_op, BLOCK_SIZE_EXEC, N_PAD, N_PAD, N_PAD)

    out_eijk = z1_M + z2_D                                                    # (BE,i',j',k')
    out_ekji = tl.permute(out_eijk, (0, 3, 2, 1))                             # (BE,k,j,i)
    out_kjie = tl.permute(out_ekji, (1, 2, 3, 0))                             # (k,j,i,BE)
    tl.store(out_ptr + offs_4d, out_kjie, mask=full_mask)


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------

def _laplace_apply(u, stiffness_1d, mass_1d, block_size_data, block_size_exec):
    n_blocks = u.shape[0]
    num_elements = n_blocks * block_size_data
    grid = num_elements // block_size_exec

    O_MD = _stacked_MD(mass_1d, stiffness_1d)
    M_op = _pad_op(mass_1d).contiguous()
    D_op = _pad_op(stiffness_1d).contiguous()
    out = torch.empty_like(u)

    _laplace_kernel_p4_cuda[(grid,)](
        u, O_MD, M_op, D_op, out,
        num_elements,
        N=_N,
        N_PAD=_N_PAD,
        BLOCK_SIZE_DATA=block_size_data,
        BLOCK_SIZE_EXEC=block_size_exec,
    )
    return out


def gpu_timeit(fn, *, warmup=10, iters=100, pipeline_warmup=50):
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        fn(); torch.cuda.synchronize(stream)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--log2-elements", type=int, default=19)
    parser.add_argument("--data-block", type=int, default=32)
    parser.add_argument("--exec-block", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters",  type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    N = _N
    BD = args.data_block
    BE = args.exec_block
    num_elements = 2 ** args.log2_elements
    if num_elements % BD != 0:
        raise ValueError(f"num_elements={num_elements} must be divisible by data-block={BD}")
    if BD % BE != 0:
        raise ValueError(f"data-block={BD} must be divisible by exec-block={BE}")
    n_blocks = num_elements // BD

    device = torch.cuda.current_device()
    print(f"Device:      {torch.cuda.get_device_name(device)}")
    print(f"Layout:      BXYZE (n_blocks, 5, 5, 5, block_size_data)")
    print(f"P=4  N=5  E=2^{args.log2_elements}={num_elements}")
    print(f"data_block={BD}  exec_block={BE}  n_blocks={n_blocks}  kernel=p4_cuda_core")
    print()

    mfl = math_flops(num_elements)
    tb  = theory_bytes(num_elements)
    print(f"Math   FLOPs (18*N**4 + 2*N**3)*E:  {mfl:.3e}")
    print(f"Theory DRAM  (2*N**3*E*8):          {tb:.3e}  ({tb/1e9:.3f} GB)")
    print(f"Theory AI (math FLOPs / DRAM):      {mfl/tb:.2f} FLOPs/byte")
    print()

    u = torch.randn(n_blocks, N, N, N, BD, dtype=torch.float64, device="cuda")
    stiffness = torch.randn(N, N, dtype=torch.float64, device="cuda")
    mass = torch.randn(N, N, dtype=torch.float64, device="cuda")

    t0 = time.perf_counter()
    _ = _laplace_apply(u, stiffness, mass, BD, BE)
    torch.cuda.synchronize()
    print(f"Compilation time: {(time.perf_counter()-t0)*1000:.1f} ms")

    if args.verify:
        print("Verifying against einsum reference...")
        out_t = _laplace_apply(u, stiffness, mass, BD, BE)
        u_ref   = u.permute(0, 4, 1, 2, 3).reshape(num_elements, N, N, N).contiguous()
        out_ref = out_t.permute(0, 4, 1, 2, 3).reshape(num_elements, N, N, N).contiguous()
        expected = (
            torch.einsum("kK,jJ,iI,eKJI->ekji", stiffness, mass, mass, u_ref)
            + torch.einsum("kK,jJ,iI,eKJI->ekji", mass, stiffness, mass, u_ref)
            + torch.einsum("kK,jJ,iI,eKJI->ekji", mass, mass, stiffness, u_ref)
        )
        max_err = float(torch.max(torch.abs(out_ref - expected)))
        rel = max_err / float(torch.max(torch.abs(expected)))
        status = "PASS" if rel < 1e-10 else "FAIL"
        print(f"  max_abs_err={max_err:.3e}  rel_err={rel:.3e}  [{status}]")
        if status == "FAIL":
            raise SystemExit(1)
        print()

    ms = gpu_timeit(lambda: _laplace_apply(u, stiffness, mass, BD, BE),
                    warmup=args.warmup, iters=args.iters)
    tflops = mfl / (ms * 1e-3) / 1e12
    bw     = tb  / (ms * 1e-3) / 1e9
    dofs   = N**3 * num_elements
    gdofs  = dofs / (ms * 1e-3) / 1e9
    print(f"Timing ({args.iters} iters):  avg = {ms:.3f} ms")
    print(f"  TFLOPS (math FLOPs / time):      {tflops:.3f}")
    print(f"  Bandwidth (theory bytes / time): {bw:.1f} GB/s")
    print(f"  Throughput:                      {gdofs:.3f} GDoF/s")


if __name__ == "__main__":
    main()
