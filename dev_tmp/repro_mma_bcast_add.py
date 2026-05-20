"""Minimal reproducer: MMA→Blocked layout-convert cost in `dot + bcast-mul-add → next dot`.

Two F64 kernels with the **identical** load/compute structure except for one line:

  A — does `acc = acc + x[:,None] * y[None,:]` after each dot
  B — same chain, no add

Everything else is matched: same per-iter inputs, same reshape/permute/split scaffold,
same final dot. Both K=4, N=8 → F64 MMA 8×8×4 tile hit exactly (no padding waste).

What the A-vs-B gap measures: the **incremental** cost of the MMA→Blocked round-trip
the compiler is forced to insert because the bcast-mul-add operand is in BlockedEncoding
while the dot result is in MmaEncoding. The reshape/permute/split scaffold below the
add lives in BlockedEncoding in both kernels, so its convert cost cancels in the A-B diff.

HBM is hoisted out of the chain loop in both kernels so per-iter time reflects compute +
register-resident layout converts only — not memory latency.

Run:
  /scratch/mwichro/triton/.venv/bin/python dev_tmp/repro_mma_bcast_add.py
  /scratch/mwichro/triton/.venv/bin/python dev_tmp/repro_mma_bcast_add.py --chain 32 -M 128
  DUMP_IR=1 ... --chain 4   # dump TTGIR for inspection
"""
from __future__ import annotations

import os
import re
import time

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels — identical except for the line marked "PATTERN"
# ---------------------------------------------------------------------------

@triton.jit
def kernel_A(
    A_ptr, B_ptr, C_ptr, X_ptr, Y_ptr, OUT_ptr,
    NUM_CTAS,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, N2: tl.constexpr,
    CHAIN: tl.constexpr,
):
    """dot → bcast-mul-add → reshape/split → dot, chained CHAIN times.

    All HBM loads happen ONCE outside the chain loop. The per-iter "+ x*y"
    uses register-resident slices of the preloaded X/Y tensors.
    """
    pid = tl.program_id(0)
    base = pid * M * K
    base_out = pid * M * N2

    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    rn2 = tl.arange(0, 2 * N)
    rN2 = tl.arange(0, N2)
    # ---- HBM loads (all hoisted; per-iter loads inside loop go through L1) ----
    b = tl.load(B_ptr + rk[:, None] * (2 * N) + rn2[None, :])         # (K, 2N)
    c = tl.load(C_ptr + rk[:, None] * N2 + rN2[None, :])              # (K, N2)
    a = tl.load(A_ptr + base + rm[:, None] * K + rk[None, :])         # (M, K)

    # ---- Chain ----
    for step in tl.static_range(CHAIN):
        x = tl.load(X_ptr + step * M + rm)                            # (M,)
        y = tl.load(Y_ptr + step * (2 * N) + rn2)                     # (2N,)

        acc = tl.dot(a, b)                                            # (M, 2N) MmaEncoding
        acc = acc + x[:, None] * y[None, :]                           # PATTERN: forces MMA→Blocked convert

        acc2 = tl.reshape(acc, (M, 2, N))
        acc_p = tl.permute(acc2, (0, 2, 1))                           # (M, N, 2)
        top, _bot = tl.split(acc_p)                                   # (M, N)
        top_2 = tl.reshape(top, (M, 2, K))
        top_p = tl.permute(top_2, (0, 2, 1))                          # (M, K, 2)
        a, _ = tl.split(top_p)                                        # (M, K)

    out = tl.dot(a, c)                                                # (M, N2)
    tl.store(OUT_ptr + base_out + rm[:, None] * N2 + rN2[None, :], out)


@triton.jit
def kernel_B(
    A_ptr, B_ptr, C_ptr, X_ptr, Y_ptr, OUT_ptr,
    NUM_CTAS,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, N2: tl.constexpr,
    CHAIN: tl.constexpr,
):
    """Baseline: identical structure to A but the PATTERN line is removed.

    Crucially, this kernel performs the **same** HBM loads as A so the A−B
    delta reflects only the per-iter MMA→Blocked layout-convert cost, not
    differential memory traffic. The loaded X/Y are sunk via a single
    scalar FMA into the output to prevent dead-code elimination.
    """
    pid = tl.program_id(0)
    base = pid * M * K
    base_out = pid * M * N2

    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    rn2 = tl.arange(0, 2 * N)
    rN2 = tl.arange(0, N2)
    b = tl.load(B_ptr + rk[:, None] * (2 * N) + rn2[None, :])
    c = tl.load(C_ptr + rk[:, None] * N2 + rN2[None, :])
    a = tl.load(A_ptr + base + rm[:, None] * K + rk[None, :])

    # Tensor sinks — parallel adds across M / 2N (no serial scalar dependency
    # like dummy_sink+=tl.sum(...) would create).
    x_sink = tl.zeros((M,), dtype=tl.float64)
    y_sink = tl.zeros((2 * N,), dtype=tl.float64)
    for step in tl.static_range(CHAIN):
        # Identical per-iter HBM loads as kernel A.
        x = tl.load(X_ptr + step * M + rm)                            # (M,)
        y = tl.load(Y_ptr + step * (2 * N) + rn2)                     # (2N,)

        acc = tl.dot(a, b)
        # NO PATTERN HERE — this is the only difference vs kernel A.
        x_sink = x_sink + x                                           # keep x live (parallel)
        y_sink = y_sink + y                                           # keep y live (parallel)

        acc2 = tl.reshape(acc, (M, 2, N))
        acc_p = tl.permute(acc2, (0, 2, 1))
        top, _bot = tl.split(acc_p)
        top_2 = tl.reshape(top, (M, 2, K))
        top_p = tl.permute(top_2, (0, 2, 1))
        a, _ = tl.split(top_p)

    out = tl.dot(a, c)
    # Sink x_sink and y_sink into out × 0 — one final reduction (not per-iter,
    # so no serialization). Math unchanged but DCE cannot remove the loop loads.
    out = out + (tl.sum(x_sink) * tl.sum(y_sink)) * 0.0
    tl.store(OUT_ptr + base_out + rm[:, None] * N2 + rN2[None, :], out)


# ---------------------------------------------------------------------------
# Driver / metrics
# ---------------------------------------------------------------------------

def _count_convert_layout(compiled) -> int:
    try:
        ttgir = compiled.asm.get("ttgir", "")
    except Exception:
        return -1
    if not ttgir:
        return -1
    return len(re.findall(r"\bconvert_layout\b", ttgir))


def _gpu_time(fn, warmup=20, iters=200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _ref(A_all, B, C, X, Y, N, M, CHAIN, with_add: bool):
    """PyTorch reference. X is (CHAIN, M), Y is (CHAIN, 2N)."""
    out = torch.empty(A_all.shape[0], C.shape[1], dtype=A_all.dtype, device=A_all.device)
    for cta_start in range(0, A_all.shape[0], M):
        a = A_all[cta_start:cta_start + M].clone()
        for step in range(CHAIN):
            acc = a @ B
            if with_add:
                acc = acc + X[step][:, None] * Y[step][None, :]
            top = acc.reshape(M, 2, N).permute(0, 2, 1)[..., 0]
            a = top.reshape(M, 2, -1).permute(0, 2, 1)[..., 0]
        out[cta_start:cta_start + M] = a @ C
    return out


def main(M=128, K=4, N=8, N2=4, CHAIN=8, NUM_CTAS=2048, dtype=torch.float64):
    """M is PER-CTA. A100 has 108 SMs; NUM_CTAS=2048 = ~19 CTAs/SM, saturated.

    Per-CTA M=128 mirrors the FE p=4 kernel's (cc) slab size (BE * N_MMA**2 = 8*16).
    """
    torch.manual_seed(0)
    dev = "cuda"

    total_M = NUM_CTAS * M
    A = torch.randn(total_M, K, dtype=dtype, device=dev)
    B = torch.randn(K, 2 * N, dtype=dtype, device=dev)
    C = torch.randn(K, N2, dtype=dtype, device=dev)
    X = torch.randn(CHAIN, M, dtype=dtype, device=dev)
    Y = torch.randn(CHAIN, 2 * N, dtype=dtype, device=dev)

    out_A = torch.empty(total_M, N2, dtype=dtype, device=dev)
    out_B = torch.empty(total_M, N2, dtype=dtype, device=dev)

    grid = (NUM_CTAS,)

    print(f"Compiling A (CHAIN={CHAIN}, M={M}, grid={NUM_CTAS})...", flush=True)
    t0 = time.perf_counter()
    compiled_A = kernel_A[grid](A, B, C, X, Y, out_A, NUM_CTAS, M=M, K=K, N=N, N2=N2, CHAIN=CHAIN)
    print(f"  A: {time.perf_counter()-t0:.1f}s", flush=True)

    print("Compiling B...", flush=True)
    t0 = time.perf_counter()
    compiled_B = kernel_B[grid](A, B, C, X, Y, out_B, NUM_CTAS, M=M, K=K, N=N, N2=N2, CHAIN=CHAIN)
    print(f"  B: {time.perf_counter()-t0:.1f}s", flush=True)

    # ----- correctness -----
    ref_A = _ref(A[:M], B, C, X, Y, N, M, CHAIN, with_add=True)
    ref_B = _ref(A[:M], B, C, X, Y, N, M, CHAIN, with_add=False)
    scale_A = ref_A.abs().max().item()
    scale_B = ref_B.abs().max().item()
    err_A = (out_A[:M] - ref_A).abs().max().item() / max(scale_A, 1e-30)
    err_B = (out_B[:M] - ref_B).abs().max().item() / max(scale_B, 1e-30)

    # ----- flops (total across all CTAs) -----
    per_step_A = 2 * M * K * (2 * N) + M * (2 * N)
    per_step_B = 2 * M * K * (2 * N)
    flops_A = NUM_CTAS * (CHAIN * per_step_A + 2 * M * K * N2)
    flops_B = NUM_CTAS * (CHAIN * per_step_B + 2 * M * K * N2)

    ms_A = _gpu_time(lambda: kernel_A[grid](A, B, C, X, Y, out_A, NUM_CTAS, M=M, K=K, N=N, N2=N2, CHAIN=CHAIN))
    ms_B = _gpu_time(lambda: kernel_B[grid](A, B, C, X, Y, out_B, NUM_CTAS, M=M, K=K, N=N, N2=N2, CHAIN=CHAIN))

    cl_A = _count_convert_layout(compiled_A)
    cl_B = _count_convert_layout(compiled_B)

    print()
    print(f"Per-CTA M={M}  K={K}  N={N}  N2={N2}  CHAIN={CHAIN}  NUM_CTAS={NUM_CTAS}  dtype={dtype}")
    print()
    print(f"{'Kernel':<28} {'us':>8}  {'#convert_layout':>16}  {'rel_err':>10}")
    print("-" * 80)
    for name, ms, cl, err in [
        ("A: chain w/ bcast-mul-add  ", ms_A, cl_A, err_A),
        ("B: chain (no add) baseline ", ms_B, cl_B, err_B),
    ]:
        print(f"{name:<28} {ms*1000:8.1f}  {cl:>16d}  {err:10.2e}")
    print()
    print("A − B  = per-iter MMA→Blocked convert cost × CHAIN (HBM matched, scaffold matched).")
    print(f"  Δus           = {(ms_A - ms_B)*1000:.2f}")
    print(f"  Δus / CHAIN   = {(ms_A - ms_B)*1000/CHAIN:.3f}    (per-iter convert cost)")
    print(f"  A slowdown    = {ms_A / ms_B:.2f}x")
    print(f"  extra converts = {cl_A - cl_B}   (expect ~CHAIN={CHAIN} if compiler doesn't CSE)")

    if os.environ.get("DUMP_IR"):
        print("=" * 80)
        print("TTGIR for kernel A:")
        print("=" * 80)
        print(compiled_A.asm.get("ttgir", "<missing>"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=4)
    p.add_argument("-N", type=int, default=8)
    p.add_argument("--N2", type=int, default=4)
    p.add_argument("--chain", type=int, default=32,
                   help="Default 32 so total runtime stays >100us — past launch-overhead noise.")
    p.add_argument("--num-ctas", type=int, default=2048)
    args = p.parse_args()
    main(M=args.M, K=args.K, N=args.N, N2=args.N2, CHAIN=args.chain, NUM_CTAS=args.num_ctas)
