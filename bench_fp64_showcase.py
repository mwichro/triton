"""FP64 GEMM showcase benchmark: what Triton can *really* do vs cuBLAS on H100.

Methodology (designed around two measured pitfalls on this box):
  1. Clock drift: neighboring GPUs swing SM clocks +-10% within minutes, so
     cross-run or even cross-minute numbers are not comparable. All comparisons
     here are PAIRED: cuBLAS and Triton are measured back-to-back inside each
     round, and the reported gap is the median of per-round gaps.
  2. Autotuner boost-noise: a single short autotune pass under boost clocks
     mispicks between near-tied configs. Instead we steady-state screen a
     curated config set per size and let the best one represent Triton.

Usage:
  CUDA_VISIBLE_DEVICES=1 python bench_fp64_showcase.py [size ...]
  (default sizes cover the wave-quantization landscape: 1024..4096)
"""
import sys

import torch
import triton
import triton.language as tl
import triton.testing

DEVICE = "cuda"
ROUNDS = 5          # paired head-to-head rounds per size
FINALISTS = 3       # configs that advance from screening to head-to-head

# Curated from the 2026-06 H100 sweeps: small-size winners (64x64 nw2),
# wave-fit large tiles (64x128 / 128x64 nw4), and narrow tiles for tiny sizes.
CONFIGS = [
    # (BM, BN, BK, nw, ns)
    (64, 64, 16, 2, 3),
    (64, 64, 16, 2, 4),
    (64, 128, 16, 4, 3),
    (64, 128, 16, 4, 4),
    (128, 64, 16, 4, 3),
    (128, 64, 16, 4, 4),
    (128, 128, 16, 8, 3),
    (32, 64, 16, 2, 4),
    (64, 32, 16, 2, 4),
]


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    group_size_m = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    for _ in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float64)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc)


def bench_size(size):
    M = N = K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float64)
    ref = torch.matmul(a, b)
    tf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    def make_launch(BM, BN, BK, nw, ns):
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def launch():
            matmul_kernel[grid](a, b, c, M, N, K,
                                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),
                                BM=BM, BN=BN, BK=BK, GM=8,
                                num_warps=nw, num_stages=ns)
        return launch

    # Screening: steady-state bench every config, keep the top FINALISTS.
    screened = []
    for cfg in CONFIGS:
        launch = make_launch(*cfg)
        try:
            c.zero_()
            launch()
            torch.cuda.synchronize()
            if not torch.allclose(c, ref, atol=1e-8, rtol=1e-5):
                continue
            ms = triton.testing.do_bench(launch, quantiles=[0.5])
            screened.append((tf(ms), cfg, launch))
        except Exception:
            continue
    screened.sort(key=lambda t: -t[0])
    finalists = screened[:FINALISTS]

    # Paired head-to-head: cuBLAS + finalists measured back-to-back per round.
    cublas = lambda: torch.matmul(a, b)
    gaps, tri_best, cu_best, win_cfg = [], 0.0, 0.0, None
    for _ in range(ROUNDS):
        cu = tf(triton.testing.do_bench(cublas, quantiles=[0.5]))
        best_round, best_cfg = 0.0, None
        for _, cfg, launch in finalists:
            t = tf(triton.testing.do_bench(launch, quantiles=[0.5]))
            if t > best_round:
                best_round, best_cfg = t, cfg
        gaps.append(100 * (cu - best_round) / cu)
        if best_round > tri_best:
            tri_best, win_cfg = best_round, best_cfg
        cu_best = max(cu_best, cu)
    gaps.sort()
    med_gap = gaps[len(gaps) // 2]
    return cu_best, tri_best, med_gap, win_cfg


def main():
    sizes = [int(s) for s in sys.argv[1:]] or \
        [1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096]
    print(f"{'size':>6} {'cuBLAS':>8} {'Triton':>8} {'gap(med)':>9}  best config")
    print("-" * 70)
    for size in sizes:
        cu, tri, gap, cfg = bench_size(size)
        BM, BN, BK, nw, ns = cfg
        print(f"{size:>6} {cu:>8.2f} {tri:>8.2f} {gap:>+8.1f}%  "
              f"BM{BM} BN{BN} BK{BK} nw{nw} ns{ns}", flush=True)


if __name__ == "__main__":
    main()


"""
RESULTS (2026-06-11 H100):
  size   cuBLAS   Triton  gap(med)  best config
----------------------------------------------------------------------
  1024    47.76    47.13     +1.3%  BM64 BN32 BK16 nw2 ns4
  1280    43.46    41.97     +3.4%  BM64 BN128 BK16 nw4 ns3
  1536    50.37    43.94    +12.8%  BM64 BN32 BK16 nw2 ns4
  1792    56.15    51.53     +8.6%  BM64 BN64 BK16 nw2 ns4
  2048    55.68    55.17     +0.5%  BM64 BN128 BK16 nw4 ns4
  2304    55.02    50.39     +9.0%  BM64 BN64 BK16 nw2 ns4
  2560    51.98    50.55     +2.7%  BM128 BN64 BK16 nw4 ns4
  3072    55.38    50.94     +7.6%  BM64 BN128 BK16 nw4 ns4
  3584    55.68    54.31     +2.7%  BM64 BN128 BK16 nw4 ns3
  4096    55.30    53.83     +1.8%  BM128 BN64 BK16 nw4 ns3

"""