"""FP64 GEMM showcase benchmark: what Triton can *really* do vs cuBLAS on H100.

Methodology (designed around two measured pitfalls on this box):
  1. Clock drift: neighboring GPUs swing SM clocks +-10% within minutes, so
     cross-run or even cross-minute numbers are not comparable. All comparisons
     here are PAIRED: cuBLAS and Triton are measured back-to-back inside each
     round, and the reported gap is the median of per-round gaps.
  2. Autotuner boost-noise: a single short autotune pass under boost clocks
     mispicks between near-tied configs. Instead we steady-state screen a
     curated config set per size and let the best one represent Triton.

Two kernels compete:
  - pow2:  the classic single-dot kernel (tile sides limited to powers of 2 by
    tl.arange), so e.g. N=1536/2304/3072 hit wave-quantization tails.
  - split: BM = BM0 + BM1 (e.g. 96 = 64 + 32) via two accumulators/dots that
    share one B tile per K-step — no masking, no padding, and non-power-of-2
    M-tiles become available to fix the wave fit (e.g. 3072: 96x64 -> 1536
    blocks -> 3.88 waves at 3 CTAs/SM = 97% tail efficiency vs 87% for 64x64).

Usage:
  CUDA_VISIBLE_DEVICES=1 python bench_fp64_showcase.py [size ...]
  (default sizes cover the wave-quantization landscape: 1024..4096)
"""
import sys

import torch
import triton
import triton.language as tl
import triton.testing
from time import sleep

DEVICE = "cuda"
ROUNDS = 5          # paired head-to-head rounds per size
FINALISTS = 2       # configs per kernel family that advance to head-to-head

# Curated from the 2026-06 H100 sweeps: small-size winners (64x64 nw2),
# wave-fit large tiles (64x128 / 128x64 nw4), and narrow tiles for tiny sizes.
POW2_CONFIGS = [
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

SPLIT_CONFIGS = [
    # (BM0, BM1, BN, BK, nw, ns) -> BM = BM0 + BM1
    (64, 32, 64, 16, 2, 3),    # 96x64
    (64, 32, 64, 16, 4, 3),
    (64, 32, 64, 16, 4, 4),
    (64, 32, 128, 16, 4, 3),   # 96x128
    (64, 32, 128, 16, 4, 4),
    (128, 32, 64, 16, 4, 3),   # 160x64
    (64, 16, 64, 16, 2, 3),    # 80x64
    (64, 16, 64, 16, 2, 4),
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


@triton.jit
def matmul_kernel_split(a_ptr, b_ptr, c_ptr, M, N, K,
                        stride_am, stride_ak, stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BM0: tl.constexpr, BM1: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, GM: tl.constexpr):
    """Non-power-of-2 M-tile (BM = BM0 + BM1) without masking the math: the
    tile is two stacked sub-tiles with separate accumulators sharing one B
    load per K-step. Loads wrap with % M; only the store is masked (so sizes
    BM doesn't divide stay correct at epilogue-only cost)."""
    BM: tl.constexpr = BM0 + BM1
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    group_size_m = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am0 = (pid_m * BM + tl.arange(0, BM0)) % M
    offs_am1 = (pid_m * BM + BM0 + tl.arange(0, BM1)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a0_ptrs = a_ptr + offs_am0[:, None] * stride_am + offs_k[None, :] * stride_ak
    a1_ptrs = a_ptr + offs_am1[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc0 = tl.zeros((BM0, BN), dtype=tl.float64)
    acc1 = tl.zeros((BM1, BN), dtype=tl.float64)
    for _ in range(0, tl.cdiv(K, BK)):
        b = tl.load(b_ptrs)
        acc0 = tl.dot(tl.load(a0_ptrs), b, acc0, out_dtype=tl.float64)
        acc1 = tl.dot(tl.load(a1_ptrs), b, acc1, out_dtype=tl.float64)
        a0_ptrs += BK * stride_ak
        a1_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    offs_cn = pid_n * BN + tl.arange(0, BN)
    offs_cm0 = pid_m * BM + tl.arange(0, BM0)
    offs_cm1 = pid_m * BM + BM0 + tl.arange(0, BM1)
    c0_ptrs = c_ptr + stride_cm * offs_cm0[:, None] + stride_cn * offs_cn[None, :]
    c1_ptrs = c_ptr + stride_cm * offs_cm1[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c0_ptrs, acc0, mask=(offs_cm0[:, None] < M) & (offs_cn[None, :] < N))
    tl.store(c1_ptrs, acc1, mask=(offs_cm1[:, None] < M) & (offs_cn[None, :] < N))


def bench_size(size):
    M = N = K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float64)
    ref = torch.matmul(a, b)
    tf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    def make_pow2(BM, BN, BK, nw, ns):
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def launch():
            matmul_kernel[grid](a, b, c, M, N, K,
                                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1),
                                BM=BM, BN=BN, BK=BK, GM=8,
                                num_warps=nw, num_stages=ns)
        return launch, f"BM{BM} BN{BN} nw{nw} ns{ns}"

    def make_split(BM0, BM1, BN, BK, nw, ns):
        BM = BM0 + BM1
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def launch():
            matmul_kernel_split[grid](a, b, c, M, N, K,
                                      a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                      c.stride(0), c.stride(1),
                                      BM0=BM0, BM1=BM1, BN=BN, BK=BK, GM=8,
                                      num_warps=nw, num_stages=ns)
        return launch, f"BM{BM}({BM0}+{BM1}) BN{BN} nw{nw} ns{ns}"

    def screen(entries):
        scored = []
        for launch, label in entries:
            try:
                c.zero_()
                launch()
                torch.cuda.synchronize()
                if not torch.allclose(c, ref, atol=1e-8, rtol=1e-5):
                    continue
                ms = triton.testing.do_bench(launch, quantiles=[0.5])
                scored.append((tf(ms), launch, label))
            except Exception:
                continue
        scored.sort(key=lambda t: -t[0])
        return scored[:FINALISTS]

    pow2_finalists = screen(make_pow2(*cfg) for cfg in POW2_CONFIGS)
    split_finalists = screen(make_split(*cfg) for cfg in SPLIT_CONFIGS)

    # Paired head-to-head: cuBLAS + finalists of both families, back-to-back
    # per round so clock drift hits everyone equally. The first measurement of
    # a round sits in the most favorable clock/thermal slot, so alternate
    # whether cuBLAS goes first or last to average out the ordering bias.
    cublas = lambda: torch.matmul(a, b)
    gaps, cu_best = [], 0.0
    best = {"pow2": (0.0, "-"), "split": (0.0, "-")}

    def bench_triton_finalists():
        round_best = 0.0
        for family, finalists in (("pow2", pow2_finalists), ("split", split_finalists)):
            for _, launch, label in finalists:
                t = tf(triton.testing.do_bench(launch, quantiles=[0.5]))
                round_best = max(round_best, t)
                if t > best[family][0]:
                    best[family] = (t, label)
        return round_best

    for rnd in range(ROUNDS):
        if rnd % 2 == 0:
            cu = tf(triton.testing.do_bench(cublas, quantiles=[0.5]))
            sleep(0.5)  # let the boost noise settle before the next measurement
            round_best = bench_triton_finalists()
        else:
            sleep(0.5)  # let the boost noise settle before the next measurement
            round_best = bench_triton_finalists()
            cu = tf(triton.testing.do_bench(cublas, quantiles=[0.5]))
        cu_best = max(cu_best, cu)
        gaps.append(100 * (cu - round_best) / cu)
    gaps.sort()
    med_gap = gaps[len(gaps) // 2]
    return cu_best, best["pow2"], best["split"], med_gap


def main():
    sizes = [int(s) for s in sys.argv[1:]] or \
        [1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096]
    print(f"{'size':>6} {'cuBLAS':>8} {'pow2':>8} {'split':>8} {'gap(med)':>9}  best config")
    print("-" * 78)
    for size in sizes:
        cu, (p2, p2cfg), (sp, spcfg), gap = bench_size(size)
        winner = spcfg if sp > p2 else p2cfg
        print(f"{size:>6} {cu:>8.2f} {p2:>8.2f} {sp:>8.2f} {gap:>+8.1f}%  {winner}",
              flush=True)


if __name__ == "__main__":
    main()


"""
RESULTS with swizzle fix + prefetch-on-sm90 + pressure gate (2026-06-11 night):
  size   cuBLAS     pow2    split  gap(med)  best config
------------------------------------------------------------------------------
  1024    47.76    49.24    34.15     -2.9%  BM64 BN32 nw2 ns4     <- Triton wins
  1280    43.39    42.02    33.14     +3.3%  BM64 BN128 nw4 ns3
  1536    50.65    46.63    47.31     +6.4%  BM96(64+32) BN64 nw4 ns4
  1792    56.43    51.21    41.56     +9.5%  BM64 BN64 nw2 ns4
  2048    56.88    55.13    46.37     +1.2%  BM64 BN128 nw4 ns4
  2304    55.76    51.01    52.01     +5.9%  BM80(64+16) BN64 nw2 ns3
  2560    52.14    49.42    50.90     +0.4%  BM80(64+16) BN64 nw2 ns3
  3072    54.84    50.39    48.81     +8.2%  BM64 BN128 nw4 ns3
  3584    55.21    54.29    49.44     +2.0%  BM64 BN128 nw4 ns3
  4096    54.33    53.71    49.45     -0.1%  BM64 BN128 nw4 ns3    <- tie
Median +2.7%; Triton wins outright at 1024 (prefetch fires on the small
low-pressure tile) and ties 4096. Residual >5%: 1536/1792/2304/3072.

RESULTS two-kernel run (2026-06-11 H100, paired rounds, median gap):
  size   cuBLAS     pow2    split  gap(med)  best config
------------------------------------------------------------------------------
  1024    47.97    47.29    34.20     +1.4%  BM64 BN32 nw2 ns4
  1280    43.55    42.05    33.20     +3.5%  BM64 BN128 nw4 ns3
  1536    50.55    44.09    47.40     +6.7%  BM96(64+32) BN64 nw4 ns4  <- split
  1792    56.09    51.15    41.50     +8.7%  BM64 BN64 nw2 ns4
  2048    55.93    55.30    46.16     +0.9%  BM128 BN64 nw4 ns3
  2304    55.63    51.26    51.93     +6.8%  BM80(64+16) BN64 nw2 ns3  <- split
  2560    51.98    50.56    50.93     +1.1%  BM80(64+16) BN64 nw2 ns3  <- split
  3072    55.23    50.26    49.71     +8.8%  BM64 BN128 nw4 ns3
  3584    55.57    54.10    49.46     +1.4%  BM64 BN128 nw4 ns3
  4096    54.85    53.48    49.37     +2.0%  BM128 BN64 nw4 ns4
Median gap 2.8%, worst 8.8%. The split kernel fixes the 1536 cliff
(+12.8% -> +6.7%) and wins 2304/2560 via the 80(64+16) tile; its ~5% per-SM
overhead (two dots) keeps it from winning at 3072 despite the 3.88-wave fit.
warpsPerCTA note: the hasChainedDot fallback in warpsPerTileV2 gives both
stacked dots [1,4]; patching [2,2]/[4,1] into the TTGIR measured SLOWER
(52.4/42.5 vs 53.5T) - [1,4] is genuinely right for this shape.

RESULTS pow2-only run (2026-06-11 H100):
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
