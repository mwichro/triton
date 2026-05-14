"""Layer 1 measurement probe.

Question: are BLOCK_K >= 32 configs (kWidth >= 2, where the swizzle vec cap
bites) within striking distance of the BLOCK_K=16 winner? If yes, Layer 1
has headroom. If no, skip to Layer 2.

Method: import the kernel module without running its benchmark, override
the autotune config list with two restricted sets (BLOCK_K=16-only and
BLOCK_K>=32-only), benchmark each at three representative sizes, compare.

Run under: TRITON_FP64_MMA_K is honored — leave unset for autoselect.
"""
import importlib.util
import sys
import torch
import triton

# Load the kernel module without executing its top-level benchmark loop.
with open("fp64mma_test.py") as f:
    src = f.read()
src_top = src.split("# ── Unit test ──")[0]
ns: dict = {}
exec(compile(src_top, "fp64mma_test.py", "exec"), ns)

matmul_fp64 = ns["matmul_fp64"]
matmul_kernel_fp64 = ns["matmul_kernel_fp64"]

# Pull out the original config list, then build two subsets.
all_configs = ns["get_fp64_autotune_config"]()
small_bk = [c for c in all_configs if c.kwargs["BLOCK_SIZE_K"] <= 16]
large_bk = [c for c in all_configs if c.kwargs["BLOCK_SIZE_K"] >= 32]
print(f"small_bk configs: {len(small_bk)}, large_bk configs: {len(large_bk)}")


def bench(size, configs, label):
    # Reach into the autotuner and override its config list, then clear its cache.
    autotuner = matmul_kernel_fp64
    autotuner.configs = configs
    autotuner.cache = {}
    autotuner.pre_hook = autotuner.pre_hook  # no-op, but keeps attr alive
    M = N = K = size
    a = torch.randn((M, K), device="cuda", dtype=torch.float64)
    b = torch.randn((K, N), device="cuda", dtype=torch.float64)
    matmul_fp64(a, b); torch.cuda.synchronize()
    best = autotuner.best_config
    cu, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=[0.5, 0.2, 0.8])
    tr, _, _ = triton.testing.do_bench(lambda: matmul_fp64(a, b), quantiles=[0.5, 0.2, 0.8])
    cu_t = 2*M*N*K*1e-12 / (cu * 1e-3)
    tr_t = 2*M*N*K*1e-12 / (tr * 1e-3)
    print(f"  [{label} | size={size}] best={best}")
    print(f"  [{label} | size={size}] cuBLAS={cu_t:.2f}  Triton={tr_t:.2f}  gap={100*(cu_t-tr_t)/cu_t:.1f}%")
    return tr_t


print()
for size in [1280, 2048, 4096]:
    print(f"=== size={size} ===")
    t_small = bench(size, small_bk, "BLOCK_K<=16")
    t_large = bench(size, large_bk, "BLOCK_K>=32")
    delta = 100 * (t_large - t_small) / t_small
    print(f"  → BLOCK_K>=32 is {delta:+.1f}% vs BLOCK_K<=16\n")
