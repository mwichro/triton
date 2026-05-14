"""
Step 1 of investigation plan: rerun fp64 Triton matmul on a "good" size (2048)
and a "bad" size (1280), and report which autotune config wins for each.

Reuses the kernel + autotune list from fp64mma_test.py.
"""
import importlib.util
import sys
import os

import torch
import triton

# Load the kernel module without running its top-level benchmark loop.
# Easiest: just import its function objects by execing in a controlled ns.
spec = importlib.util.spec_from_file_location("fp64mma_test", "fp64mma_test.py")
# We can't simply import-execute because the file runs a full benchmark at import.
# So read it and exec only the definitions we need.
with open("fp64mma_test.py") as f:
    src = f.read()

# Cut at the unit-test marker (everything below that is the long benchmark loop).
marker = "# ── Unit test ──"
src_top = src.split(marker)[0]
ns: dict = {}
exec(compile(src_top, "fp64mma_test.py", "exec"), ns)

matmul_fp64 = ns["matmul_fp64"]
matmul_kernel_fp64 = ns["matmul_kernel_fp64"]
DEVICE = ns["DEVICE"]


def run_and_report(size: int):
    M = N = K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)

    # Warmup + autotune lock-in.
    out = matmul_fp64(a, b)
    torch.cuda.synchronize()

    # Recover the winning config from the autotuner.
    best = matmul_kernel_fp64.best_config
    print(f"[size={size}] winning config: {best}")

    # cuBLAS timing
    cublas_ms, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=[0.5, 0.2, 0.8])
    triton_ms, _, _ = triton.testing.do_bench(lambda: matmul_fp64(a, b), quantiles=[0.5, 0.2, 0.8])
    cublas_tflops = 2 * M * N * K * 1e-12 / (cublas_ms * 1e-3)
    triton_tflops = 2 * M * N * K * 1e-12 / (triton_ms * 1e-3)
    gap = 100 * (cublas_tflops - triton_tflops) / cublas_tflops
    print(f"[size={size}] cuBLAS={cublas_tflops:.2f} TFLOPS  Triton={triton_tflops:.2f} TFLOPS  gap={gap:.1f}%")
    return best, triton_tflops, cublas_tflops


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [2048, 1280]
    for s in sizes:
        # Reset autotuner cache between sizes? The cache key is (M,N,K), so
        # different sizes get re-autotuned automatically.
        run_and_report(s)
