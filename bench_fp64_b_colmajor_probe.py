"""B column-major probe.

Question: if the user passes B with K-contiguous global strides
(b.T.contiguous().T pattern — physical N×K row-major, logical K×N),
does the compiler propagate order=[0,1] into B's shared encoding and
emit ld.shared.v2.b64 at kWidth>=2, closing the 30% BLOCK_K>=32 gap?

If yes → user-space workaround is real; fp64_swizzle.md Layer 3 work
can potentially be replaced by documentation. If no → compiler-side
fix is unavoidable.
"""
import torch
import triton

with open("fp64mma_test.py") as f:
    src = f.read()
src_top = src.split("# ── Unit test ──")[0]
ns: dict = {}
exec(compile(src_top, "fp64mma_test.py", "exec"), ns)

matmul_fp64 = ns["matmul_fp64"]
matmul_kernel_fp64 = ns["matmul_kernel_fp64"]
all_configs = ns["get_fp64_autotune_config"]()
small_bk = [c for c in all_configs if c.kwargs["BLOCK_SIZE_K"] <= 16]
large_bk = [c for c in all_configs if c.kwargs["BLOCK_SIZE_K"] >= 32]


def make_b(K, N, kind):
    if kind == "row":
        # Standard: K×N row-major. stride_bk = N, stride_bn = 1. order=[1,0].
        return torch.randn((K, N), device="cuda", dtype=torch.float64)
    elif kind == "col":
        # K-contiguous: physical N×K row-major, viewed as K×N.
        # stride_bk = 1, stride_bn = K. order=[0,1] in Triton terms.
        b_phys = torch.randn((N, K), device="cuda", dtype=torch.float64)
        return b_phys.t()  # logical K×N, stride (1, K)
    else:
        raise ValueError(kind)


def bench(size, configs, b_kind, label):
    autotuner = matmul_kernel_fp64
    autotuner.configs = configs
    autotuner.cache = {}
    M = N = K = size
    a = torch.randn((M, K), device="cuda", dtype=torch.float64)
    b = make_b(K, N, b_kind)
    # Numerical sanity vs torch
    c_tri = matmul_fp64(a, b)
    c_ref = torch.matmul(a, b)
    ok = torch.allclose(c_tri, c_ref, atol=1e-8, rtol=1e-5)
    torch.cuda.synchronize()
    best = autotuner.best_config
    cu, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=[0.5, 0.2, 0.8])
    tr, _, _ = triton.testing.do_bench(lambda: matmul_fp64(a, b), quantiles=[0.5, 0.2, 0.8])
    cu_t = 2*M*N*K*1e-12 / (cu * 1e-3)
    tr_t = 2*M*N*K*1e-12 / (tr * 1e-3)
    print(f"  [{label} | size={size} | b={b_kind} | b.strides={b.stride()}] correct={ok}")
    print(f"  [{label} | size={size}] best={best}")
    print(f"  [{label} | size={size}] cuBLAS={cu_t:.2f}  Triton={tr_t:.2f}  gap={100*(cu_t-tr_t)/cu_t:.1f}%")
    return tr_t


print()
for size in [2048]:
    print(f"=== size={size} ===")
    print("-- B row-major (baseline) --")
    t_row_small = bench(size, small_bk, "row", "BLOCK_K<=16")
    t_row_large = bench(size, large_bk, "row", "BLOCK_K>=32")
    print("-- B col-major (K-contig global) --")
    t_col_small = bench(size, small_bk, "col", "BLOCK_K<=16")
    t_col_large = bench(size, large_bk, "col", "BLOCK_K>=32")
    print(f"\n  Summary at size={size}:")
    print(f"    BLOCK_K<=16:  row={t_row_small:.2f}  col={t_col_small:.2f}  Δ={100*(t_col_small-t_row_small)/t_row_small:+.1f}%")
    print(f"    BLOCK_K>=32:  row={t_row_large:.2f}  col={t_col_large:.2f}  Δ={100*(t_col_large-t_row_large)/t_row_large:+.1f}%")
