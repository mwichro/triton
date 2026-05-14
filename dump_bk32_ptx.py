"""Step A — dump PTX for the BLOCK_K=32 (kWidth=2) winning config at size=2048.

Goal: characterize the B-side shared loads in the K-loop. We expect either:
  - Two scalar `ld.shared.b64` per thread per K-step (Layer 2 confirmed)
  - One `ld.shared.v2.b64` per thread per K-step (Layer 2 hypothesis wrong)
  - Something else (diagnostic of a different split)

We constrain the autotune list to BLOCK_K=32 configs so the winner is the
one that exhibits the kWidth=2 pathology.
"""
import sys
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
bk32_configs = [c for c in all_configs if c.kwargs["BLOCK_SIZE_K"] == 32]
print(f"BLOCK_K=32 configs: {len(bk32_configs)}")
for c in bk32_configs:
    print(f"  {c}")

matmul_kernel_fp64.configs = bk32_configs
matmul_kernel_fp64.cache = {}

M = N = K = 2048
a = torch.randn((M, K), device="cuda", dtype=torch.float64)
b = torch.randn((K, N), device="cuda", dtype=torch.float64)
matmul_fp64(a, b); torch.cuda.synchronize()

best = matmul_kernel_fp64.best_config
print(f"\nWinning config at size=2048: {best}\n")

# Find the compiled kernel and dump its PTX.
# The autotuner stores per-config compiled JITFunction in matmul_kernel_fp64.cache.
# After best_config is set, the next launch uses that config; we can grab it.
key = (M, N, K)
cached = matmul_kernel_fp64.cache.get(key)
print(f"Cache entry for key {key}: {type(cached).__name__ if cached else None}")

# Easiest path: re-invoke with explicit best config to get the compiled binary.
# Triton stores it under matmul_kernel_fp64.fn.cache (JITFunction cache).
jit = matmul_kernel_fp64.fn
print(f"JITFunction caches: {len(jit.device_caches)}")
for dev_id, dev_cache in jit.device_caches.items():
    # dev_cache is a tuple/dict of compiled kernels keyed by signature
    print(f"  device {dev_id}: {type(dev_cache).__name__}")
    if hasattr(dev_cache, "keys"):
        for k in list(dev_cache.keys())[:3]:
            print(f"    key sample: {k}")

# Dump PTX from one of the compiled kernels.
# JITFunction.device_caches is { dev: { key: CompiledKernel } } in modern triton.
import json
for dev_id, dev_cache in jit.device_caches.items():
    for k, compiled in dev_cache.items() if hasattr(dev_cache, "items") else []:
        # CompiledKernel has .asm dict {ptx, llir, ttgir, ttir, ...}
        if hasattr(compiled, "asm"):
            print(f"\n--- compiled kernel for key={k} ---")
            print(f"asm keys: {list(compiled.asm.keys())}")
            ptx = compiled.asm.get("ptx", "")
            if ptx:
                outpath = f"/tmp/fp64_bk32_dev{dev_id}.ptx"
                with open(outpath, "w") as f:
                    f.write(ptx)
                print(f"wrote {len(ptx)} bytes to {outpath}")
