"""Driver for the codegen'd mixed MMA + CUDA-core Laplacian kernel (p=4).

Calls the generated kernel from _generated_mixed_p4_kernel.py. Same CLI as
triton_BXYZE_mixed_p4.py.

Usage
-----
    /export/home/mwichro/miniconda3/envs/jax09-bisect/bin/python \
        bench/volume/triton_BXYZE_mixed_p4_cg.py -e 20 --data-block 8 --exec-block 8
    /export/home/mwichro/miniconda3/envs/jax09-bisect/bin/python \
        bench/volume/triton_BXYZE_mixed_p4_cg.py -e 20 --verify
    /export/home/mwichro/miniconda3/envs/jax09-bisect/bin/python \
        bench/volume/triton_BXYZE_mixed_p4_cg.py -e 20 --profile     # ncu roofline
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import triton

_triton_cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".triton_cache_vol"))
os.environ["TRITON_CACHE_DIR"] = _triton_cache_dir

# Make `tensorfem` importable from the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Make sure the codegen runs and the kernel module is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Regenerate every import to avoid stale code while iterating. Skip the regen
# inside subprocesses (ncu kernel-replay re-imports this module repeatedly).
import codegen_mixed_p4 as _cg
_KERNEL_PATH = os.path.join(_HERE, "_generated_mixed_p4_kernel.py")
if not os.environ.get("TRITON_BXYZE_MIXED_P4_NO_REGEN"):
    with open(_KERNEL_PATH, "w") as _f:
        _f.write(_cg.generate_kernel_source())

import importlib
import _generated_mixed_p4_kernel as _gen
_gen = importlib.reload(_gen)
_laplace_kernel_mixed_p4_codegen = _gen._laplace_kernel_mixed_p4_codegen


_N = 5
_N_MMA = 4


def math_flops(num_elements: int) -> float:
    return float((18 * _N**4 + 2 * _N**3) * num_elements)


def theory_bytes(num_elements: int, dtype_size: int = 8) -> float:
    return float(2 * _N**3 * num_elements * dtype_size)


def _laplace_apply(u, stiffness_1d, mass_1d, block_size_data, block_size_exec):
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
    M_row4 = mass_1d[4, :].contiguous()
    D_row4 = stiffness_1d[4, :].contiguous()

    out = torch.empty_like(u)
    _laplace_kernel_mixed_p4_codegen[(grid,)](
        u, O_low4, O_col4, M_row4, D_row4, out,
        num_elements,
        N=_N,
        N_MMA=_N_MMA,
        BLOCK_SIZE_DATA=block_size_data,
        BLOCK_SIZE_EXEC=block_size_exec,
        num_warps=2,
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


# ---------------------------------------------------------------------------
# nsight roofline profile (mirrors bench/volume/bench_BXYZE.py)
# ---------------------------------------------------------------------------

def _derive(
    time_ns: float,
    dram_bytes: float,
    fp32_all: float,
    fp32_fma: float,
    fp64_all: float,
    fp64_fma: float,
    tc_dmma: float,
    sol_sm_pct: float,
    sol_dram_pct: float,
    log2_num_elements: int,
    block_size_data: int,
    block_size_exec: int,
) -> dict:
    from tensorfem.utils.profiler import derive_roofline_metrics
    num_elements = 2 ** log2_num_elements
    mfl = math_flops(num_elements)
    tb = theory_bytes(num_elements)
    d = derive_roofline_metrics(
        time_ns, dram_bytes, fp32_all, fp32_fma, fp64_all, fp64_fma,
        tc_dmma, sol_sm_pct, sol_dram_pct,
        theory_flops=mfl,
        theory_bytes=tb,
    )
    d["ai_math"] = mfl / max(tb, 1.0)
    d["gdofs_per_s"] = (_N ** 3 * num_elements) / (time_ns * 1e-9) / 1e9
    return d


# ---- Deep diagnostic metrics for SM-bottleneck analysis ----
# Order MUST match the positional args of _derive_deep below.
_DEEP_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",                                              # 0
    # Shared memory:
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",            # 1  smem load bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",            # 2  smem store bank conflicts
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",                # 3  total smem load wavefronts
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",                # 4  total smem store wavefronts
    # Local memory (= register spill):
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",                         # 5  spill loads (bytes)
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",                         # 6  spill stores (bytes)
    # Pipe activity:
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",     # 7  shared (fp64+tc) pipe %
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",                 # 8  L1TEX %
    # Stall reasons (proportion-of-warps-per-cycle):
    "smsp__average_warps_active_per_inst_executed.ratio",                  # 9  warps in flight per inst
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio",      # 10 LMEM/SMEM data dep
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio",     # 11 MIO / register dep
    "smsp__warp_issue_stalled_wait_per_warp_active.ratio",                 # 12 fixed-latency pipe wait
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio",         # 13 MIO queue full
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio",   # 14 math pipe back-pressure
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.ratio",          # 15 LSU queue full
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.ratio",       # 16 i-cache miss
    "smsp__warp_issue_stalled_barrier_per_warp_active.ratio",              # 17 CTA-wide barrier wait
    "smsp__warp_issue_stalled_membar_per_warp_active.ratio",               # 18 membar wait
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.ratio",       # 19 dispatch stall
    "smsp__warp_issue_stalled_not_selected_per_warp_active.ratio",         # 20 not selected by scheduler
    "smsp__warp_issue_stalled_selected_per_warp_active.ratio",             # 21 selected (this is the "good" fraction)
)


def _combine_deep(a, b):
    """Sum across kernel instances; for ratio metrics take a weighted mean.

    For the simple PoC we sum everything — the kernel is a single tl-jit call
    so there's typically only one kernel anyway.
    """
    return a + b


def _derive_deep(
    time_ns,
    smem_ld_bc, smem_st_bc, smem_ld_wf, smem_st_wf,
    spill_ld_b, spill_st_b,
    shared_pipe_pct, l1tex_pct,
    warps_per_inst,
    stall_long_sb, stall_short_sb, stall_wait, stall_mio, stall_math,
    stall_lg, stall_noinst, stall_barrier, stall_membar, stall_dispatch,
    stall_notsel, stall_selected,
    log2_num_elements, block_size_data, block_size_exec,
) -> dict:
    """Summarize SM-bottleneck signals into one row."""
    # Bank conflict rate = conflicts / wavefronts
    smem_ld_bcr = smem_ld_bc / max(smem_ld_wf, 1.0)
    smem_st_bcr = smem_st_bc / max(smem_st_wf, 1.0)
    total_smem_wf = smem_ld_wf + smem_st_wf
    total_smem_bc = smem_ld_bc + smem_st_bc
    total_spill_bytes = spill_ld_b + spill_st_b
    # Normalize stall ratios to percentages.
    return {
        "smem_ld_bank_conflict_pct": 100.0 * smem_ld_bcr,
        "smem_st_bank_conflict_pct": 100.0 * smem_st_bcr,
        "smem_total_wavefronts": total_smem_wf,
        "smem_total_bank_conflicts": total_smem_bc,
        "spill_total_bytes": total_spill_bytes,
        "shared_pipe_pct": shared_pipe_pct,
        "l1tex_pct": l1tex_pct,
        "warps_in_flight_per_inst": warps_per_inst,
        "stall_long_scoreboard_pct": 100.0 * stall_long_sb,
        "stall_short_scoreboard_pct": 100.0 * stall_short_sb,
        "stall_wait_pct": 100.0 * stall_wait,
        "stall_mio_throttle_pct": 100.0 * stall_mio,
        "stall_math_throttle_pct": 100.0 * stall_math,
        "stall_lg_throttle_pct": 100.0 * stall_lg,
        "stall_no_instruction_pct": 100.0 * stall_noinst,
        "stall_barrier_pct": 100.0 * stall_barrier,
        "stall_membar_pct": 100.0 * stall_membar,
        "stall_dispatch_pct": 100.0 * stall_dispatch,
        "stall_not_selected_pct": 100.0 * stall_notsel,
        "selected_pct": 100.0 * stall_selected,
    }


def _build_inputs(log2_num_elements, block_size_data):
    N = _N
    num_elements = 2 ** log2_num_elements
    n_blocks = num_elements // block_size_data
    u = torch.randn(n_blocks, N, N, N, block_size_data, dtype=torch.float64, device="cuda")
    stiffness = torch.randn(N, N, dtype=torch.float64, device="cuda")
    mass = torch.randn(N, N, dtype=torch.float64, device="cuda")
    return u, stiffness, mass


def _profile_roofline(log2_num_elements: int, block_size_data: int, block_size_exec: int) -> None:
    """Original roofline pass: bandwidth, FLOPs, SoL%."""
    import nsight
    import nsight.analyze
    from tensorfem.utils.profiler import STANDARD_METRICS, combine_kernel_metrics

    @nsight.analyze.kernel(
        metrics=list(STANDARD_METRICS),
        combine_kernel_metrics=combine_kernel_metrics,
        replay_mode="kernel",
        derive_metric=_derive,
    )
    def _profile_laplace(log2_num_elements, block_size_data, block_size_exec):
        u, s, m = _build_inputs(log2_num_elements, block_size_data)
        _laplace_apply(u, s, m, block_size_data, block_size_exec)
        torch.cuda.synchronize()
        with nsight.annotate("laplace_mixed_p4"):
            _laplace_apply(u, s, m, block_size_data, block_size_exec)
            torch.cuda.synchronize()

    print("[profile/roofline] Launching ncu...")
    results = _profile_laplace(
        configs=[(log2_num_elements, block_size_data, block_size_exec)]
    )
    _print_results(results, "roofline", [
        "tflops_measured", "tflops_theory", "bandwidth_gb_s",
        "ai_measured", "ai_theory", "ai_math",
        "sol_sm_pct", "sol_dram_pct", "gdofs_per_s",
    ])


def _profile_deep(log2_num_elements: int, block_size_data: int, block_size_exec: int) -> None:
    """Second pass: shared-memory bank conflicts, spill, warp stall breakdown."""
    import nsight
    import nsight.analyze

    @nsight.analyze.kernel(
        metrics=list(_DEEP_METRICS),
        combine_kernel_metrics=_combine_deep,
        replay_mode="kernel",
        derive_metric=_derive_deep,
    )
    def _profile_laplace_deep(log2_num_elements, block_size_data, block_size_exec):
        u, s, m = _build_inputs(log2_num_elements, block_size_data)
        _laplace_apply(u, s, m, block_size_data, block_size_exec)
        torch.cuda.synchronize()
        with nsight.annotate("laplace_mixed_p4_deep"):
            _laplace_apply(u, s, m, block_size_data, block_size_exec)
            torch.cuda.synchronize()

    print("[profile/deep] Launching ncu (extra pass, ~30-60s)...")
    results = _profile_laplace_deep(
        configs=[(log2_num_elements, block_size_data, block_size_exec)]
    )
    _print_results(results, "deep", [
        "smem_ld_bank_conflict_pct", "smem_st_bank_conflict_pct",
        "smem_total_wavefronts", "smem_total_bank_conflicts",
        "spill_total_bytes",
        "shared_pipe_pct", "l1tex_pct",
        "warps_in_flight_per_inst",
        "selected_pct",
        "stall_long_scoreboard_pct", "stall_short_scoreboard_pct",
        "stall_wait_pct", "stall_mio_throttle_pct", "stall_math_throttle_pct",
        "stall_lg_throttle_pct", "stall_no_instruction_pct",
        "stall_barrier_pct", "stall_membar_pct",
        "stall_dispatch_pct", "stall_not_selected_pct",
    ])
    print()
    print("Deep-profile legend:")
    print("  smem_ld/st_bank_conflict_pct  — % of smem wavefronts that hit bank conflicts")
    print("                                  >5% is bad, >20% is dominant")
    print("  spill_total_bytes             — register spill traffic (local memory). 0 is ideal.")
    print("  shared_pipe_pct               — utilization of the FP64+TensorCore shared pipe")
    print("  l1tex_pct                     — L1TEX (shared+L1 cache) pipe utilization")
    print("  warps_in_flight_per_inst      — A100 max = 64 (per SMSP). <8 → low occupancy")
    print("  selected_pct                  — fraction of cycles a warp was issued; the higher")
    print("                                  the better. The rest is divided among stall_*.")
    print("  stall_long_scoreboard_pct     — waiting on global/local memory (data dep)")
    print("  stall_short_scoreboard_pct    — waiting on MIO / shared mem / register dep")
    print("  stall_wait_pct                — waiting on fixed-latency pipes (DMMA, FP64)")
    print("  stall_mio_throttle_pct        — MIO queue full (too many LSU / SFU in flight)")
    print("  stall_lg_throttle_pct         — LSU queue full (global mem op stack-up)")
    print("  stall_math_throttle_pct       — math pipe back-pressure")
    print()
    print("Typical signatures:")
    print("  HIGH short_scoreboard + HIGH smem_bank_conflict → fix bank conflicts")
    print("  HIGH long_scoreboard                            → memory-bound; need more reuse")
    print("  HIGH wait + LOW SoL                             → MMA latency; need more warps")
    print("  HIGH spill_total_bytes                          → register spill; reduce BE or hoisted vars")


def _print_results(results, label: str, metric_keys) -> None:
    if results is None:
        # We're inside the ncu subprocess — the collector returns None.
        # Real printing happens in the controller process.
        return
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    df = results.to_dataframe()
    sub = df[df["Metric"].isin(metric_keys)][
        ["log2_num_elements", "block_size_data", "block_size_exec", "Metric", "AvgValue"]
    ]
    # Preserve a sensible order
    order = {m: i for i, m in enumerate(metric_keys)}
    sub = sub.assign(_o=sub["Metric"].map(order)).sort_values("_o").drop(columns="_o")
    print(f"\n--- nsight {label} results (mixed p=4 codegen) ---")
    print(sub.to_string(index=False))


def _profile(log2_num_elements: int, block_size_data: int, block_size_exec: int, *, deep: bool) -> None:
    # NSPY_NCU_PROFILE is set inside ncu's replay subprocess (its value is a
    # hash of the target function+config — we can't infer "which pass" from
    # it). The nsight decorator handles the signature matching internally:
    # when the inside-subprocess wrapper sees a non-matching signature it
    # just runs the inner function inline (no profiling). So both passes
    # being called sequentially is safe — exactly one of them gets the live
    # ncu attachment per subprocess.
    _profile_roofline(log2_num_elements, block_size_data, block_size_exec)
    if deep:
        _profile_deep(log2_num_elements, block_size_data, block_size_exec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--log2-elements", type=int, default=20)
    parser.add_argument("--data-block", type=int, default=8)
    parser.add_argument("--exec-block", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Run nsight roofline profile (requires ncu).")
    parser.add_argument("--deep", action="store_true",
                        help="Add a second ncu pass with shared-memory bank conflict + "
                             "warp-stall breakdown metrics. Implies --profile.")
    args = parser.parse_args()
    if args.deep:
        args.profile = True

    N = _N
    num_elements = 2 ** args.log2_elements
    BD, BE = args.data_block, args.exec_block
    if num_elements % BD != 0:
        raise ValueError(f"num_elements={num_elements} not divisible by data-block={BD}")
    n_blocks = num_elements // BD

    device = torch.cuda.current_device()
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"P=4 N={N}  E=2^{args.log2_elements}={num_elements}")
    print(f"data_block={BD}  exec_block={BE}  n_blocks={n_blocks}  kernel=mixed_p4_codegen")
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
    _laplace_apply(u, stiffness, mass, BD, BE)
    torch.cuda.synchronize()
    print(f"Compilation time: {(time.perf_counter()-t0)*1000:.1f} ms")

    if args.verify:
        out_triton = _laplace_apply(u, stiffness, mass, BD, BE)
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

    ms = gpu_timeit(lambda: _laplace_apply(u, stiffness, mass, BD, BE),
                    warmup=args.warmup, iters=args.iters)
    tflops_math = mfl / (ms * 1e-3) / 1e12
    bw = tb / (ms * 1e-3) / 1e9
    dofs = N**3 * num_elements
    gdofs_s = dofs / (ms * 1e-3) / 1e9
    print(f"Timing ({args.iters} iters): avg = {ms:.3f} ms")
    print(f"  TFLOPS (math):  {tflops_math:.3f}")
    print(f"  Bandwidth:      {bw:.1f} GB/s")
    print(f"  Throughput:     {gdofs_s:.3f} GDoF/s")

    if args.profile:
        print()
        _profile(args.log2_elements, BD, BE, deep=args.deep)
    else:
        print()
        print("(Run with --profile [--deep] to launch nsight measurement.)")


if __name__ == "__main__":
    main()
