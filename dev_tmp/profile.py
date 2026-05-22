"""
profile.py — deep diagnostic for the fp64 matmul residual gap to cuBLAS.

Context
-------
Triton fp64 matmul (BM=64,BN=64,BK=16,nw=2,ns=3) sits ~4-5% below cuBLAS
on H100. Static analysis (SASS/ELF) ruled out spilling, occupancy, cp.async
depth, and BK>16. The remaining gap is suspected to be DMMA/LDGSTS scheduling
— needs HW counters to confirm. This script collects everything needed.

Modes
-----
1. Timing + static analysis only (no profiler):
       python profile.py [--size 2048] [--config best]

2. Full nsight profile (Triton + cuBLAS baseline, every metric, CSV out):
       python profile.py --profile [--size 2048] [--config best]
       python profile.py --profile --config all

Output
------
With --profile, every metric is dumped to a CSV per (config, kernel) into
--out-dir (default: ./profile_out/). Filenames encode the run settings:
       prof_<gpu>_N<size>_<config>_BM<>BN<>BK<>_nw<>_ns<>_<triton|cublas>.csv

Note: profiling uses the nsight Python API (everything inside the script,
no external ncu invocation needed). Requires RmProfilingAdminOnly=0:
       echo 0 | sudo tee /proc/driver/nvidia/params

Configs probed
--------------
  best   BM=64,BN=64,BK=16,nw=2,ns=3   [~54 TFLOPS H100, true best]
  ns4    BM=64,BN=64,BK=16,nw=2,ns=4   [~54 TFLOPS variant]
  bm128  BM=128,BN=128,BK=16,nw=8,ns=3 [~52 TFLOPS, autotune picks this]
  small  BM=32,BN=64,BK=16,nw=2,ns=4   [~48 TFLOPS smaller tile]
  spill  BM=64,BN=64,BK=32,nw=2,ns=3   [~39 TFLOPS, BK=32 spills!]

What to look for
----------------
  dmma_util_pct > 85%    → near peak DMMA throughput; gap is fundamental
  dmma_util_pct < 70%    → scheduling gap; check stall breakdown

  stall_wait_pct high    → waiting on fixed-latency pipes (DMMA, FP64);
                           DMMA latency not hidden; need more warps or ILP
  stall_long_sb_pct high → global-memory latency not hidden; try more stages
  stall_short_sb_pct high→ MIO / shmem / register dep
  stall_mio_pct high     → MIO queue full; check bank_conf
  smem_bank_conflict > 5%→ Layer 1 swizzle fix didn't apply to this config!

  Compare "best" (BM=64,nw=2) vs "bm128" (BM=128,nw=8):
  Look for which config has better stall_wait/dmma_util — that's the
  scheduling story and explains why smaller tiles beat larger ones here.
"""

from __future__ import annotations
import argparse
import os

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ── configs to probe ────────────────────────────────────────────────────────
# Each entry: (label, BM, BN, BK, nw, ns)
# "best" = the 54.4 TFLOPS winner on H100 GPU3.

PROBE_CONFIGS = {
    "best":   ("BM64 BN64 BK16 nw2 ns3 [best ~54 TFLOPS H100]",   64,  64, 16, 2, 3),
    "ns4":    ("BM64 BN64 BK16 nw2 ns4 [ns4 variant ~54 TFLOPS]",  64,  64, 16, 2, 4),
    "bm128":  ("BM128 BN128 BK16 nw8 ns3 [autotune picks ~52T]",  128, 128, 16, 8, 3),
    "small":  ("BM32 BN64 BK16 nw2 ns4 [smaller tile ~48 TFLOPS]", 32,  64, 16, 2, 4),
    "spill":  ("BM64 BN64 BK32 nw2 ns3 [BK=32 spills! ~39 TFLOPS]", 64, 64, 32, 2, 3),
}


# ── kernel (self-contained, no autotune to keep profiler captures clean) ───

@triton.jit
def matmul_kernel_fp64_profile(
    a_ptr, b_ptr, c_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr,
):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    npg = GM * npn
    gid = pid // npg; fpm = gid * GM
    gsm = min(npm - fpm, GM)
    pm = fpm + (pid % npg) % gsm
    pn = (pid % npg) // gsm
    tl.assume(pm >= 0); tl.assume(pn >= 0)
    tl.assume(sam > 0); tl.assume(sak > 0)
    tl.assume(sbn > 0); tl.assume(sbk > 0)
    tl.assume(scm > 0); tl.assume(scn > 0)
    ra = (pm * BM + tl.arange(0, BM)) % M
    rb = (pn * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    ap = a_ptr + ra[:, None] * sam + rk[None, :] * sak
    bp = b_ptr + rk[:, None] * sbk + rb[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float64)
    for kk in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(
            tl.load(ap, mask=rk[None, :] < K - kk * BK, other=0.0),
            tl.load(bp, mask=rk[:, None] < K - kk * BK, other=0.0),
            acc, out_dtype=tl.float64,
        )
        ap += BK * sak
        bp += BK * sbk
    rc = pm * BM + tl.arange(0, BM)
    qc = pn * BN + tl.arange(0, BN)
    tl.store(c_ptr + scm * rc[:, None] + scn * qc[None, :], acc,
             mask=(rc[:, None] < M) & (qc[None, :] < N))


def _build_inputs(size):
    M = N = K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float64)
    return a, b, c


def launch(a, b, c, BM, BN, BK, nw, ns):
    M, K = a.shape; _, N = b.shape
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_kernel_fp64_profile[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, GM=8,
        num_warps=nw, num_stages=ns,
    )


# ── timing ──────────────────────────────────────────────────────────────────

def time_kernel(a, b, c, BM, BN, BK, nw, ns, warmup=5) -> float:
    """Returns median kernel time in ms."""
    shmem = ns * (BM * BK + BK * BN) * 8
    if shmem > 232448:
        return float("nan")
    for _ in range(warmup):
        launch(a, b, c, BM, BN, BK, nw, ns)
    torch.cuda.synchronize()
    ms, _, _ = triton.testing.do_bench(
        lambda: launch(a, b, c, BM, BN, BK, nw, ns),
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms


def tflops(M, N, K, ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)



# ── nsight profiling — FULL pass (everything, raw dump) ─────────────────────
# One mega-list covering: roofline, stalls, bank conflicts, spill, issue
# rates, instruction mix, occupancy, scheduler stats, L2, DRAM. Dumped raw
# so nothing is lost. Used for cloud H100 sessions where re-running is costly.

_FULL_METRICS: tuple[str, ...] = (
    # ── timing / SoL ──────────────────────────────────────────────────────
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpc__cycles_elapsed.avg.per_second",
    # ── DMMA / tensor pipe ────────────────────────────────────────────────
    "sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_dmma.sum",
    "sm__inst_executed_pipe_tensor_op_dmma.avg.per_cycle_active",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active",
    # ── FP64 (non-tensor) pipe (for sanity) ───────────────────────────────
    "sm__inst_executed_pipe_fp64.sum",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    # ── issue rate / instruction mix ──────────────────────────────────────
    "sm__inst_issued.sum",
    "sm__inst_issued.avg.per_cycle_active",
    "smsp__inst_executed.sum",
    "smsp__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "sm__inst_executed_pipe_lsu.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_xu.sum",
    "sm__inst_executed_pipe_adu.sum",
    # ── memory: global / cp.async / LDGSTS ────────────────────────────────
    "smsp__inst_executed_op_global_ld.sum",
    "smsp__inst_executed_op_global_st.sum",
    "smsp__inst_executed_op_ldgsts.sum",
    # ── shared memory ─────────────────────────────────────────────────────
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    # ── L1 / L2 / DRAM ────────────────────────────────────────────────────
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_bytes.sum",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # ── spill / local mem ─────────────────────────────────────────────────
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
    # ── occupancy / warps ─────────────────────────────────────────────────
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__average_warps_active_per_inst_executed.ratio",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__thread_count",
    "launch__block_size",
    "launch__grid_size",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    # ── scheduler / eligibility ───────────────────────────────────────────
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_issue_stalled_barrier.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    # ── stall reasons (full set) ──────────────────────────────────────────
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio",
    "smsp__warp_issue_stalled_wait_per_warp_active.ratio",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio",
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.ratio",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.ratio",
    "smsp__warp_issue_stalled_imc_miss_per_warp_active.ratio",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.ratio",
    "smsp__warp_issue_stalled_barrier_per_warp_active.ratio",
    "smsp__warp_issue_stalled_membar_per_warp_active.ratio",
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.ratio",
    "smsp__warp_issue_stalled_drain_per_warp_active.ratio",
    "smsp__warp_issue_stalled_misc_per_warp_active.ratio",
    "smsp__warp_issue_stalled_sleeping_per_warp_active.ratio",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.ratio",
    "smsp__warp_issue_stalled_selected_per_warp_active.ratio",
)


def _combine_full(a, b):
    return a + b


def _derive_full(*args):
    # last two positional args are (size, config_key); everything else are
    # raw metric values in declaration order of _FULL_METRICS.
    metric_values = args[:-2]
    out = {}
    for name, val in zip(_FULL_METRICS, metric_values):
        out[name] = val
    # add a couple of trivially-derived conveniences:
    time_ns = out.get("gpu__time_duration.sum", 0.0) or 0.0
    if time_ns > 0:
        out["_derived_time_us"] = time_ns / 1e3
    return out


def _gpu_short_name() -> str:
    name = torch.cuda.get_device_name(0)
    # "NVIDIA A100-SXM4-80GB" → "A100", "NVIDIA H100 80GB HBM3" → "H100"
    for tag in ("H100", "H200", "A100", "A800", "L40", "L4", "B200"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def _run_full_profile(label: str, run_fn, out_csv: str | None) -> None:
    """Profile `run_fn` (zero-arg callable that launches one kernel) with
    the full metric set, print, and optionally save CSV."""
    import nsight
    import nsight.analyze

    @nsight.analyze.kernel(
        metrics=list(_FULL_METRICS),
        combine_kernel_metrics=_combine_full,
        replay_mode="kernel",
        derive_metric=_derive_full,
    )
    def _runner(label):
        run_fn()      # warmup
        torch.cuda.synchronize()
        with nsight.annotate(f"full_{label}"):
            run_fn()
            torch.cuda.synchronize()

    print(f"[profile] {label} (~60-120s) ...")
    results = _runner(configs=[(label,)])
    if results is None:
        return

    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    df = results.to_dataframe()
    sub = df[["label", "Metric", "AvgValue"]].copy()
    order = {m: i for i, m in enumerate(_FULL_METRICS)}
    sub["_o"] = sub["Metric"].map(lambda m: order.get(m, 1_000_000))
    sub = sub.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    print(f"\n--- metrics: {label} ---")
    print(sub.to_string(index=False))
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"  saved → {out_csv}")


def _csv_path(out_dir: str, size: int, config_key: str, kernel: str) -> str:
    _, BM, BN, BK, nw, ns = PROBE_CONFIGS[config_key]
    gpu = _gpu_short_name()
    fname = (f"prof_{gpu}_N{size}_{config_key}"
             f"_BM{BM}BN{BN}BK{BK}_nw{nw}_ns{ns}_{kernel}.csv")
    return os.path.join(out_dir, fname)


def profile_full_triton(size: int, config_key: str, out_dir: str) -> None:
    a, b, c = _build_inputs(size)
    _, BM, BN, BK, nw, ns = PROBE_CONFIGS[config_key]
    _run_full_profile(
        f"triton_{config_key}",
        lambda: launch(a, b, c, BM, BN, BK, nw, ns),
        _csv_path(out_dir, size, config_key, "triton"),
    )


def profile_full_cublas(size: int, config_key: str, out_dir: str) -> None:
    M = N = K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)
    _run_full_profile(
        "cublas",
        lambda: torch.matmul(a, b),
        _csv_path(out_dir, size, config_key, "cublas"),
    )


# ── report helpers ───────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=2048,
                        help="Matrix size M=N=K (default 2048)")
    parser.add_argument("--config", default="best",
                        choices=list(PROBE_CONFIGS) + ["all"],
                        help="Which config(s) to probe (default: best)")
    parser.add_argument("--profile", action="store_true",
                        help="Run full nsight profile (Triton + cuBLAS baseline). "
                             "Dumps every metric to CSV. Requires ncu permission.")
    parser.add_argument("--out-dir", type=str, default="profile_out",
                        help="Directory to save per-config CSVs (default: profile_out/)")
    args = parser.parse_args()

    M = N = K = args.size
    configs = (list(PROBE_CONFIGS.items())
               if args.config == "all"
               else [(args.config, PROBE_CONFIGS[args.config])])

    # ── environment info ────────────────────────────────────────────────────
    print_section("Environment")
    gpu_name = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability(0)
    print(f"  GPU              : {gpu_name}  (sm_{sm[0]}{sm[1]})")
    print(f"  Matrix size      : {M}×{N}×{K}  (fp64)")
    print(f"  Profile mode     : {'full (Triton + cuBLAS, CSV out)' if args.profile else 'timing + static only'}")
    if args.profile:
        print(f"  Output dir       : {args.out_dir}/")

    # ── cuBLAS reference ────────────────────────────────────────────────────
    print_section("cuBLAS reference timing")
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float64)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float64)
    cu_ms, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                          quantiles=[0.5, 0.2, 0.8])
    cu_tflops = tflops(M, N, K, cu_ms)
    print(f"  cuBLAS           : {cu_ms:.3f} ms  →  {cu_tflops:.2f} TFLOPS")

    # ── per-config analysis ─────────────────────────────────────────────────
    results = {}
    for key, (label, BM, BN, BK, nw, ns) in configs:
        print_section(f"Config: {key}  —  {label}")
        a, b, c = _build_inputs(M)
        shmem = ns * (BM * BK + BK * BN) * 8
        shmem_kb = shmem // 1024

        if shmem > 232448:
            print(f"  SKIP: shmem {shmem_kb} KB exceeds hardware limit (228 KB)")
            continue

        # Compile + warmup
        print(f"  Compiling / warming up … ", end="", flush=True)
        try:
            for _ in range(3):
                launch(a, b, c, BM, BN, BK, nw, ns)
            torch.cuda.synchronize()
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Correctness check
        ref = torch.matmul(a, b)
        ok = torch.allclose(c, ref, atol=1e-8, rtol=1e-5)
        print(f"  Correctness      : {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  Max diff         : {(c - ref).abs().max().item():.2e}")

        # Timing
        ms = time_kernel(a, b, c, BM, BN, BK, nw, ns)
        tf = tflops(M, N, K, ms)
        gap = (cu_tflops - tf) / cu_tflops * 100
        print(f"  Timing           : {ms:.3f} ms  →  {tf:.2f} TFLOPS  "
              f"({gap:+.1f}% vs cuBLAS {cu_tflops:.2f} T)")

        results[key] = {
            "label": label, "BM": BM, "BN": BN, "BK": BK, "nw": nw, "ns": ns,
            "shmem_kb": shmem_kb,
            "ms": ms, "tflops": tf, "gap_pct": gap,
            "cublas_tflops": cu_tflops,
        }

        # nsight profile: full pass for Triton, plus matching cuBLAS baseline
        if args.profile:
            profile_full_triton(M, key, args.out_dir)
            profile_full_cublas(M, key, args.out_dir)

    # ── summary table ───────────────────────────────────────────────────────
    if len(results) > 1:
        print_section("Summary")
        print(f"  {'key':8s} {'TFLOPS':>8} {'gap':>7}")
        print(f"  {'-'*8} {'-'*8} {'-'*7}")
        for key, r in results.items():
            print(f"  {key:8s} {r['tflops']:>8.2f} {r['gap_pct']:>+6.1f}%")


if __name__ == "__main__":
    main()
