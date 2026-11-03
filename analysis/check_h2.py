#!/usr/bin/env python3
"""
check_h2.py -- Gate 1 first look at H2 for one laptop.

Reads a2_bandwidth.csv and b_matmul.csv from data/raw/<machine_id>/ and prints
the two roofline ceilings and how GEMV compares with the bandwidth prediction:
    B       = faster of median read_sum8 and median read_sum16, runs 1-5 at
              256 MiB per array (GB/s)          memory ceiling (see bench/bandwidth.c)
    P_peak  = highest median GEMM (GFLOP/s)     compute ceiling
    predicted GEMV = 0.5 FLOP/byte x B
A2 runs every size forward and then backward (run_all.py), so each size has
runs 1-5 and 6-10. B uses only runs 1-5 at 256 MiB, measured exactly as before
the extra sizes existed. Sizes are never pooled. The comparison between sizes
(B at each size and GEMV / predicted with B at each size) uses all runs of a size.
Rows whose matrix (4 n^2 bytes) is larger than the laptop's last-level cache
(llc_mib in data/machines.csv) are marked H2; smaller ones are served partly
from cache. A machine without llc_mib falls back to n >= 8192. The "GEMV beat
the ceiling" WARNING checks only rows with n >= 8192.

Usage: python analysis/check_h2.py data/raw/<machine_id>
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_SIZE = 256 << 20                  # A2 bytes per array that define B
B_RUNS = 5                          # B uses runs 1..B_RUNS (the first pass)
OTHER_SIZES = (512 << 20, 1024 << 20)
FALLBACK_N = 8192                   # H2 rows when the machine has no llc_mib
WARN_N = 8192                       # the WARNING checks GEMV rows with n >= WARN_N


def medians(path, kernels, key_col, max_run=None):
    """{(kernel, int(row[key_col])): (median, number of runs)} over rows whose kernel
    is in kernels and, if max_run is given, whose run is <= max_run."""
    vals = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["kernel"] in kernels and (max_run is None or int(row["run"]) <= max_run):
                vals[(row["kernel"], int(row[key_col]))].append(float(row["value"]))
    return {k: (statistics.median(v), len(v)) for k, v in vals.items()}


def faster_read(a2):
    """{size_bytes: (faster read kernel, median GB/s, runs)} from medians() of the read kernels."""
    bw = {}
    for (k, size), (v, runs) in a2.items():
        if size not in bw or v > bw[size][1]:
            bw[size] = (k, v, runs)
    return bw


def llc_bytes(machine):
    """llc_mib from data/machines.csv in bytes; None if the machine or value is missing."""
    try:
        with open(os.path.join(ROOT, "data", "machines.csv"), newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["machine_id"] == machine:
                    return float(row["llc_mib"]) * 2**20
    except (OSError, KeyError, ValueError):
        pass
    return None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python analysis/check_h2.py data/raw/<machine_id>")
    d = sys.argv[1]
    machine = os.path.basename(os.path.normpath(d))
    a2_path = os.path.join(d, "a2_bandwidth.csv")
    reads = ("read_sum8", "read_sum16")
    roof = faster_read(medians(a2_path, reads, "size_bytes", max_run=B_RUNS))
    bw = faster_read(medians(a2_path, reads, "size_bytes"))      # all runs, per size
    b = {k: v for k, (v, _) in medians(os.path.join(d, "b_matmul.csv"),
                                        ("gemv_fp32", "gemm_fp32"), "n").items()}

    if B_SIZE not in roof:
        print(f"\nWARNING: {machine}: no A2 read rows (runs 1-{B_RUNS}) at {B_SIZE >> 20} MiB "
              "per array (skipped for lack of RAM?), so B is undefined. This run is not data.")
        return
    b_kernel, B, _ = roof[B_SIZE]

    peak = max(v for (k, _), v in b.items() if k == "gemm_fp32")
    pred = 0.5 * B
    print(f"\nRoofline ceilings for {machine}")
    print(f"  B      = {B:7.2f} GB/s     (A2 {b_kernel}, median of runs 1-{B_RUNS} at "
          f"{B_SIZE >> 20} MiB; the faster read kernel)")
    print("  Between sizes (all runs of each size, not used for the roofline):")
    for size in sorted(set(bw) | {B_SIZE, *OTHER_SIZES}):
        if size in bw:
            k, v, runs = bw[size]
            print(f"           {v:7.2f} GB/s     at {size >> 20:4d} MiB per array "
                  f"({k}, {runs} runs)")
        else:
            print(f"              -          at {size >> 20:4d} MiB per array (not measured)")
    print(f"  P_peak = {peak:7.2f} GFLOP/s  (best GEMM median)")
    print(f"  ridge point = P_peak / B = {peak / B:.1f} FLOP/byte\n")

    llc = llc_bytes(machine)
    if llc is None:
        print(f"NOTE: {machine} has no llc_mib in data/machines.csv; "
              f"H2 rows fall back to n >= {FALLBACK_N}.")
    tests_h2 = (lambda n: 4 * n * n > llc) if llc is not None else (lambda n: n >= FALLBACK_N)

    sizes = sorted(bw)                        # A2 sizes in this run, 256 MiB first
    print(f"H2 first look: predicted GEMV = 0.5 x B = {pred:.2f} GFLOP/s")
    print("  vs B = GEMV / predicted (roofline B); B@size = GEMV / (0.5 x B at that size, all runs)")
    print("  H2 = row tests H2 (matrix larger than the LLC)")
    print(f"  {'n':>5}  {'matrix':>8}  {'GEMV GFLOP/s':>12}  {'implied GB/s':>12}  {'vs B':>6}"
          + "".join(f"  {f'B@{s >> 20} MiB':>11}" for s in sizes))
    over = []
    for (k, n), v in sorted(b.items()):
        if k != "gemv_fp32" or n < 1024:
            continue
        mib = 4 * n * n / 2**20
        ratios = "".join(f"  {v / (0.5 * bw[s][1]):11.2f}" for s in sizes)
        print(f"  {n:5d}  {mib:5.0f} MiB  {v:12.2f}  {2 * v:12.2f}  {v / pred:6.2f}{ratios}"
              f"{'  H2' if tests_h2(n) else ''}")
        if n >= WARN_N and v / pred > 1.05:
            over.append(f"n = {n} ({v / pred:.2f})")
    if llc is not None:
        print(f"\n  Only rows whose matrix is larger than this laptop's last-level cache "
              f"({llc / 2**20:g} MiB) test H2.")
    else:
        print(f"\n  Only rows with n >= {FALLBACK_N} test H2 (no llc_mib for {machine}).")
    if not any(k == "gemv_fp32" and n >= WARN_N for (k, n) in b):
        print(f"  Quick run: no GEMV with n >= {WARN_N}, so the ceiling check is skipped.")
    elif over:
        print(f"  WARNING: GEMV beat the bandwidth 'ceiling' by more than 5% at {', '.join(over)}.\n"
              "  B is understated (A2 loop or build flags), not an H2 pass. Fix before Gate 1.")


if __name__ == "__main__":
    main()
