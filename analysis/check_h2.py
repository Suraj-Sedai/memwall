#!/usr/bin/env python3
"""
check_h2.py -- Gate 1 first look at H2 for one laptop.

Reads a2_bandwidth.csv and b_matmul.csv from data/raw/<machine_id>/ and prints
the two roofline ceilings and how GEMV compares with the bandwidth prediction:
    B       = median read_sum (GB/s)            memory ceiling
    P_peak  = highest median GEMM (GFLOP/s)     compute ceiling
    predicted GEMV = 0.5 FLOP/byte x B
Only GEMV sizes whose matrix (4 n^2 bytes) is larger than the laptop's
last-level cache test H2; smaller ones are served partly from cache.

Usage: python analysis/check_h2.py data/raw/<machine_id>
"""
import csv
import os
import statistics
import sys
from collections import defaultdict


def medians(path, kernel_filter):
    vals = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if kernel_filter(row["kernel"]):
                key = (row["kernel"], int(row["n"]) if row["n"] else None)
                vals[key].append(float(row["value"]))
    return {k: statistics.median(v) for k, v in vals.items()}


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python analysis/check_h2.py data/raw/<machine_id>")
    d = sys.argv[1]
    a2 = medians(os.path.join(d, "a2_bandwidth.csv"), lambda k: k == "read_sum")
    b = medians(os.path.join(d, "b_matmul.csv"), lambda k: k in ("gemv_fp32", "gemm_fp32"))

    bw = a2[("read_sum", None)]
    peak = max(v for (k, _), v in b.items() if k == "gemm_fp32")
    pred = 0.5 * bw
    print(f"\nRoofline ceilings for {os.path.basename(os.path.normpath(d))}")
    print(f"  B      = {bw:7.2f} GB/s     (A2 read_sum, median)")
    print(f"  P_peak = {peak:7.2f} GFLOP/s  (best GEMM median)")
    print(f"  ridge point = P_peak / B = {peak / bw:.1f} FLOP/byte\n")
    print(f"H2 first look: predicted GEMV = 0.5 x B = {pred:.2f} GFLOP/s")
    print(f"  {'n':>5}  {'matrix':>8}  {'GEMV GFLOP/s':>12}  {'implied GB/s':>12}  {'GEMV / predicted':>16}")
    n_max = max(n for (k, n) in b if k == "gemv_fp32")
    worst = 0.0
    for (k, n), v in sorted(b.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if k != "gemv_fp32" or n < 1024:
            continue
        mib = 4 * n * n / 2**20
        ratio = v / pred
        print(f"  {n:5d}  {mib:5.0f} MiB  {v:12.2f}  {2 * v:12.2f}  {ratio:16.2f}")
        if n == n_max:          # 256 MiB in a full run: larger than any laptop's cache
            worst = ratio
    print("\n  Only rows whose matrix is larger than this laptop's last-level cache test H2.")
    if n_max < 8192:
        print("  Quick run: no GEMV matrix exceeds the cache, so H2 is not checked.")
    elif worst > 1.05:
        print(f"  WARNING: GEMV beat the bandwidth 'ceiling' by more than 5% at n = {n_max}.\n"
              "  B is understated (A2 loop or build flags), not an H2 pass. Fix before Gate 1.")


if __name__ == "__main__":
    main()
