#!/usr/bin/env python3
"""Summarize raw A1 latency CSVs and check the Gate 1 criteria.

Reads  data/raw/<machine_id>/a1_latency*.csv  (each file = one full sweep,
                                               one row per timed run)
Writes data/summary/a1_latency_summary.csv   (one row per machine and size)

Prints, per machine:
  1. candidate cache boundaries: sizes where median latency rises >= 30%
     over the previous size. Compare with the vendor sizes in machines.csv.
  2. repeatability (Gate 1): if the folder holds 2+ sweeps
     (a1_latency.csv, a1_latency_repeat.csv, ...), the largest disagreement
     between sweep medians at each size. Gate 1 needs <= 5% on the plateaus.
     Sizes right at a boundary can swing more; that alone is not a failure.

Standard library only. Usage: python3 analysis/summarize.py [raw_dir]
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path("data/summary/a1_latency_summary.csv")
JUMP = 1.30      # >= 30% rise between adjacent sizes counts as a step
AGREE = 0.05     # Gate 1: sweep medians must agree within 5%


def human(nbytes):
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes} B"


def load(path):
    runs = defaultdict(list)                     # size -> [ns per load]
    with path.open() as f:
        for row in csv.DictReader(f):
            runs[int(row["size_bytes"])].append(float(row["value"]))
    return runs


def main():
    machines = sorted(d for d in RAW.iterdir() if d.is_dir()) if RAW.is_dir() else []
    sweeps_by_machine = {d.name: [load(p) for p in sorted(d.glob("a1_latency*.csv"))]
                         for d in machines}
    sweeps_by_machine = {m: s for m, s in sweeps_by_machine.items() if s}
    if not sweeps_by_machine:
        sys.exit(f"no a1_latency*.csv files under {RAW}/")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine_id", "size_bytes", "size_kib", "median_ns",
                    "min_ns", "max_ns", "runs"])
        for machine, sweeps in sweeps_by_machine.items():
            first = sweeps[0]                    # the summary uses the first sweep
            sizes = sorted(first)
            meds = [statistics.median(first[s]) for s in sizes]
            for s, med in zip(sizes, meds):
                v = first[s]
                w.writerow([machine, s, f"{s / 1024:.1f}", f"{med:.3f}",
                            f"{min(v):.3f}", f"{max(v):.3f}", len(v)])

            print(f"\n== {machine}: {len(sweeps)} sweep(s), {len(sizes)} sizes ==")
            print("Candidate cache boundaries (median rises >= 30%):")
            for (s0, m0), (s1, m1) in zip(zip(sizes, meds), zip(sizes[1:], meds[1:])):
                if m1 >= JUMP * m0:
                    print(f"  {human(s0):>10} -> {human(s1):>10}: {m0:7.2f} -> {m1:7.2f} ns")

            if len(sweeps) < 2:
                print("Repeatability: needs a second sweep (a1_latency_repeat.csv).")
                continue
            print(f"Repeatability: sizes where sweep medians differ by > {AGREE:.0%}:")
            bad = 0
            for s in sizes:
                ms = [statistics.median(sw[s]) for sw in sweeps if s in sw]
                if len(ms) < 2:
                    continue
                diff = (max(ms) - min(ms)) / min(ms)
                if diff > AGREE:
                    bad += 1
                    print(f"  {human(s):>10}: {100 * diff:5.1f}%")
            if bad == 0:
                print("  none")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
