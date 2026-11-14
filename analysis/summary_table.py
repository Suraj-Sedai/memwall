#!/usr/bin/env python3
"""One summary row per machine folder in data/raw/, for readers who don't use the code.

Reads  data/raw/<machine_id>/*.csv and data/machines.csv (never data/archive/).
Writes data/summary/summary_table.csv and data/summary/summary_table.xlsx.

The H1/H2/H3 verdicts and the B / P_peak definitions are copied from
analysis/explore.ipynb (thresholds at lines 36-41, ceilings() at 175-206,
h1/h2/h3 at 604-658). Keep the two in step: if a rule changes there, change it here.

Usage (repo root): .venv-analysis\\Scripts\\python analysis\\summary_table.py
Needs pandas and openpyxl.
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"                      # data/archive is a sibling, never read
OUT_CSV = ROOT / "data" / "summary" / "summary_table.csv"
OUT_XLSX = ROOT / "data" / "summary" / "summary_table.xlsx"

# Fixed verdict criteria, copied from explore.ipynb lines 37-41. Do not tune these to the data.
JUMP = 1.30                  # H1: a step is a median rise >= 30% between adjacent sizes
H1_RANGE = (0.5, 2.0)        # H1: detected step / nearest vendor size
H2_RANGE = (0.75, 1.25)      # H2: GEMV / (0.5 x B), matrices larger than the LLC
H3_MIN = 10.0                # H3: best GEMM / GEMV at n = H3_N
H3_N = 8192                  # H3: fixed GEMV size

# B definition, explore.ipynb lines 175-176.
B_SIZE = 256 * 2**20         # A2 bytes per array that define B
B_RUNS = 5                   # B uses runs 1-5 (the first, forward pass)
BIG_SIZE = 1024 * 2**20      # extra column: B at 1024 MiB per array, all runs (info only)

TYPE = {"marco-x86": "desktop"}  # every other machine is a laptop


# ---- load (explore.ipynb lines 130-157 and 171-173) ----
frames = []
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    for f in sorted(d.glob("*.csv")):
        df = pd.read_csv(f)
        df["machine"] = d.name
        df["file"] = f.stem
        frames.append(df)
data = pd.concat(frames, ignore_index=True)

specs = pd.read_csv(ROOT / "data" / "machines.csv").set_index("machine_id")
for col in ("l1d_kib_per_core", "l2_kib", "llc_mib"):
    specs[col] = pd.to_numeric(specs[col], errors="coerce")

MACHINES = sorted(data["machine"].unique())

med = (data.groupby(["machine", "file", "benchmark", "kernel", "size_bytes", "n"], dropna=False)["value"]
           .agg(["median", "min", "max", "count"]).reset_index())
a1 = med[med["benchmark"] == "A1"]


def vendor_sizes(m):
    # explore.ipynb lines 150-157: vendor cache sizes in bytes; {} if no machines.csv row.
    if m not in specs.index:
        return {}
    r = specs.loc[m]
    sizes = {"L1D": r["l1d_kib_per_core"] * 1024, "L2": r["l2_kib"] * 1024,
             "LLC": r["llc_mib"] * 2**20}
    return {k: v for k, v in sizes.items() if pd.notna(v)}


def bandwidth(m, size):
    # explore.ipynb lines 187-195: faster read_sum8/read_sum16 median over ALL runs at one size.
    a2 = med[(med["machine"] == m) & med["kernel"].isin(["read_sum8", "read_sum16"])
             & (med["size_bytes"] == size)]
    if a2.empty:
        return float("nan")
    return a2["median"].max()


def ceilings(m):
    # explore.ipynb lines 197-206: B = faster read median of runs 1-5 at 256 MiB;
    # P_peak = best GEMM median.
    first = data[(data["machine"] == m) & data["kernel"].isin(["read_sum8", "read_sum16"])
                 & (data["size_bytes"] == B_SIZE) & (data["run"] <= B_RUNS)]
    roof = first.groupby("kernel")["value"].median()
    B = roof.max() if not roof.empty else float("nan")
    gemm = med[(med["machine"] == m) & (med["kernel"] == "gemm_fp32")]
    P = gemm["median"].max() if not gemm.empty else float("nan")
    return B, P


def first_sweep(m):
    # explore.ipynb line 609: the first A1 sweep (file names sorted), by size.
    sweeps = a1[a1["machine"] == m]
    if sweeps.empty:
        return sweeps
    return sweeps[sweeps["file"] == sorted(sweeps["file"].unique())[0]].sort_values("size_bytes")


def steps(m):
    # explore.ipynb lines 610-611: a step is the last size before a >= 30% rise.
    first = first_sweep(m)
    sizes, meds = first["size_bytes"].tolist(), first["median"].tolist()
    return [s0 for s0, m0, m1 in zip(sizes, meds, meds[1:]) if m1 >= JUMP * m0]


def h1(m):
    # explore.ipynb lines 604-620.
    vs = vendor_sizes(m)
    if not vs or first_sweep(m).empty:
        return "n/a"
    st = steps(m)
    if not st:
        return "FAIL"
    ok = True
    for s in st:
        name, v = min(vs.items(), key=lambda kv: abs(math.log(s / kv[1])))
        ok &= H1_RANGE[0] <= s / v <= H1_RANGE[1]
    return "PASS" if ok else "FAIL"


def h2_rows(m):
    # explore.ipynb lines 624-635: GEMV rows whose matrix is larger than the LLC
    # (n >= 8192 if no llc_mib), and GEMV / (0.5 x B) for each. None if n/a.
    B, _ = ceilings(m)
    vs = vendor_sizes(m)
    gemv = med[(med["machine"] == m) & (med["kernel"] == "gemv_fp32")]
    if pd.isna(B) or gemv.empty:
        return None
    if "LLC" in vs:
        big = gemv[gemv["size_bytes"] > vs["LLC"]].sort_values("n")
    else:
        big = gemv[gemv["n"] >= 8192].sort_values("n")
    if big.empty:
        return None
    return list(zip(big["n"].astype(int), big["median"] / (0.5 * B)))


def h2(m):
    # explore.ipynb lines 636 and 645.
    rows = h2_rows(m)
    if rows is None:
        return "n/a"
    ok = all(H2_RANGE[0] <= r <= H2_RANGE[1] for _, r in rows)
    return "PASS" if ok else "FAIL"


def gemm_over_gemv(m):
    # explore.ipynb lines 649-654: best GEMM median / GEMV median at n = 8192.
    _, P = ceilings(m)
    gemv = med[(med["machine"] == m) & (med["kernel"] == "gemv_fp32")].set_index("n")["median"]
    if pd.isna(P) or H3_N not in gemv.index:
        return float("nan")
    return P / gemv[H3_N]


def h3(m):
    # explore.ipynb lines 651-658.
    ratio = gemm_over_gemv(m)
    if pd.isna(ratio):
        return "n/a"
    return "PASS" if ratio >= H3_MIN else "FAIL"


def human(nbytes):
    if nbytes >= 2**20:
        return f"{nbytes / 2**20:.3g} MiB"
    return f"{nbytes / 1024:.3g} KiB"


def spec(m, col):
    if m not in specs.index:
        return None
    v = specs.loc[m, col]
    return None if pd.isna(v) else v


rows = []
for m in MACHINES:
    B, P = ceilings(m)
    first = first_sweep(m)
    h2r = h2_rows(m)
    rows.append({
        "machine": m,
        "type": TYPE.get(m, "laptop"),
        "cpu": spec(m, "cpu"),
        "l1d_kib": spec(m, "l1d_kib_per_core"),
        "l2_kib": spec(m, "l2_kib"),
        "llc_mib": spec(m, "llc_mib"),
        "measured_cache_steps": ", ".join(human(s) for s in steps(m)),
        "dram_latency_ns": round(first["median"].iloc[-1], 1) if not first.empty else None,
        "B_GBps": round(B, 2),
        "B_1024MiB_GBps": round(bandwidth(m, BIG_SIZE), 2),
        "peak_gemm_GFLOPs": round(P, 1),
        "ridge_point": round(P / B, 1),
        "gemv_over_predicted": "; ".join(f"n={n}: {r:.2f}" for n, r in h2r) if h2r else "",
        "gemm_over_gemv_n8192": round(gemm_over_gemv(m), 1),
        "H1": h1(m),
        "H2": h2(m),
        "H3": h3(m),
    })
table = pd.DataFrame(rows)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT_CSV, index=False)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    table.to_excel(xw, sheet_name="summary", index=False)
    ws = xw.sheets["summary"]
    for i, col in enumerate(table.columns, start=1):
        width = max(len(col), *(len(str(v)) for v in table[col] if v is not None)) + 2
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, 60)
    ws.freeze_panes = "B2"

with pd.option_context("display.max_columns", None, "display.width", 250,
                       "display.max_colwidth", None):
    print(table.to_string(index=False))
print(f"\nWrote {OUT_CSV.relative_to(ROOT)} and {OUT_XLSX.relative_to(ROOT)}")
