# The Memory Wall on Commodity Laptops (CISC 341, Fall 2026)

Measures each team member's laptop memory hierarchy (latency, bandwidth) and matrix
multiply speed, and builds a roofline from those measurements.

| Benchmark | File | Measures |
|---|---|---|
| A1 load latency (pointer chasing) | `bench/latency.c` | ns per load, 4 KiB to 512 MiB |
| A2 streaming bandwidth | `bench/bandwidth.c` | GB/s; the faster of `read_sum8` / `read_sum16` is the roofline's B |
| B GEMV / GEMM (NumPy, 1 thread) | `bench/matmul.py` | GFLOP/s; best GEMM is P_peak |

`run_all.py` builds and runs everything on Windows, macOS and Linux. All OS-specific
C code is in `bench/platform.h`.

## One-time setup

Keep the repo in a plain folder such as `C:\memwall`, **not** inside OneDrive
(Desktop/Documents are often synced, and syncing during a run disturbs the timings).

**Windows (cmd, no Ubuntu/WSL needed)**

```
winget install BrechtSanders.WinLibs.POSIX.UCRT
winget install Python.Python.3.12
```

Close cmd, open a new one, then in the repo folder:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install numpy threadpoolctl
```

**macOS:** `xcode-select --install`, then `python3 -m venv .venv && .venv/bin/pip install numpy threadpoolctl`

**Linux:** `sudo apt install build-essential python3-venv`, then the same two Python commands.

## Before every real run

1. Plugged in. Power mode: Best performance (Windows) / Low Power Mode off (macOS).
2. Close the browser, Discord, IDEs, OneDrive, everything. Let the laptop cool down.
3. Don't touch the laptop or switch windows until it says Done.

A full run refuses to start on battery. A2 prints a WARNING if the laptop changed
speed while measuring B, and the run ends with `RESULT:`. Any WARNING means that run
is not data. NOTE lines are expected findings to keep for the report, not errors.

## Run

```
.venv\Scripts\python run_all.py <name>-<isa> --quick     (Windows; about 1 min; NOT data)
.venv\Scripts\python run_all.py <name>-<isa>             (Windows; full run, 5-8 min)
.venv/bin/python run_all.py <name>-<isa>                 (macOS / Linux)
```

`<name>-<isa>` is lowercase, e.g. `suraj-x86` or `maya-arm`.

Results go to `data/raw/<machine_id>/`: `a1_latency.csv`, `a1_latency_repeat.csv`,
`a2_bandwidth.csv`, `b_matmul.csv`, `numpy_config.txt`, `system_info.txt` and
`run_log.txt` (everything the run printed). Then add one row for your laptop to
`data/machines.csv`, with cache sizes from the **vendor's product page** and its URL.
Never edit files in `data/raw/`.

## Known limitations (go in the report)

- macOS cannot pin a thread to a core; the benchmarks request a performance core, but
  the OS may still move them. Windows and Linux pin to logical CPU 0.
- Beyond a few MiB, A1 latency includes TLB misses, not only DRAM time.
- A2 counts bytes the STREAM way: copy and triad exclude the write-allocate read.
- On at least one Intel laptop, the 16-sum read kernel slows to under half speed within
  0.5 s while the 8-sum kernel does not (`bench/bw_trace.c` shows it over time). B is
  therefore the faster of the two read kernels; both are reported.
- Small-n GEMV is understated by per-call Python overhead; only matrices larger than
  the last-level cache test H2.
