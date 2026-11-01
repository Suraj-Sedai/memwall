# The Memory Wall on Commodity Laptops

CISC 341 (Fall 2026) term project. Measures one CPU core's memory latency, memory
bandwidth and matrix-multiply speed on each team member's laptop, and builds a
roofline model from the results.

| Benchmark | What it measures |
|---|---|
| A1 `bench/latency.c` | Time per memory load, 4 KiB to 512 MiB (shows the cache levels) |
| A2 `bench/bandwidth.c` | Streaming bandwidth from RAM, GB/s |
| B `bench/matmul.py` | Matrix-vector and matrix-matrix speed, GFLOP/s |

All data is collected with code version `data-v3`. Do not edit any file.

## 1. Install (once)

**Windows**, in Command Prompt (cmd, not PowerShell):

```
winget install BrechtSanders.WinLibs.POSIX.UCRT
winget install Python.Python.3.12
```

Close cmd and open a new one. If `winget` is not recognized:
- Python: python.org → Downloads → Windows → "Windows installer (64-bit)".
  Tick **Add python.exe to PATH**.
- GCC: winlibs.com → UCRT runtime, POSIX threads, Win64, without LLVM, Zip.
  Extract to `C:\`, then add `C:\mingw64\bin` to your user Path
  (Start → "Edit environment variables for your account" → Path → New).

**macOS (Apple Silicon)**

```
xcode-select --install
```

**Linux**

```
sudo apt install build-essential python3-venv
```

## 2. Get the code (once)

On GitHub: **Code → Download ZIP**. Unzip it to a plain folder **outside
OneDrive or iCloud** (for example `C:\memwall` or `~/memwall`) and open a
terminal in that folder.

**Windows (cmd)**

```
py -m venv .venv
.venv\Scripts\python -m pip install numpy threadpoolctl
```

**macOS / Linux**

```
python3 -m venv .venv
.venv/bin/python -m pip install numpy threadpoolctl
```

## 3. Choose your machine id

Your first name in lowercase, then `-x86` for Intel/AMD or `-arm` for Apple
Silicon. Example: `maya-arm`. Use the same id every time. In the commands
below, replace `<id>` with it.

## 4. Before every run

- Charger plugged in.
- Windows: Power mode **Best performance**. macOS: **Low Power Mode off**.
- Wi-Fi off (Airplane mode). Every other app closed.
- Don't touch the laptop until it prints `Done`.

## 5. Run

Quick test first (about 1 minute; checks that everything works, not data):

**Windows (cmd)**

```
.venv\Scripts\python run_all.py <id> --quick
```

**macOS / Linux**

```
.venv/bin/python run_all.py <id> --quick
```

If it ends with `RESULT: no warnings.`, do the full run (about 5 minutes):

**Windows (cmd)**

```
.venv\Scripts\python run_all.py <id>
```

**macOS / Linux**

```
.venv/bin/python run_all.py <id>
```

Always type the `.venv` path as shown. Plain `python` uses the wrong Python.

## 6. Check the result

- `RESULT: no warnings.` means this is your data. `NOTE` lines are fine.
- Any `WARNING` means the run is not data. Move it aside and run again:

**Windows (cmd)**

```
mkdir data\archive
move data\raw\<id> data\archive\<id>-rejected1
```

**macOS / Linux**

```
mkdir -p data/archive
mv data/raw/<id> data/archive/<id>-rejected1
```

Number them `rejected1`, `rejected2`, and so on ("already exists" from mkdir is
fine). After 3 rejected runs, stop and contact Suraj.

## 7. Send your data

Zip the folder `data/raw/<id>` (and `data/archive`, if you have one) and send it
to Suraj. Don't edit the files and don't push to GitHub.

## Troubleshooting

| Message | Fix |
|---|---|
| `'gcc' is not recognized` or `No C compiler found` | Open a new terminal after installing. Check with `gcc --version`. |
| `numpy/threadpoolctl missing` | Run with the `.venv` path, not plain `python`. |
| `machine_id says arm but this CPU is x86_64` (Mac) | Python is running under Rosetta. Install the macOS installer from python.org and redo step 2. |
| `The laptop is on battery` | Plug in the charger. |
| Windows laptop with a Snapdragon (ARM) chip | Contact Suraj before running. |

## Known limitations

- macOS cannot pin a program to one core. Windows and Linux pin to logical CPU 0.
- Beyond a few MiB, A1 latency includes TLB misses, not only cache or DRAM time.
- A2 counts bytes the STREAM way: copy and triad exclude the write-allocate read.
- B (the bandwidth ceiling) is the faster of two read kernels, because on the
  first Intel laptop the 16-sum kernel ran at half speed.
- On Apple Silicon, NumPy runs GEMM on Apple's matrix unit, so the compute
  ceiling there is not one core's vector units.