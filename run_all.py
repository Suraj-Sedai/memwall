#!/usr/bin/env python3
"""
run_all.py  --  build and run every benchmark for ONE laptop.
Works the same on Windows (cmd), macOS and Linux.

  Windows:      .venv\\Scripts\\python run_all.py suraj-x86 --quick
                .venv\\Scripts\\python run_all.py suraj-x86
  macOS/Linux:  .venv/bin/python run_all.py maya-arm --quick
                .venv/bin/python run_all.py maya-arm

Full run (5-8 min):  A1 latency sweep, A2 bandwidth, B matmul, a second A1
sweep (a1_latency_repeat.csv) for the repeatability check, then
analysis/summarize.py and analysis/check_h2.py.
--quick (about 1 min): A1 stops at 8 MiB, B uses small sizes, no repeat
sweep. Checks that everything works; it is NOT data.

Results: data/raw/<machine_id>/. Everything printed is also saved to
data/raw/<machine_id>/run_log.txt.
"""
import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_X86 = platform.machine().lower() in ("x86_64", "amd64")
EXE = ".exe" if IS_WIN else ""
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


class Log:
    """Print to the console and append to run_log.txt."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.warnings = 0

    def __call__(self, line=""):
        if line.lstrip().startswith("WARNING"):
            self.warnings += 1
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def fail(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_compiler():
    if os.environ.get("CC"):
        return os.environ["CC"]
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    if IS_WIN:      # WinLibs installed by winget but not on PATH yet
        roots = [os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "WinGet", "Packages")]
        for root in roots:
            for pattern in ("BrechtSanders.WinLibs*/mingw64/bin/gcc.exe",
                            "BrechtSanders.WinLibs*/*/mingw64/bin/gcc.exe"):
                hits = glob.glob(os.path.join(root, pattern))
                if hits:
                    return hits[0]
        fail("No C compiler found. Install one in cmd with:\n"
             "  winget install BrechtSanders.WinLibs.POSIX.UCRT\n"
             "then open a NEW cmd window and run this again.")
    fail("No C compiler found. macOS: xcode-select --install   "
         "Linux: sudo apt install build-essential")


def build(cc, src, out, extra, log):
    static = ["-static"] if IS_WIN else []     # Windows: no DLLs needed next to the .exe
    cmd = [cc, "-O2", "-Wall", *extra, *static, "-o", out, src]
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(cc) + os.pathsep + env.get("PATH", "")
    log("  " + " ".join(os.path.basename(c) if c == cc else c for c in cmd))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.stdout.strip() or r.stderr.strip():
        log((r.stdout + r.stderr).rstrip())
    if r.returncode != 0:
        fail(f"compiling {src} failed (see messages above)")


def run_bench(cmd, csv_path, log, env=None):
    """stdout -> CSV file; stderr (progress) -> console and log."""
    with open(csv_path, "w", newline="") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE, text=True,
                             env=env, bufsize=1, errors="replace")
        for line in p.stderr:
            log(line.rstrip())
        p.wait()
    if p.returncode != 0:
        fail(f"{os.path.basename(cmd[0])} exited with code {p.returncode}")


def run_report(cmd, log):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    for line in (r.stdout + r.stderr).rstrip().splitlines():
        log(line)


def capture(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, errors="replace")
        return r.stdout + r.stderr
    except Exception as e:                                        # noqa: BLE001
        return f"({' '.join(cmd)} failed: {e})\n"


def on_battery():
    """True = on battery, False = plugged in, None = unknown."""
    try:
        if IS_WIN:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                            ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            st = Status()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
                return {0: True, 1: False}.get(st.ACLineStatus)
        elif IS_MAC:
            out = capture(["pmset", "-g", "batt"])
            if "AC Power" in out:
                return False
            if "Battery Power" in out:
                return True
        else:
            vals = [open(p).read().strip() for p in glob.glob("/sys/class/power_supply/*/online")]
            if vals:
                return "1" not in vals
    except Exception:                                             # noqa: BLE001
        pass
    return None


def system_info(path):
    parts = [f"date_utc: {time.strftime('%Y-%m-%d %H:%M', time.gmtime())}",
             f"platform: {platform.platform()}",
             f"machine: {platform.machine()}  processor: {platform.processor()}",
             f"python: {sys.version.split()[0]}", ""]
    if IS_WIN:
        ps = ("Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,"
              "NumberOfLogicalProcessors,L2CacheSize,L3CacheSize,MaxClockSpeed; "
              "Get-CimInstance Win32_PhysicalMemory | Format-List Capacity,Speed,"
              "ConfiguredClockSpeed,SMBIOSMemoryType,Manufacturer,PartNumber; "
              "Get-CimInstance Win32_ComputerSystem | Format-List Manufacturer,Model")
        parts.append(capture(["powershell", "-NoProfile", "-Command", ps]))
    elif IS_MAC:
        out = capture(["sysctl", "-a"])
        keep = re.compile(r"^hw\.(memsize|cachelinesize|l1dcachesize|l2cachesize|l3cachesize|perflevel)"
                          r"|^machdep\.cpu\.brand_string")
        parts.append("\n".join(l for l in out.splitlines() if keep.match(l)))
        parts.append(capture(["system_profiler", "SPHardwareDataType"]))
    else:
        parts.append(capture(["lscpu"]))
        parts.append(capture(["lscpu", "-C"]))
        try:
            if "microsoft" in open("/proc/version").read().lower():
                parts.append("environment: WSL2")
        except OSError:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run all memory-wall benchmarks on this laptop.")
    ap.add_argument("machine_id", help="<name>-<isa>, lowercase, e.g. suraj-x86 or maya-arm")
    ap.add_argument("--quick", action="store_true", help="1-minute functional test; not data")
    args = ap.parse_args()

    mid = args.machine_id
    if not re.fullmatch(r"[a-z0-9]+-(x86|arm)", mid):
        fail("machine_id must look like <name>-x86 or <name>-arm, lowercase, e.g. suraj-x86")
    if (mid.endswith("-x86")) != IS_X86:
        hint = ("\nOn an Apple Silicon Mac this means Python runs under Rosetta: "
                "install the arm64 Python instead.") if IS_MAC else ""
        fail(f"machine_id says {mid.split('-')[-1]} but this CPU is {platform.machine()}{hint}")
    try:
        import numpy  # noqa: F401
        import threadpoolctl  # noqa: F401
    except ImportError:
        fail("numpy/threadpoolctl missing. Run this with the project's Python:\n"
             "  Windows:      .venv\\Scripts\\python run_all.py " + mid + "\n"
             "  macOS/Linux:  .venv/bin/python run_all.py " + mid + "\n"
             "(one-time setup is in README.md)")

    os.chdir(ROOT)
    out = os.path.join("data", "raw", mid)
    os.makedirs(out, exist_ok=True)
    log = Log(os.path.join(out, "run_log.txt"))
    log(f"== {mid}  {'QUICK TEST (not data)' if args.quick else 'FULL RUN'}  "
        f"{time.strftime('%Y-%m-%d %H:%M')} ==")

    batt = on_battery()
    log("Power: " + {True: "ON BATTERY", False: "plugged in", None: "unknown"}[batt])
    if batt and not args.quick:
        fail("The laptop is on battery. Plug it in, set Power mode to Best performance, "
             "close other apps, and run again.")
    log("Don't switch windows or use the laptop until it says Done.")

    cc = find_compiler()
    log(f"Compiler: {cc}")
    lat = os.path.join("bench", "latency" + EXE)
    bw = os.path.join("bench", "bandwidth" + EXE)
    build(cc, os.path.join("bench", "latency.c"), lat, [], log)
    # -fno-builtin keeps copy as our loop, not libc memcpy.
    # -march=native (x86 only) gives the read kernels the wide loads BLAS uses (see bandwidth.c).
    build(cc, os.path.join("bench", "bandwidth.c"), bw,
          ["-fno-builtin"] + (["-march=native"] if IS_X86 else []), log)
    system_info(os.path.join(out, "system_info.txt"))

    t0 = time.time()
    a1_max = "8" if args.quick else "512"
    log("\n-- A1 latency sweep (1-2 min). Don't touch the laptop.")
    run_bench([lat, mid, a1_max], os.path.join(out, "a1_latency.csv"), log)

    log("\n-- A2 bandwidth (about 15 s)")
    run_bench([bw, mid], os.path.join(out, "a2_bandwidth.csv"), log)

    log("\n-- B matmul (about 1 min)")
    env = dict(os.environ, QUICK="1" if args.quick else "0", **{v: "1" for v in THREAD_VARS})
    run_bench([sys.executable, os.path.join("bench", "matmul.py"), mid, out],
              os.path.join(out, "b_matmul.csv"), log, env=env)

    if not args.quick:
        log("\n-- A1 repeat sweep for Gate 1 repeatability (1-2 min)")
        run_bench([lat, mid, a1_max], os.path.join(out, "a1_latency_repeat.csv"), log)
    else:
        rep = os.path.join(out, "a1_latency_repeat.csv")
        if os.path.exists(rep):
            os.remove(rep)                 # never mix an old repeat with a quick run

    log(f"\n-- Benchmarks finished in {time.time() - t0:.0f} s\n")
    log("== A1: cache steps and repeatability ==")
    run_report([sys.executable, os.path.join("analysis", "summarize.py")], log)
    log("\n== A2 + B: roofline ceilings and H2 ==")
    run_report([sys.executable, os.path.join("analysis", "check_h2.py"), out], log)
    log(f"\nDone. Results and run_log.txt are in {out}")
    if log.warnings:
        log(f"RESULT: {log.warnings} WARNING line(s) above. This run is NOT usable as data: "
            "fix what they say and run again.")
    else:
        log("RESULT: no warnings.")
    if args.quick:
        log("This was a QUICK TEST. Run again without --quick for real data.")


if __name__ == "__main__":
    main()
