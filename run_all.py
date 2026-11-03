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
A2_MIB = (256, 512, 1024)        # A2 MiB per array; bandwidth.c allocates 3 arrays of this size
A2_ORDER = A2_MIB + A2_MIB[::-1] # forward, then backward: 10 runs per size
A2_RUNS = 5                      # timed runs per kernel in one bandwidth call (RUNS in bandwidth.c)
A2_HEADER = "machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit"
RAM_FRACTION = 0.5               # skip a run that needs more than this share of free RAM


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


def run_bench(cmd, csv_path, log, env=None, relabel=None):
    """stdout -> CSV file; stderr (progress) -> console and log, through relabel() if given."""
    with open(csv_path, "w", newline="") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE, text=True,
                             env=env, bufsize=1, errors="replace")
        for line in p.stderr:
            line = line.rstrip()
            log(relabel(line) if relabel else line)
        p.wait()
    if p.returncode != 0:
        fail(f"{os.path.basename(cmd[0])} exited with code {p.returncode}")


def free_ram_bytes():
    """RAM a new allocation can get without paging, in bytes; None if unknown.
    Standard library only. Same function as in bench/matmul.py."""
    try:
        if IS_WIN:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemStatus()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys
        elif IS_MAC:                     # free + inactive + speculative pages
            out = capture(["vm_stat"])
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            pages = re.findall(r"^Pages (?:free|inactive|speculative):\s+(\d+)", out, re.M)
            if pages:
                return page * sum(int(p) for p in pages)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:                                             # noqa: BLE001
        pass
    return None


def enough_ram(need, what, log):
    """False (and log why) if `need` bytes is more than RAM_FRACTION of free RAM."""
    free = free_ram_bytes()
    if free is None:
        log(f"  {what}: free RAM unknown, running anyway")
        return True
    if need > RAM_FRACTION * free:
        log(f"  {what}: skipped: not enough RAM "
            f"(needs {need / 2**30:.2f} GiB, {free / 2**30:.1f} GiB free)")
        return False
    log(f"  {what}: needs {need / 2**30:.2f} GiB, {free / 2**30:.1f} GiB free")
    return True


def stability_note(line):
    """bandwidth.c's 'changed speed by more than 30%' WARNING, logged as a NOTE.
    Used for every A2 call except the one that defines B, so only that one can
    mark the run as not usable."""
    if line.startswith("WARNING:") and "changed speed by more than 30%" in line:
        return "NOTE:" + line[len("WARNING:"):]
    return line


def run_a2(bw, mid, csv_path, log):
    """bandwidth once per size in A2_ORDER; every row goes to one CSV with one header.
    The second call at a size is renumbered runs 6-10 (B uses runs 1-5 at 256 MiB)."""
    part = csv_path + ".part"
    done = {}                                   # MiB -> bandwidth calls already made
    with open(csv_path, "w") as out:
        out.write(A2_HEADER + "\n")
        for mib in A2_ORDER:
            offset = A2_RUNS * done.get(mib, 0)
            done[mib] = done.get(mib, 0) + 1
            if not enough_ram(3 * (mib << 20), f"A2 {mib} MiB per array", log):
                continue
            defines_b = mib == A2_MIB[0] and offset == 0    # first 256 MiB call: runs 1-5 = B
            run_bench([bw, mid, str(mib)], part, log,
                      relabel=None if defines_b else stability_note)
            with open(part) as f:
                lines = f.readlines()
            os.remove(part)
            if not lines or lines[0].strip() != A2_HEADER:
                fail(f"unexpected header from {os.path.basename(bw)}: {lines[:1]}")
            for line in lines[1:]:
                fields = line.rstrip("\n").split(",")
                fields[5] = str(int(fields[5]) + offset)        # column 5 = run
                out.write(",".join(fields) + "\n")
            out.flush()


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
        hw = capture(["system_profiler", "SPHardwareDataType"])
        private = re.compile(r"Serial Number|UUID|UDID")   # device identifiers; not for a public repo
        parts.append("\n".join(l for l in hw.splitlines() if not private.search(l)))
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
    settle = 2 if args.quick else 15
    log(f"\nSettling {settle} s before measuring...")
    time.sleep(settle)

    t0 = time.time()
    a1_max = "8" if args.quick else "512"
    log("\n-- A1 latency sweep (1-2 min). Don't touch the laptop.")
    run_bench([lat, mid, a1_max], os.path.join(out, "a1_latency.csv"), log)

    log(f"\n-- A2 bandwidth at {', '.join(map(str, A2_ORDER))} MiB per array (about 40-60 s)")
    run_a2(bw, mid, os.path.join(out, "a2_bandwidth.csv"), log)

    log("\n-- B matmul (about 40-60 s)")
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
    system_info(os.path.join(out, "system_info.txt"))
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
