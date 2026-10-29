#!/usr/bin/env python3
"""
matmul.py  --  Benchmark B: single-core FP32 GEMV and GEMM in NumPy.

Algorithm (spec Section 8):
  1. Force the math library (BLAS) to one thread: environment variables are
     set BEFORE numpy is imported, then threadpoolctl limits it again.
  2. Save np.show_config() and threadpool_info() to numpy_config.txt, which
     records the BLAS library and its thread count.
  3. For each kernel and size n: random FP32 matrices (fixed seed), one
     warm-up call, then 5 trials. One trial = repeat the call until >= 0.5 s.
  4. GFLOP/s = FLOPs per call * calls / seconds / 1e9.
       gemv_fp32  y = A x   2 n^2 FLOPs   reads 4 n^2 bytes   -> 0.5 FLOP/byte
       gemm_fp32  C = A B   2 n^3 FLOPs   >= 12 n^2 bytes     -> n/6 FLOP/byte

Known bias at small n: every call pays a fixed Python + NumPy dispatch cost
(about a microsecond). A 0.5 s trial does not remove it, because it is paid
per call. GEMV at n = 256 takes only a few microseconds, so small-n GEMV
understates the hardware. H2 uses only sizes larger than the last-level
cache, where one call takes milliseconds and the bias is negligible.

Output: long-format CSV on stdout, one row per trial:
  machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit
size_bytes = 4 n^2 (bytes of matrix A). Progress goes to stderr.

Normally run by run_all.py. By hand:
        python bench/matmul.py suraj-x86 data/raw/suraj-x86 > b_matmul.csv
        (environment variable QUICK=1: n <= 1024 and short trials, for testing)
"""
import os

# Must happen before numpy is imported: BLAS reads these when it loads.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
_overridden = {v: os.environ[v] for v in THREAD_VARS
               if os.environ.get(v, "1") != "1"}
for v in THREAD_VARS:
    os.environ[v] = "1"

import contextlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

QUICK = os.environ.get("QUICK") == "1"
RUNS = 5
MIN_TRIAL_NS = 100_000_000 if QUICK else 500_000_000
SEED = 341
SIZES = {
    "gemv_fp32": [256, 512, 1024] if QUICK else [256, 512, 1024, 2048, 4096, 8192],
    "gemm_fp32": [256, 512, 1024] if QUICK else [256, 512, 1024, 2048, 4096],
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def pin_to_one_core():
    """Keep this process on one core, like the C benchmarks (platform.h)."""
    try:
        if hasattr(os, "sched_setaffinity"):                      # Linux / WSL2
            os.sched_setaffinity(0, {0})
            return "pinned to CPU 0"
        if sys.platform == "win32":                               # native Windows
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            # Opt out of Windows power throttling (as in platform.h):
            # PROCESS_POWER_THROTTLING_STATE {Version=1, ControlMask=EXECUTION_SPEED, StateMask=0}
            pt = (ctypes.c_ulong * 3)(1, 1, 0)
            k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_ulong]
            k32.SetProcessInformation(k32.GetCurrentProcess(), 4,   # 4 = ProcessPowerThrottling
                                      ctypes.byref(pt), ctypes.sizeof(pt))
            ok = k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 1)
            return "pinned to logical CPU 0" if ok else "WARNING: pinning failed"
    except Exception as e:                                        # noqa: BLE001
        return f"WARNING: pinning failed ({e})"
    return "macOS: not pinned (pinning not possible)"


def save_config(path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"numpy {np.__version__}\n\n== np.show_config() ==\n{buf.getvalue()}\n")
        f.write("== threadpoolctl.threadpool_info() inside the 1-thread limit ==\n")
        for pool in threadpool_info():
            f.write(f"{pool}\n")
        if _overridden:
            f.write(f"\nNOTE: these were set to other values and forced to 1: {_overridden}\n")


def trial(call, flops):
    """Repeat call() until MIN_TRIAL_NS has passed; return GFLOP/s."""
    calls = 0
    t0 = time.perf_counter_ns()
    while True:
        call()
        calls += 1
        dt = time.perf_counter_ns() - t0
        if dt >= MIN_TRIAL_NS:
            return flops * calls / dt          # FLOP per ns == GFLOP/s


def main():
    if len(sys.argv) < 2 or not re.fullmatch(r"[a-z0-9]+-(x86|arm)", sys.argv[1]):
        log("machine_id must look like <label>-x86 or <label>-arm, lowercase, e.g. laptop2-arm\n"
            "usage: python bench/matmul.py <machine_id> [config_dir]")
        sys.exit(1)
    machine_id = sys.argv[1]
    config_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    log(f"B: {pin_to_one_core()}")
    rng = np.random.default_rng(SEED)

    with threadpool_limits(limits=1):
        save_config(os.path.join(config_dir, "numpy_config.txt"))
        busy = [p for p in threadpool_info() if p.get("num_threads", 1) != 1]
        if busy:
            log(f"WARNING: a thread pool is not at 1 thread: {busy}")
        if _overridden:
            log(f"NOTE: forced to 1 thread (were {_overridden})")

        print("machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit")
        for kernel, sizes in SIZES.items():
            for n in sizes:
                A = rng.standard_normal((n, n), dtype=np.float32)
                if kernel == "gemv_fp32":
                    x = rng.standard_normal(n, dtype=np.float32)
                    out = np.empty(n, dtype=np.float32)
                    call = lambda: np.matmul(A, x, out=out)       # noqa: E731
                    flops = 2 * n * n
                else:
                    Bm = rng.standard_normal((n, n), dtype=np.float32)
                    out = np.empty((n, n), dtype=np.float32)
                    call = lambda: np.matmul(A, Bm, out=out)      # noqa: E731
                    flops = 2 * n ** 3

                call()                                            # warm-up
                vals = [trial(call, flops) for _ in range(RUNS)]
                for r, v in enumerate(vals, start=1):
                    print(f"{machine_id},B,{kernel},{4 * n * n},{n},{r},1,{v:.3f},GFLOP_per_s")
                sys.stdout.flush()
                log(f"  {kernel} n={n:5d}  {sorted(vals)[RUNS // 2]:8.2f} GFLOP/s (median)")
                del A, out
                if kernel == "gemm_fp32":
                    del Bm


if __name__ == "__main__":
    main()
