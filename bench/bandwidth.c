/*
 * bandwidth.c  --  Benchmark A2: single-core streaming bandwidth (STREAM-style).
 *
 * Algorithm:
 *   1. Allocate three arrays of doubles, 256 MiB each (>= 4x any laptop LLC),
 *      and write every element so the OS maps every page before timing.
 *   2. For each kernel:
 *        a. Warm up: run the kernel for >= 0.3 s (CPU clock, kernel switch).
 *        b. Calibrate: double the number of full passes until one timed run
 *           lasts >= 100 ms.
 *        c. 5 timed runs with that pass count. One CSV row per run.
 *   3. GB/s = bytes counted per element * elements * passes / seconds / 1e9.
 *      Bytes counted (STREAM convention, excludes write-allocate reads):
 *        read_sum8   s += a[i], 8 independent sums     8  (read)
 *        read_sum16  s += a[i], 16 independent sums    8  (read)
 *        copy        a[i] = b[i]                      16  (8 read + 8 write)
 *        triad       a[i] = b[i] + q*c[i]             24  (16 read + 8 write)
 *
 * B (the roofline's memory ceiling) = the FASTER of read_sum8 and read_sum16
 * (median of 5 runs each). Why two read kernels:
 *   - One running sum measures the adder, not memory: each add waits for the
 *     previous one (FP add latency). Independent partial sums break the chain.
 *     How many are needed depends on the core: more FP pipes and higher
 *     bandwidth (e.g. Apple M-series) need more sums in flight.
 *   - But on the first test laptop (Intel, Core Ultra class) read_sum16 starts
 *     at ~19 GB/s and falls to ~9-11 GB/s within 0.5 s, every time it starts,
 *     while read_sum8 holds ~23 GB/s (bench/bw_trace.c shows this). The loops
 *     are equivalent apart from the number of sums, so this is a hardware
 *     behaviour (most likely an adaptive prefetcher); its cause is not
 *     documented. One fixed kernel cannot be trusted on every core, so both
 *     run and the faster defines B. Both are in the CSV and in the report.
 *   Experiment: build with -DONE_ACC and read_sum16 becomes read_sum1 (one sum).
 *
 * Stability: a kernel's 5 runs should agree within 10% and match its last
 * calibration run within 15%; otherwise a NOTE records the range. For the
 * kernel that defines B, a change of more than 30% is a WARNING (the run is
 * not data). The 30% limit comes from a noise test on the first laptop:
 * 6 standalone runs on two P-cores all varied 5-26%, a property of the
 * memory system, while real disturbances earlier caused ~2x drops.
 *
 * Copy and triad: built with -fno-builtin, otherwise gcc/clang replace the
 * copy loop with a call to libc memcpy, which differs by OS (glibc uses
 * non-temporal stores at this size) and would not be the loop we describe.
 *
 * Controls (same as A1): >= 100 ms per timed run; 5 runs after warm-up; the
 * results feed a volatile sink so the compiler cannot delete the loops;
 * kernels are noinline with a memory barrier between passes so repeated
 * passes cannot be merged or skipped.
 *
 * Output: long-format CSV on stdout, one row per timed run:
 *   machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit
 * size_bytes = bytes per array; n blank; unit GB_per_s. Progress on stderr.
 *
 * Vector width (x86 only): build with -march=native so the loops use the
 * same wide loads (AVX2/AVX-512) that BLAS GEMV uses. Without it gcc emits
 * 16-byte SSE2 loads, and on the test machine the read rate came out ~30%
 * below what OpenBLAS GEMV itself pulled from memory: a "ceiling" GEMV can
 * beat. Apple Silicon needs no flag (NEON is always on).
 *
 * Build:  normally done by run_all.py. By hand:
 *         cc -O2 -Wall -fno-builtin -march=native -o bench/bandwidth bench/bandwidth.c   (x86)
 *         cc -O2 -Wall -fno-builtin -o bench/bandwidth bench/bandwidth.c                 (arm)
 * Run:    bench/bandwidth suraj-x86 > a2_bandwidth.csv
 *         bench/bandwidth suraj-x86 512    (512 MiB per array: only if LLC > 64 MiB)
 */
#include "platform.h"            /* first: timer, allocation, pinning per OS */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_MIB_DEF 256          /* MiB per array */
#define RUNS          5            /* timed runs per kernel */
#define MIN_SECONDS   0.100        /* minimum duration of one timed run */
#define WARMUP_SEC    0.300        /* untimed warm-up per kernel */
#define ALIGN         4096         /* page-aligned arrays */

#define BARRIER() __asm__ volatile("" ::: "memory")

static volatile double sink;       /* consumer of every result */

/* ---- kernels: one full pass each ------------------------------------- */

__attribute__((noinline))
static double k_read_sum8(const double *restrict a, size_t n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (size_t i = 0; i < n; i += 8) {     /* n is a multiple of 16 */
        s0 += a[i];     s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
        s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
    }
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

__attribute__((noinline))
static double k_read_sum16(const double *restrict a, size_t n) {
#if defined(ONE_ACC)
    double s = 0.0;                          /* one dependency chain: add-latency bound */
    for (size_t i = 0; i < n; i++) s += a[i];
    return s;
#else
    /* Named variables: an array of sums would live on the stack, and every
     * add would then wait on a store and a reload. */
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    for (size_t i = 0; i < n; i += 16) {
        s0 += a[i];      s1 += a[i + 1];  s2 += a[i + 2];  s3 += a[i + 3];
        s4 += a[i + 4];  s5 += a[i + 5];  s6 += a[i + 6];  s7 += a[i + 7];
        t0 += a[i + 8];  t1 += a[i + 9];  t2 += a[i + 10]; t3 += a[i + 11];
        t4 += a[i + 12]; t5 += a[i + 13]; t6 += a[i + 14]; t7 += a[i + 15];
    }
    return (((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))) +
           (((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)));
#endif
}

__attribute__((noinline))
static void k_copy(double *restrict a, const double *restrict b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = b[i];
}

__attribute__((noinline))
static void k_triad(double *restrict a, const double *restrict b,
                    const double *restrict c, double q, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = b[i] + q * c[i];
}

/* ---- kernel table ------------------------------------------------------ */

enum { READ8, READ16, COPY, TRIAD, NK };
static const char *const KNAME[NK] = {
#if defined(ONE_ACC)
    "read_sum8", "read_sum1", "copy", "triad"
#else
    "read_sum8", "read_sum16", "copy", "triad"
#endif
};
static const double KBYTES[NK] = { 8.0, 8.0, 16.0, 24.0 };  /* counted per element */

static double *A, *B, *C;
static size_t N;

static double timed(int k, uint64_t passes) {
    double acc = 0.0;
    double t0 = now();
    for (uint64_t p = 0; p < passes; p++) {
        switch (k) {
        case READ8:  acc += k_read_sum8(A, N);    break;
        case READ16: acc += k_read_sum16(A, N);   break;
        case COPY:   k_copy(A, B, N);             break;
        case TRIAD:  k_triad(A, B, C, 3.0, N);    break;
        }
        BARRIER();
    }
    double dt = now() - t0;
    sink = acc + A[N / 2];                   /* results are observable */
    return dt;
}

static double gbs_of(int k, uint64_t passes, double dt) {
    return KBYTES[k] * (double)N * (double)passes / dt / 1e9;
}

static int valid_id(const char *s) {          /* <label>-x86 or <label>-arm */
    size_t len = strlen(s);
    if (len < 5) return 0;
    const char *isa = s + len - 4;
    if (strcmp(isa, "-x86") != 0 && strcmp(isa, "-arm") != 0) return 0;
    for (const char *p = s; p < isa; p++)
        if (!(islower((unsigned char)*p) || isdigit((unsigned char)*p))) return 0;
    return 1;
}

static double median5(const double v[RUNS]) {
    double t[RUNS];
    memcpy(t, v, sizeof t);
    for (int i = 1; i < RUNS; i++)
        for (int j = i; j > 0 && t[j - 1] > t[j]; j--) {
            double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x;
        }
    return t[RUNS / 2];
}

/* 1 if the 5 runs disagree by > 10% or drift > 15% from the calibration run. */
static int unstable(double warm, const double v[RUNS], double med, double *lo, double *hi) {
    *lo = *hi = v[0];
    for (int i = 1; i < RUNS; i++) { if (v[i] < *lo) *lo = v[i]; if (v[i] > *hi) *hi = v[i]; }
    double drift = med > warm ? med / warm - 1.0 : warm / med - 1.0;
    return *hi / *lo - 1.0 > 0.10 || drift > 0.15;
}

int main(int argc, char **argv) {
    if (argc < 2 || !valid_id(argv[1])) {
        fprintf(stderr, "machine_id must look like <label>-x86 or <label>-arm, "
                        "lowercase, e.g. laptop2-arm\n"
                        "usage: %s <machine_id> [MiB_per_array]\n", argv[0]);
        return 1;
    }
    const char *machine_id = argv[1];
    long mib = argc > 2 ? strtol(argv[2], NULL, 10) : ARRAY_MIB_DEF;
    if (mib < 1 || mib > 4096) { fprintf(stderr, "bad MiB_per_array\n"); return 1; }

    fprintf(stderr, "A2: %s\n", pin_to_one_core());

    size_t bytes = (size_t)mib << 20;
    N = bytes / sizeof(double);              /* multiple of 16 for any whole MiB */
    void *pa = alloc_aligned(ALIGN, bytes), *pb = alloc_aligned(ALIGN, bytes),
         *pc = alloc_aligned(ALIGN, bytes);
    if (!pa || !pb || !pc) {
        fprintf(stderr, "could not allocate 3 x %ld MiB\n", mib);
        return 1;
    }
    A = pa; B = pb; C = pc;
    for (size_t i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; C[i] = 0.5; }

    printf("machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit\n");
    fprintf(stderr, "A2: 3 arrays x %ld MiB\n", mib);

    double warm[NK], runs[NK][RUNS], med[NK];
    for (int k = 0; k < NK; k++) {
        double t_end = now() + WARMUP_SEC;   /* untimed warm-up */
        while (now() < t_end) timed(k, 1);

        uint64_t passes = 1;                 /* calibration */
        double dt;
        while ((dt = timed(k, passes)) < MIN_SECONDS) passes *= 2;
        warm[k] = gbs_of(k, passes, dt);

        for (int r = 1; r <= RUNS; r++) {
            runs[k][r - 1] = gbs_of(k, passes, timed(k, passes));
            printf("%s,A2,%s,%llu,,%d,1,%.3f,GB_per_s\n",
                   machine_id, KNAME[k], (unsigned long long)bytes, r, runs[k][r - 1]);
        }
        fflush(stdout);
        med[k] = median5(runs[k]);
        fprintf(stderr, "  %-10s %7.2f GB/s (median of %d, %llu passes/run)\n",
                KNAME[k], med[k], RUNS, (unsigned long long)passes);
    }

    int bk = med[READ16] > med[READ8] ? READ16 : READ8;
    int ok = bk == READ16 ? READ8 : READ16;
    fprintf(stderr, "  B = %.2f GB/s from %s (the faster read kernel)\n", med[bk], KNAME[bk]);
    if (med[bk] / med[ok] - 1.0 > 0.10)
        fprintf(stderr, "NOTE: %s is %.0f%% slower than %s. Expected on some CPUs; "
                        "B uses the faster one. Keep both in the report.\n",
                KNAME[ok], 100.0 * (1.0 - med[ok] / med[bk]), KNAME[bk]);

    for (int k = 0; k < NK; k++) {
        double lo, hi;
        if (!unstable(warm[k], runs[k], med[k], &lo, &hi)) continue;
        double drift = med[k] > warm[k] ? med[k] / warm[k] - 1.0 : warm[k] / med[k] - 1.0;
        if (k == bk && (hi / lo - 1.0 > 0.30 || drift > 0.30))
            fprintf(stderr, "WARNING: %s (defines B) changed speed by more than 30%% during the run "
                            "(calibration %.2f GB/s, runs %.2f-%.2f GB/s). The laptop was busy "
                            "or changed power state: plug in, Best performance, close "
                            "everything, don't switch windows, re-run.\n",
                    KNAME[k], warm[k], lo, hi);
        else
            fprintf(stderr, "NOTE: %s varied during the run (calibration %.2f GB/s, "
                            "runs %.2f-%.2f GB/s). %s\n",
                    KNAME[k], warm[k], lo, hi,
                    k == bk ? "It defines B: report B with this range."
                            : "Not used for B; keep it for the report.");
    }
    fprintf(stderr, "Plausibility: B must be below the RAM's theoretical peak "
                    "(channels x 8 B x MT/s) recorded in machines.csv.\n");

    free_aligned(pa); free_aligned(pb); free_aligned(pc);
    return 0;
}
