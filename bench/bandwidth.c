/*
 * bandwidth.c  --  Benchmark A2: single-core streaming bandwidth (STREAM-style).
 *
 * Algorithm:
 *   1. Allocate three arrays of doubles, 256 MiB each (>= 4x any laptop LLC),
 *      and write every element so the OS maps every page before timing.
 *   2. For each kernel (read_sum, copy, triad):
 *        a. Calibrate: double the number of full passes over the arrays until
 *           one timed run lasts >= 100 ms. Calibration doubles as warm-up.
 *        b. 5 timed runs with that pass count. One CSV row per run.
 *   3. GB/s = bytes counted per element * elements * passes / seconds / 1e9.
 *      Bytes counted (STREAM convention, excludes write-allocate reads):
 *        read_sum  s += a[i]              8  (read; 16 independent sums)
 *        copy      a[i] = b[i]           16  (8 read + 8 write)
 *        triad     a[i] = b[i] + q*c[i]  24  (16 read + 8 write)
 *
 * Why 16 accumulators in read_sum: with one running sum, each add waits for
 * the previous add (FP add latency, ~3-4 cycles), so the loop measures the
 * adder, not memory. Independent partial sums break that chain. The spec
 * said 8; on the test machine 16 read 4-7% faster in interleaved runs, and
 * B should be the highest read rate the core can sustain, so 16 it is.
 * Experiment: build with -DONE_ACC to get the single-accumulator version.
 *
 * Saturation check: after the sweep, read_sum is rerun with 8 sums (stderr
 * only). If 16 sums beat 8 by more than 10%, the loop may still be the limit.
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
 * Vector width (x86 only): build with -march=native so the loop uses the
 * same wide loads (AVX2/AVX-512) that BLAS GEMV uses. Without it gcc emits
 * 16-byte SSE2 loads, and on the test machine read_sum came out ~30% below
 * what OpenBLAS GEMV itself pulled from memory: a "ceiling" GEMV can beat.
 * Apple Silicon needs no flag (NEON is always on).
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
#define ALIGN         4096         /* page-aligned arrays */

#define BARRIER() __asm__ volatile("" ::: "memory")

static volatile double sink;       /* consumer of every result */

/* ---- kernels: one full pass each ------------------------------------- */

__attribute__((noinline))
static double k_read_sum(const double *restrict a, size_t n) {
#if defined(ONE_ACC)
    double s = 0.0;                          /* one dependency chain: add-latency bound */
    for (size_t i = 0; i < n; i++) s += a[i];
    return s;
#else
    /* 16 independent sums in named variables (an array would live on the
     * stack, and every add would then wait on a store and a reload). */
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    for (size_t i = 0; i < n; i += 16) {     /* n is a multiple of 16 */
        s0 += a[i];      s1 += a[i + 1];  s2 += a[i + 2];  s3 += a[i + 3];
        s4 += a[i + 4];  s5 += a[i + 5];  s6 += a[i + 6];  s7 += a[i + 7];
        t0 += a[i + 8];  t1 += a[i + 9];  t2 += a[i + 10]; t3 += a[i + 11];
        t4 += a[i + 12]; t5 += a[i + 13]; t6 += a[i + 14]; t7 += a[i + 15];
    }
    return (((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))) +
           (((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)));
#endif
}

/* Saturation check only (stderr, never in the CSV): half as many sums. If
 * read_sum is much faster than this, more sums might help further and B may
 * be understated. */
__attribute__((noinline))
static double k_read_sum8(const double *restrict a, size_t n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (size_t i = 0; i < n; i += 8) {
        s0 += a[i];     s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
        s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
    }
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
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

/* ---- one timed run: `passes` full passes of one kernel ---------------- */

enum { READ_SUM, COPY, TRIAD, READ_SUM8 };
static const char *const KNAME[] = { "read_sum", "copy", "triad" };
static const double KBYTES[] = { 8.0, 16.0, 24.0 };   /* counted per element */

static double *A, *B, *C;
static size_t N;

static double timed(int k, uint64_t passes, size_t n) {
    double acc = 0.0;
    double t0 = now();
    for (uint64_t p = 0; p < passes; p++) {
        switch (k) {
        case READ_SUM: acc += k_read_sum(A, n);     break;
        case COPY:     k_copy(A, B, n);             break;
        case TRIAD:    k_triad(A, B, C, 3.0, n);    break;
        case READ_SUM8: acc += k_read_sum8(A, n);   break;
        }
        BARRIER();
    }
    double dt = now() - t0;
    sink = acc + A[N / 2];                   /* results are observable */
    return dt;
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

/* Warn if the laptop changed speed while one kernel was measured: the last
 * calibration pass (>= 100 ms, a full measurement) and the 5 timed runs should
 * agree. On a quiet, plugged-in laptop they agree within a few percent. */
static int unstable(const char *name, double warm, const double v[RUNS], double med) {
    double lo = v[0], hi = v[0];
    for (int i = 1; i < RUNS; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    double drift = med > warm ? med / warm - 1.0 : warm / med - 1.0;
    if (hi / lo - 1.0 > 0.10 || drift > 0.15) {
        fprintf(stderr, "WARNING: %s changed speed during the run (warm-up %.2f GB/s, "
                        "runs %.2f-%.2f GB/s). The laptop was busy or changed power state: "
                        "plug in, Best performance, close everything, don't switch windows, re-run.\n",
                name, warm, lo, hi);
        return 1;
    }
    return 0;
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
    double med[3];
    for (int k = READ_SUM; k <= TRIAD; k++) {
        uint64_t passes = 1;                 /* calibration = warm-up */
        double dt;
        while ((dt = timed(k, passes, N)) < MIN_SECONDS) passes *= 2;
        double warm = KBYTES[k] * (double)N * (double)passes / dt / 1e9;

        double gbs[RUNS];
        for (int r = 1; r <= RUNS; r++) {
            double dt = timed(k, passes, N);
            gbs[r - 1] = KBYTES[k] * (double)N * (double)passes / dt / 1e9;
            printf("%s,A2,%s,%llu,,%d,1,%.3f,GB_per_s\n",
                   machine_id, KNAME[k], (unsigned long long)bytes, r, gbs[r - 1]);
        }
        fflush(stdout);
        med[k] = median5(gbs);
        fprintf(stderr, "  %-8s %7.2f GB/s (median of %d, %llu passes/run)\n",
                KNAME[k], med[k], RUNS, (unsigned long long)passes);
        unstable(KNAME[k], warm, gbs, med[k]);
    }

    /* Is read_sum limited by memory or by its own loop? Rerun with half the
     * independent sums. Doubling the sums (8 -> 16) should gain little once
     * memory is the limit; a large gain means the loop may still be limiting.
     * (read_sum < copy is NOT a valid test: copy's counted 16 B/element omit
     * the write-allocate read, so copy really moves 24 B per element.) */
    {
        uint64_t passes = 1;
        double dt;
        while ((dt = timed(READ_SUM8, passes, N)) < MIN_SECONDS) passes *= 2;
        double warm = 8.0 * (double)N * (double)passes / dt / 1e9;
        double g[RUNS];
        for (int r = 0; r < RUNS; r++)
            g[r] = 8.0 * (double)N * (double)passes / timed(READ_SUM8, passes, N) / 1e9;
        double m8 = median5(g);
        double gain = med[READ_SUM] / m8 - 1.0;
        fprintf(stderr, "  check: 16 sums %.2f GB/s vs 8 sums %.2f GB/s (%+.1f%%)\n",
                med[READ_SUM], m8, 100.0 * gain);
        if (!unstable("8-sum check", warm, g, m8)) {
            if (gain > 0.10)
                fprintf(stderr, "WARNING: 8 -> 16 sums gained more than 10%%, so the loop may "
                                "still limit read_sum and B may be understated. Report it.\n");
            else if (gain < -0.10)
                fprintf(stderr, "WARNING: 8 sums beat 16 sums by more than 10%%. Both kernels "
                                "should run at memory speed, so the laptop most likely changed "
                                "speed between them. Fix the run conditions and re-run.\n");
        }
    }
    fprintf(stderr, "Plausibility: read_sum must be below the RAM's theoretical peak "
                    "(channels x 8 B x MT/s) recorded in machines.csv.\n");

    free_aligned(pa); free_aligned(pb); free_aligned(pc);
    return 0;
}
