/*
 * bw_trace.c  --  DIAGNOSTIC ONLY (not a benchmark; writes no CSV).
 *
 * Shows read bandwidth over time, in 0.25 s windows, for a fixed sequence of
 * kernels. Tells apart two explanations for A2's slow read_sum:
 *   - the laptop's power management changes memory/CPU speed over time
 *     (then every kernel slows after a similar time, and idle may reset it)
 *   - something specific to the 16-sum kernel (then read_sum is slow in every
 *     phase while read_sum8 is fast in every phase)
 *
 * Kernels are identical to bench/bandwidth.c.
 *
 * Build (Windows):  gcc -O2 -fno-builtin -march=native -static -o bw_trace.exe bench\bw_trace.c
 * Run:              bw_trace.exe          (about 40 s; don't touch the laptop)
 */
#include "platform.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BARRIER() __asm__ volatile("" ::: "memory")
#define MIB 256
static volatile double sink;

__attribute__((noinline))
static double k_read_sum(const double *restrict a, size_t n) {
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
}

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

static void rest(double seconds) {
#if defined(_WIN32)
    Sleep((DWORD)(seconds * 1000));
#else
    struct timespec t = { (time_t)seconds, (long)((seconds - (time_t)seconds) * 1e9) };
    nanosleep(&t, NULL);
#endif
}

enum { RS16, RS8, COPY, IDLE };
static const char *const NAME[] = { "read_sum (16)", "read_sum8", "copy", "idle" };

static double *A, *B;
static size_t N;

/* Run one kernel for `secs`, printing GB/s (counted bytes) per 0.25 s window. */
static void phase(int k, double secs) {
    if (k == IDLE) {
        printf("\n%-14s %.0f s (CPU sleeps)\n", NAME[k], secs);
        fflush(stdout);
        rest(secs);
        return;
    }
    const double bytes_per_pass = (k == COPY ? 16.0 : 8.0) * (double)N;
    printf("\n%-14s GB/s per 0.25 s window:\n ", NAME[k]);
    double start = now(), win_start = start, win_bytes = 0, acc = 0;
    int col = 0;
    while (now() - start < secs) {
        switch (k) {
        case RS16: acc += k_read_sum(A, N);  break;
        case RS8:  acc += k_read_sum8(A, N); break;
        case COPY: k_copy(B, A, N);          break;
        }
        BARRIER();
        win_bytes += bytes_per_pass;
        double t = now();
        if (t - win_start >= 0.25) {
            printf(" %5.1f", win_bytes / (t - win_start) / 1e9);
            if (++col % 12 == 0) printf("\n ");
            fflush(stdout);
            win_start = t;
            win_bytes = 0;
        }
    }
    printf("\n");
    sink = acc;
}

int main(void) {
    fprintf(stderr, "bw_trace: %s\n", pin_to_one_core());
    N = ((size_t)MIB << 20) / sizeof(double);
    A = alloc_aligned(4096, (size_t)MIB << 20);
    B = alloc_aligned(4096, (size_t)MIB << 20);
    if (!A || !B) { fprintf(stderr, "allocation failed\n"); return 1; }
    for (size_t i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }

    printf("bw_trace: 2 arrays x %d MiB. About 40 s. Don't touch the laptop.\n", MIB);
    phase(RS16, 6);   /* same position as in A2: first kernel after setup */
    phase(RS8, 6);
    phase(COPY, 4);
    phase(RS16, 6);
    phase(IDLE, 4);
    phase(RS8, 4);    /* after idle: which kernel starts fast? */
    phase(RS16, 4);
    printf("\ndone\n");
    free_aligned(A);
    free_aligned(B);
    return 0;
}
