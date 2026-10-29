/*
 * latency.c  --  Benchmark A1: load latency by pointer chasing.
 *
 * For each working-set size S (4 KiB up to 512 MiB, 4 sizes per doubling):
 *   1. Split S bytes into 128-byte slots.
 *   2. Link the slots into ONE random cycle (Sattolo's algorithm).
 *      Random order defeats the hardware prefetcher.
 *   3. Time p = p->next. Each load needs the previous load's result, so loads
 *      cannot overlap: elapsed time / number of loads = latency per load.
 *
 * Why 128-byte slots, not 64: Apple M-series caches use 128-byte lines, and
 * Intel's adjacent-line prefetcher fetches 64-byte lines in 128-byte pairs.
 * 128 bytes puts every slot on its own line (or line pair) on all machines.
 *
 * Controls built in (see spec, "Measurement controls"):
 *   - every measurement lasts >= 100 ms (timer resolution)
 *   - calibration doubles as warm-up; then 5 timed runs, all written out
 *   - final pointer goes to a volatile sink, so the loop cannot be deleted
 *   - the cycle is checked: it must visit every slot exactly once
 *   - fixed seed: every machine chases the same cycle for a given size
 *
 * Output: long-format CSV on stdout, one row per timed run:
 *   machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit
 * Progress goes to stderr.
 *
 * Build:  normally done by run_all.py; by hand: cc -O2 -Wall -o bench/latency bench/latency.c
 * Run:    bench/latency suraj-x86 > a1_latency.csv
 *         bench/latency suraj-x86 64      (stop at 64 MiB: quick test)
 */
#include "platform.h"            /* first: timer, allocation, pinning per OS */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SLOT          128          /* bytes per slot (see header) */
#define MIN_BYTES     4096u        /* 4 KiB */
#define MAX_MIB_DEF   512          /* default sweep ceiling */
#define RUNS          5            /* timed runs per size */
#define MIN_SECONDS   0.100        /* minimum duration of one timed run */
#define UNROLL        16           /* loads per loop iteration */

typedef struct slot {
    struct slot *next;
    char pad[SLOT - sizeof(struct slot *)];
} slot_t;
_Static_assert(sizeof(slot_t) == SLOT, "slot must be exactly SLOT bytes");

static slot_t *volatile sink;      /* consumer of the final pointer */

/* splitmix64: small, fast, good enough for shuffling */
static uint64_t rng_state;
static uint64_t rng(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Sattolo's algorithm: a random permutation that is a single n-cycle. */
static void build_cycle(slot_t *buf, uint32_t *perm, size_t n) {
    rng_state = 0x5EED0341ull;
    for (size_t i = 0; i < n; i++) perm[i] = (uint32_t)i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rng() % i);          /* 0 <= j < i */
        uint32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (size_t i = 0; i < n; i++) buf[i].next = &buf[perm[i]];
}

/* Walk the cycle once: it must return to the start after exactly n steps.
 * Also touches every slot, which pre-warms the caches and the TLB. */
static int cycle_ok(slot_t *buf, size_t n) {
    slot_t *p = buf;
    for (size_t i = 1; i < n; i++) {
        p = p->next;
        if (p == buf) return 0;                  /* closed too early */
    }
    return p->next == buf;
}

#define STEP   p = p->next;
#define STEP4  STEP STEP STEP STEP
#define STEP16 STEP4 STEP4 STEP4 STEP4

static slot_t *chase(slot_t *p, uint64_t iters) {
    for (uint64_t i = 0; i < iters; i++) { STEP16 }
    return p;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <machine_id> [max_MiB]\n", argv[0]);
        return 1;
    }
    const char *machine_id = argv[1];
    size_t max_bytes = (size_t)(argc > 2 ? atol(argv[2]) : MAX_MIB_DEF) << 20;
    size_t max_slots = max_bytes / SLOT;

    fprintf(stderr, "A1: %s\n", pin_to_one_core());

    slot_t *buf = alloc_aligned(4096, max_slots * SLOT);
    if (!buf) {
        fprintf(stderr, "could not allocate %llu MiB\n", (unsigned long long)(max_bytes >> 20));
        return 1;
    }
    uint32_t *perm = malloc(max_slots * sizeof *perm);
    if (!perm) { fprintf(stderr, "could not allocate index array\n"); return 1; }

    /* 4 sizes per doubling: x1, x2^(1/4), x2^(1/2), x2^(3/4) */
    static const double step[4] = {1.0, 1.189207115, 1.414213562, 1.681792831};

    printf("machine_id,benchmark,kernel,size_bytes,n,run,threads,value,unit\n");

    for (int k = 0; ; k++) {
        size_t bytes = (size_t)((double)((size_t)MIN_BYTES << (k / 4)) * step[k % 4]);
        bytes = bytes / SLOT * SLOT;
        if (bytes > max_bytes) break;
        size_t n = bytes / SLOT;

        build_cycle(buf, perm, n);
        if (!cycle_ok(buf, n)) {
            fprintf(stderr, "BUG: cycle check failed at %llu bytes\n", (unsigned long long)bytes);
            return 2;
        }

        /* Calibrate the run length; this is also the warm-up. */
        slot_t *p = buf;
        uint64_t iters = 1024;
        for (;;) {
            double t0 = now();
            p = chase(p, iters);
            double dt = now() - t0;
            if (dt >= MIN_SECONDS) break;
            iters = dt > 1e-3 ? (uint64_t)((double)iters * 1.25 * MIN_SECONDS / dt) + 1
                              : iters * 8;
        }

        double last = 0.0;
        for (int r = 1; r <= RUNS; r++) {
            double t0 = now();
            p = chase(p, iters);
            double dt = now() - t0;
            last = dt * 1e9 / ((double)iters * UNROLL);
            printf("%s,A1,pointer_chase,%llu,,%d,1,%.3f,ns_per_load\n",
                   machine_id, (unsigned long long)bytes, r, last);
        }
        sink = p;
        fflush(stdout);
        fprintf(stderr, "%11.1f KiB  %7.2f ns/load\n", (double)bytes / 1024.0, last);
    }

    free(perm);
    free_aligned(buf);
    return 0;
}
