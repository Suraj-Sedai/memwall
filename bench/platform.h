/*
 * platform.h  --  the only OS-specific code in the benchmarks.
 *
 * Include it FIRST in every benchmark .c file. It provides:
 *   now()              monotonic time in seconds
 *   alloc_aligned()    aligned allocation, freed with free_aligned()
 *   pin_to_one_core()  keep the benchmark on one core; returns a note for stderr
 *
 *                Windows (native)          macOS                    Linux / WSL2
 *   timer        QueryPerformanceCounter   clock_gettime            clock_gettime
 *   allocation   _aligned_malloc           posix_memalign           posix_memalign
 *   pinning      affinity: logical CPU 0   P-core request only      affinity: CPU 0
 *                + power throttling off
 */
#ifndef MEMWALL_PLATFORM_H
#define MEMWALL_PLATFORM_H

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00         /* Windows 10+: needed for power-throttling control */
#endif
#include <windows.h>
#include <malloc.h>
#elif defined(__APPLE__)
#define _DARWIN_C_SOURCE
#include <pthread.h>
#include <time.h>
#else
#define _GNU_SOURCE                 /* sched_setaffinity, posix_memalign */
#include <sched.h>
#include <time.h>
#endif
#include <stdlib.h>

static double now(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}

static void *alloc_aligned(size_t align, size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void *p = NULL;
    return posix_memalign(&p, align, bytes) == 0 ? p : NULL;
#endif
}

static void free_aligned(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static const char *pin_to_one_core(void) {
#if defined(_WIN32)
    /* Opt out of Windows power throttling ("EcoQoS"), which can lower the
     * clock of a process whose window is not in front. Ignored if unsupported. */
    PROCESS_POWER_THROTTLING_STATE pt = {0};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    pt.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof pt);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    return SetThreadAffinityMask(GetCurrentThread(), 1) ? "pinned to logical CPU 0"
                                                        : "WARNING: pinning failed";
#elif defined(__APPLE__)
    /* macOS cannot pin a thread to a core. Highest QoS asks the scheduler
     * for a performance core on Apple Silicon; not a guarantee. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    return "macOS: requested a performance core (pinning not possible)";
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0 ? "pinned to CPU 0"
                                                       : "WARNING: pinning failed";
#endif
}

#endif
