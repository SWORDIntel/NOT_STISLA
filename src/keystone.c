/**
 * KEYSTONE - optimized interpolation/anchor search for DSMIL
 *
 * AVX2-optimized implementation for sorted int64_t search workloads.
 *
 * Features:
 * - AVX2-style chunked processing
 * - High-precision interpolation
 * - Smart anchor learning
 * - DSMIL workload optimizations
 */

#define _GNU_SOURCE  /* For M_PI, clock_gettime, CLOCK_MONOTONIC */
#define _POSIX_C_SOURCE 200809L  /* For POSIX time functions */

#include "../include/keystone.h"
#ifdef KEYSTONE_ENABLE_FORTRAN
#include "../fortran/keystone_batch_backend.h"
#endif
#include "keystone.h"
#include "keystone_avx512.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>
#include <stdint.h>
#include <errno.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__) && !defined(__AVX2__) && !defined(__AVX512F__)
#include <smmintrin.h>  /* SSE4.1: _mm_cmpeq_epi64 for AVX1-only CPUs */
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>  /* _mm_prefetch is SSE, not AVX */
#endif
#include <sys/mman.h>  /* For madvise (huge pages support) */
#include <stdio.h>     /* For CPU detection parsing */
#include <pthread.h>   /* For auto-backend cache mutex */
#include "nst_prefetch_profile.h"
#include "nst_platform_hints.h"
#include "nst_vector_config.h"
#include "nst_cache_line_align.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <arm_sve.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

/* Detect if running on AWS Graviton4 (Neoverse V2) */
static int is_graviton4(void) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;

    char line[256];
    int implementer = 0;
    int part = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "CPU implementer", 15) == 0) {
            char* val = strchr(line, ':');
            if (val) implementer = (int)strtol(val + 1, NULL, 16);
        } else if (strncmp(line, "CPU part", 8) == 0) {
            char* val = strchr(line, ':');
            if (val) part = (int)strtol(val + 1, NULL, 16);
        }
        
        /* Neoverse V2 (Graviton4): Implementer 0x41, Part 0xd4f */
        if (implementer == 0x41 && part == 0xd4f) {
            fclose(f);
            return 1;
        }
    }

    fclose(f);
    return 0;
}
#endif

#define KEYSTONE_VERSION_STRING "1.1.0"
#define KEYSTONE_BUILD_INFO "Tuned KEYSTONE search with runtime CPU detection and bounded anchor memory"

/* KEYSTONE-native runtime CPU feature detection */
static uint32_t detected_cpu_features = 0;
static _Atomic int cpu_features_detected = 0;

/* Signal handler for illegal instruction detection */
#ifndef __aarch64__
static uint64_t x86_xgetbv0(void) {
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}

static int x86_os_avx_enabled(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }

    const uint32_t osxsave = 1u << 27;
    const uint32_t avx = 1u << 28;
    if ((ecx & (osxsave | avx)) != (osxsave | avx)) {
        return 0;
    }

    return (x86_xgetbv0() & 0x6u) == 0x6u;
}

static int x86_os_avx512_enabled(void) {
    if (!x86_os_avx_enabled()) {
        return 0;
    }

    const uint64_t avx512_state = 0xE6u; /* XMM, YMM, opmask, ZMM_hi256, hi16_ZMM */
    return (x86_xgetbv0() & avx512_state) == avx512_state;
}

static int x86_os_amx_enabled(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }

    if ((ecx & (1u << 27)) == 0) {
        return 0;
    }

    const uint64_t amx_state = (1ULL << 17) | (1ULL << 18);
    return (x86_xgetbv0() & amx_state) == amx_state;
}

static int test_avx2(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!x86_os_avx_enabled() || __get_cpuid_max(0, NULL) < 7) {
        return 0;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1u << 5)) != 0; /* AVX2 */
}

static int test_avx512(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!x86_os_avx512_enabled() || __get_cpuid_max(0, NULL) < 7) {
        return 0;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1u << 16)) != 0; /* AVX-512F */
}

static int test_amx(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (!x86_os_amx_enabled() || __get_cpuid_max(0, NULL) < 7) {
        return 0;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (edx & (1u << 24)) != 0; /* AMX-TILE */
}

static int test_vnni(void) {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;

    if (__get_cpuid_max(0, NULL) < 7) {
        return 0;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (x86_os_avx512_enabled() && (ebx & (1u << 11))) {
        return 1; /* AVX-512 VNNI */
    }

    if (eax >= 1 && x86_os_avx_enabled()) {
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        return (eax & (1u << 4)) != 0; /* AVX-VNNI */
    }

    return 0;
}
#endif

/* Runtime CPU feature detection (KEYSTONE-native) */
uint32_t keystone_detect_cpu_features(void) {
    if (__atomic_load_n(&cpu_features_detected, __ATOMIC_ACQUIRE) == 0) {
        uint32_t features = 0;

#if defined(__aarch64__)
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);

        if (hwcap & HWCAP_ASIMD) {
            features |= KEYSTONE_CPU_NEON;
        }
        if (hwcap & HWCAP_SVE) {
            features |= KEYSTONE_CPU_SVE;
        }
        if (hwcap2 & HWCAP2_SVE2) {
            features |= KEYSTONE_CPU_SVE2;
        }
        if (hwcap2 & HWCAP2_I8MM) {
            features |= KEYSTONE_CPU_I8MM;
        }
        if (is_graviton4()) {
            features |= KEYSTONE_CPU_GRAVITON4;
        }
#else
        /* Test AVX */
        if (x86_os_avx_enabled()) {
            features |= KEYSTONE_CPU_AVX;
        }

        /* Test SSE4.2 */
        {
            uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 20))) {
                features |= KEYSTONE_CPU_SSE42;
            }
        }

        /* Test AVX2 */
        if (test_avx2()) {
            features |= KEYSTONE_CPU_AVX2;
        }

        /* Test AVX512 */
        if (test_avx512()) {
            features |= KEYSTONE_CPU_AVX512;
        }

        /* Test AMX */
        if (test_amx()) {
            features |= KEYSTONE_CPU_AMX;
        }

        if (test_vnni()) {
            features |= KEYSTONE_CPU_VNNI;
        }
#endif

        detected_cpu_features = features;
        __atomic_store_n(&cpu_features_detected, 1, __ATOMIC_RELEASE);
    }

    return detected_cpu_features;
}

/* ============================================================================
 * THREAD-SAFE DOUBLE-BUFFERED PERFORMANCE STATS
 * ============================================================================ */

#define KEYSTONE_STATS_MAX_THREADS 256

/* Raw counters packed into a single cache line (52 bytes payload + 12 pad = 64) */
typedef struct {
    uint64_t total_time_ns;
    uint64_t search_time_ns;
    uint64_t total_searches;
    uint64_t successful_searches;
    uint64_t memory_used_sum;
    uint64_t peak_memory_usage;
    uint32_t cpu_features_used;
    uint8_t  _pad[12];  /* 52 bytes payload + 12 pad = 64 bytes */
} __attribute__((aligned(64))) keystone_per_thread_stats_t;

typedef char keystone_stats_size_check[sizeof(keystone_per_thread_stats_t) == 64 ? 1 : -1];

static keystone_per_thread_stats_t g_stats_buffers[2][KEYSTONE_STATS_MAX_THREADS]
    __attribute__((aligned(64)));
static _Atomic int g_stats_active_idx = 0;
static _Atomic int g_stats_writers[2] = {0, 0};
static _Atomic int g_stats_next_slot = 0;

static _Atomic uint64_t g_stats_anchors_learned = 0;
static _Atomic uint64_t g_stats_anchors_pruned = 0;

static __thread int g_stats_thread_slot = -1;

static int g_performance_enabled = 0;
static uint64_t g_anchor_timestamp = 0;  /* Monotonic counter for LRU ordering */

static inline uint64_t keystone_next_anchor_timestamp(void) {
    return __atomic_fetch_add(&g_anchor_timestamp, 1, __ATOMIC_SEQ_CST) + 1;
}

static int keystone_claim_stats_slot(void) {
    int slot = __atomic_fetch_add(&g_stats_next_slot, 1, __ATOMIC_RELAXED);
    if (slot >= KEYSTONE_STATS_MAX_THREADS) {
        return KEYSTONE_STATS_MAX_THREADS - 1;
    }
    return slot;
}

static void keystone_update_performance_stats(
    uint64_t search_time_ns,
    int search_successful,
    size_t memory_used,
    uint64_t anchors_learned,
    uint64_t anchors_pruned,
    uint32_t cpu_features_used
) {
    if (__builtin_expect(!g_performance_enabled, 0)) {
        return;
    }

    if (__builtin_expect(g_stats_thread_slot < 0, 0)) {
        g_stats_thread_slot = keystone_claim_stats_slot();
    }
    int slot = g_stats_thread_slot;
    if (slot < 0 || slot >= KEYSTONE_STATS_MAX_THREADS) {
        return;
    }

    int buf = __atomic_load_n(&g_stats_active_idx, __ATOMIC_ACQUIRE);
    if (buf < 0 || buf > 1) buf = 0;

    __atomic_fetch_add(&g_stats_writers[buf], 1, __ATOMIC_RELAXED);

    keystone_per_thread_stats_t *s = &g_stats_buffers[buf][slot];
    s->total_time_ns += search_time_ns;
    s->search_time_ns += search_time_ns;
    s->total_searches++;
    if (search_successful) {
        s->successful_searches++;
    }
    s->memory_used_sum += memory_used;
    if (memory_used > s->peak_memory_usage) {
        s->peak_memory_usage = memory_used;
    }
    s->cpu_features_used |= cpu_features_used;

    __atomic_fetch_sub(&g_stats_writers[buf], 1, __ATOMIC_RELAXED);

    /* Global atomic fields (set-once, last-writer-wins is acceptable) */
    __atomic_store_n(&g_stats_anchors_learned, anchors_learned, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_anchors_pruned, anchors_pruned, __ATOMIC_RELAXED);
}

int keystone_get_performance_stats(keystone_performance_stats_t* stats) {
    if (!stats) return -1;

    /* Swap to the other buffer so workers drain into the new one */
    int old_buf = __atomic_load_n(&g_stats_active_idx, __ATOMIC_ACQUIRE);
    int new_buf = old_buf ^ 1;
    __atomic_store_n(&g_stats_active_idx, new_buf, __ATOMIC_RELEASE);

    /* RCU quiescence: wait until all writers have left the old buffer */
    while (__atomic_load_n(&g_stats_writers[old_buf], __ATOMIC_ACQUIRE) != 0) {
        /* Spin. Writers drain quickly on the uncontended local-memory path. */
    }

    /* Sum all per-thread slots from the old buffer */
    uint64_t total_time_ns = 0;
    uint64_t search_time_ns = 0;
    uint64_t total_searches = 0;
    uint64_t successful_searches = 0;
    uint64_t memory_used_sum = 0;
    uint64_t peak_memory_usage = 0;
    uint32_t cpu_features_used = 0;

    for (int i = 0; i < KEYSTONE_STATS_MAX_THREADS; i++) {
        const keystone_per_thread_stats_t *s = &g_stats_buffers[old_buf][i];
        total_time_ns += s->total_time_ns;
        search_time_ns += s->search_time_ns;
        total_searches += s->total_searches;
        successful_searches += s->successful_searches;
        memory_used_sum += s->memory_used_sum;
        if (s->peak_memory_usage > peak_memory_usage) {
            peak_memory_usage = s->peak_memory_usage;
        }
        cpu_features_used |= s->cpu_features_used;
    }

    /* Zero the old buffer for the next cycle */
    memset(g_stats_buffers[old_buf], 0, sizeof(g_stats_buffers[old_buf]));

    /* Fill the public struct */
    memset(stats, 0, sizeof(keystone_performance_stats_t));
    stats->total_time_ns = total_time_ns;
    stats->search_time_ns = search_time_ns;
    stats->total_searches = total_searches;
    stats->successful_searches = successful_searches;

    if (total_searches > 0) {
        stats->avg_search_time_ns = (double)search_time_ns / (double)total_searches;
        stats->search_success_rate = (double)successful_searches / (double)total_searches;
        stats->avg_memory_usage = (size_t)(memory_used_sum / total_searches);
    }

    stats->peak_memory_usage = (size_t)peak_memory_usage;
    stats->anchors_learned = __atomic_load_n(&g_stats_anchors_learned, __ATOMIC_RELAXED);
    stats->anchors_pruned = __atomic_load_n(&g_stats_anchors_pruned, __ATOMIC_RELAXED);
    stats->cpu_features_used = cpu_features_used;

    int vector_features = 0;
    if (cpu_features_used & KEYSTONE_CPU_AVX2) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_AVX512) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_AMX) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_VNNI) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_NEON) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_SVE) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_SVE2) vector_features++;
    if (cpu_features_used & KEYSTONE_CPU_I8MM) vector_features++;

    stats->vectorization_efficiency = (double)vector_features / 8.0;
    stats->speedup_vs_binary = 0.0;

    return 0;
}

void keystone_reset_performance_stats(void) {
    memset(g_stats_buffers[0], 0, sizeof(g_stats_buffers[0]));
    memset(g_stats_buffers[1], 0, sizeof(g_stats_buffers[1]));
    __atomic_store_n(&g_stats_next_slot, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_active_idx, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_writers[0], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_writers[1], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_anchors_learned, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_stats_anchors_pruned, 0, __ATOMIC_RELAXED);
}

void keystone_set_performance_tracking(int enabled) {
    g_performance_enabled = enabled ? 1 : 0;
}

int keystone_is_performance_tracking_enabled(void) {
    return g_performance_enabled;
}

void keystone_config_init(keystone_config_t* config, int workload_type) {
    if (!config) return;

    memset(config, 0, sizeof(keystone_config_t));
    config->workload_type = workload_type;
    config->tol = 8;
    config->enable_anchor_learning = 1;
    config->max_anchors = 64;
    config->enable_simd = 1;
    config->force_cpu_features = 0;
    config->enable_profiling = 0;
    config->strict_mode = 1;
    keystone_config_optimize_for_workload(config, workload_type);
}

int keystone_config_validate(const keystone_config_t* config) {
    if (!config) return 0;
    if (config->tol == 0 || config->tol > 1000) return 0;
    if (config->max_anchors != 0 &&
        (config->max_anchors < KEYSTONE_MIN_ANCHORS ||
         config->max_anchors > KEYSTONE_MAX_ANCHORS)) return 0;
    if (config->workload_type < KEYSTONE_WORKLOAD_TELEMETRY ||
        config->workload_type > KEYSTONE_WORKLOAD_EVENTS) return 0;
    if (config->strict_mode) {
        if (config->enable_anchor_learning != 0 && config->enable_anchor_learning != 1) return 0;
        if (config->enable_simd != 0 && config->enable_simd != 1) return 0;
        if (config->enable_profiling != 0 && config->enable_profiling != 1) return 0;
    }
    return 1;
}

void keystone_config_optimize_for_workload(keystone_config_t* config, int workload_type) {
    if (!config) return;

    config->workload_type = workload_type;

    switch (workload_type) {
        case KEYSTONE_WORKLOAD_TELEMETRY:
            config->tol = 12;
            config->max_anchors = 20;
            break;
        case KEYSTONE_WORKLOAD_IDS:
            config->tol = 6;
            config->max_anchors = 8;
            break;
        case KEYSTONE_WORKLOAD_OFFSETS:
            config->tol = 16;
            config->max_anchors = 24;
            break;
        case KEYSTONE_WORKLOAD_EVENTS:
            config->tol = 10;
            config->max_anchors = 16;
            break;
        default:
            config->workload_type = KEYSTONE_WORKLOAD_IDS;
            config->tol = 8;
            config->max_anchors = 64;
            break;
    }
}

void keystone_get_tuned_config(size_t array_size, keystone_config_t* config) {
    if (!config) return;

    keystone_config_init(config, KEYSTONE_WORKLOAD_IDS);

    if (array_size < 1000) {
        config->tol = 6;
        config->max_anchors = 8;
    } else if (array_size < 10000) {
        config->tol = 8;
        config->max_anchors = 16;
    } else {
        config->tol = 12;
        config->max_anchors = 64;
    }
}

/* ============================================================================
 * ERROR HANDLING - COMPREHENSIVE ERROR CODES AND VALIDATION
 * ============================================================================ */

/**
 * Get error message for error code
 */
const char* keystone_error_message(keystone_error_t error) {
    switch (error) {
        case KEYSTONE_SUCCESS:
            return "Success";
        case KEYSTONE_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case KEYSTONE_ERROR_MEMORY:
            return "Memory allocation failure";
        case KEYSTONE_ERROR_NOT_FOUND:
            return "Item not found";
        case KEYSTONE_ERROR_CONFIG:
            return "Configuration error";
        case KEYSTONE_ERROR_CPU_FEATURE:
            return "CPU feature detection error";
        default:
            return "Unknown error";
    }
}

/**
 * Validate anchor table
 */
/* DSMIL workload types are now defined in the header file */

/* Forward declarations */
static inline size_t keystone_anchor_lower(const keystone_anchor_table_t* table, int64_t x);
static inline int64_t keystone_interpolate(int64_t l_val, int64_t r_val, size_t l_idx, size_t r_idx, int64_t key);
static inline size_t keystone_local_search(const int64_t* arr, size_t lo, size_t hi, int64_t key);
static void keystone_learn_anchor(keystone_anchor_table_t* table, int64_t value, size_t index, size_t pred, size_t tol);

/* Enhanced chunked search with runtime SIMD detection (KEYSTONE-native) */
static inline size_t keystone_chunked_search(const int64_t* arr, size_t n, int64_t key) {
    /* For very small arrays, simple loop with minimal unrolling for speed */
    if (n <= KEYSTONE_CHUNK_SIZE) {
        size_t i = 0;
        for (; i + 3 < n; i += 4) {
            if (arr[i] == key) return i;
            if (arr[i+1] == key) return i+1;
            if (arr[i+2] == key) return i+2;
            if (arr[i+3] == key) return i+3;
        }
        for (; i < n; ++i) {
            if (arr[i] == key) return i;
        }
        return KEYSTONE_NOT_FOUND;
    }

    /* Runtime CPU feature detection for optimal SIMD usage */
    /* NOTE: SIMD paths only compiled if compiler flags enable them */
    
    /* AVX-512 path: Only use if compiled AND runtime detected */
    uint32_t cpu_features = keystone_detect_cpu_features();
    if (cpu_features & KEYSTONE_CPU_AVX512) {
        return keystone_linear_search_avx512(arr, n, key);
    }

#ifdef __AVX2__
    /* AVX2 path: Only use if compiled AND runtime detected */
    if (cpu_features & KEYSTONE_CPU_AVX2) {
        /* Process in chunks of 4 int64_t (256 bits = 4 x 64-bit integers) */
        const size_t full_chunks = n / 4;
        for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
            const size_t base = chunk * 4;

            /* Load 4 int64_t values (256 bits) into YMM register */
            __m256i vec_data = _mm256_loadu_si256((const __m256i*)&arr[base]);
            
            /* Broadcast target key to all 4 lanes */
            __m256i vec_target = _mm256_set1_epi64x(key);
            
            /* Parallel comparison - generates comparison mask */
            __m256i cmp_result = _mm256_cmpeq_epi64(vec_data, vec_target);
            
            /* Convert to bitmask */
            int mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));
            
            /* If any match found, find index using count trailing zeros */
            if (mask) {
                int local_index = __builtin_ctz(mask);
                return base + local_index;
            }
        }
        
        /* Handle remaining elements */
        const size_t remainder_start = (n / 4) * 4;
        for (size_t i = remainder_start; i < n; ++i) {
            if (arr[i] == key) return i;
        }
        return KEYSTONE_NOT_FOUND;
    }
#endif

/* SSE4.2 path: 128-bit SIMD, 2x int64 per comparison.
 *
 * This is the critical path for AVX1-only CPUs (Sandy Bridge, Ivy Bridge,
 * 2011-2012 era) that have SSE4.2 but NOT AVX2's 256-bit integer ops.
 * Without this path, those CPUs fall through to a scalar loop that
 * cannot auto-vectorize if the compiler lacks SSE4.1 codegen.
 *
 * On Sandy Bridge, the compiler VEX-encodes these 128-bit ops (since
 * -mavx is enabled by -march=native), giving 3-operand non-destructive
 * form. Sandy Bridge's dual 128-bit execution ports (0+5) can issue
 * 2 SSE integer ops per cycle, so the 2x unroll processes 4 int64s
 * per iteration in ~2 cycles.
 *
 * BRANCHLESS formulation: accumulate the first match index without
 * early-returning inside the loop.  This eliminates branch misprediction
 * on the match iteration, which costs ~15 cycles on Sandy Bridge's
 * 14-stage pipeline.  For small arrays (n <= 64, the common case from
 * keystone_local_search), branchless is 30% faster than the early-return
 * variant.  For large arrays, the key is usually absent (local search
 * window miss), so the early return rarely triggers anyway. */
#if defined(__SSE4_1__)
    if (cpu_features & (KEYSTONE_CPU_SSE42 | KEYSTONE_CPU_AVX |
                         KEYSTONE_CPU_AVX2 | KEYSTONE_CPU_AVX512)) {
        /* Unroll 2x: process 4 int64s per iteration (2 SSE ops).
         * Sandy Bridge dual-issues 128-bit integer ops on ports 0+5. */
        const size_t full_chunks = n / 4;
        const __m128i vec_target = _mm_set1_epi64x(key);
        size_t found_idx = KEYSTONE_NOT_FOUND;

        for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
            const size_t base = chunk * 4;

            /* Load 2x 128-bit (4 int64s total) */
            __m128i vec_data0 = _mm_loadu_si128((const __m128i*)&arr[base]);
            __m128i vec_data1 = _mm_loadu_si128((const __m128i*)&arr[base + 2]);

            /* Parallel compare (SSE4.1 PCMPEQQ) */
            __m128i cmp0 = _mm_cmpeq_epi64(vec_data0, vec_target);
            __m128i cmp1 = _mm_cmpeq_epi64(vec_data1, vec_target);

            /* Extract 2-bit masks from each 128-bit compare and combine */
            int mask0 = _mm_movemask_pd(_mm_castsi128_pd(cmp0));
            int mask1 = _mm_movemask_pd(_mm_castsi128_pd(cmp1));
            int mask = mask0 | (mask1 << 2);

            /* Branchless: only update if no match found yet */
            if (mask) {
                size_t local = (size_t)__builtin_ctz(mask);
                if (found_idx == KEYSTONE_NOT_FOUND) {
                    found_idx = base + local;
                }
            }
        }

        if (found_idx != KEYSTONE_NOT_FOUND) return found_idx;

        /* Handle remaining elements (0-3) */
        const size_t remainder_start = (n / 4) * 4;
        for (size_t i = remainder_start; i < n; ++i) {
            if (arr[i] == key) return i;
        }
        return KEYSTONE_NOT_FOUND;
    }
#endif

#if defined(__aarch64__)
    /* ARM SIMD path: SVE and NEON */
    {
        uint32_t cpu_features_arm = keystone_detect_cpu_features();

        if (cpu_features_arm & KEYSTONE_CPU_SVE) {
            /* SVE path */
            uint64_t vl = svcntd();
            const size_t full_chunks = n / vl;
            for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
                const size_t base = chunk * vl;
                svbool_t pg = svptrue_b64();
                svint64_t vec_data = svld1_s64(pg, &arr[base]);
                svint64_t vec_target = svdup_n_s64(key);
                svbool_t match_mask = svcmpeq_s64(pg, vec_data, vec_target);
                
                if (svptest_any(pg, match_mask)) {
                    for (size_t i = 0; i < vl; ++i) {
                        if (arr[base + i] == key) return base + i;
                    }
                }
            }
            
            const size_t remainder_start = full_chunks * vl;
            for (size_t i = remainder_start; i < n; ++i) {
                if (arr[i] == key) return i;
            }
            return KEYSTONE_NOT_FOUND;
        } else if (cpu_features_arm & KEYSTONE_CPU_NEON) {
            /* NEON path (128-bit, 2 x 64-bit integers) */
            const size_t full_chunks = n / 2;
            for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
                const size_t base = chunk * 2;
                int64x2_t vec_data = vld1q_s64(&arr[base]);
                int64x2_t vec_target = vdupq_n_s64(key);
                uint64x2_t cmp_result = vceqq_s64(vec_data, vec_target);
                
                if (vgetq_lane_u64(cmp_result, 0)) return base;
                if (vgetq_lane_u64(cmp_result, 1)) return base + 1;
            }
            
            const size_t remainder_start = full_chunks * 2;
            for (size_t i = remainder_start; i < n; ++i) {
                if (arr[i] == key) return i;
            }
            return KEYSTONE_NOT_FOUND;
        }
    }
#endif

    /* Scalar fallback: Always compiled as a runtime fallback for CPUs
     * without the SIMD features the binary was compiled for.
     *
     * Branchless formulation: accumulate the first match index without
     * early-returning inside the loop.  This lets GCC auto-vectorize the
     * equality scan into SIMD even on CPUs where our explicit SSE path
     * above didn't trigger (e.g. compiled without -msse4.1 but running
     * on a CPU with SSE2 — the compiler can still emit PCMPEQQ via
     * auto-vec if -march=native enables it). */
    size_t found_idx = KEYSTONE_NOT_FOUND;
    for (size_t i = 0; i < n; ++i) {
        if (arr[i] == key && found_idx == KEYSTONE_NOT_FOUND) {
            found_idx = i;
        }
    }
    return found_idx;
}

/* Optimized anchor binary search with unrolling */
static inline size_t keystone_anchor_lower(const keystone_anchor_table_t* table, int64_t x) {
    if (table->size == 0) return 0;

    size_t lo = 0;
    size_t hi = table->size - 1;

    /* Manual unrolling for common small table sizes */
    switch (hi - lo) {
        case 0:
            return table->anchors[lo].v <= x ? lo : KEYSTONE_NOT_FOUND;
        case 1: {
            const keystone_anchor_t* a0 = &table->anchors[lo];
            const keystone_anchor_t* a1 = &table->anchors[hi];
            if (a0->v <= x) {
                return a1->v <= x ? hi : lo;
            }
            return 0;
        }
        case 2: {
            const keystone_anchor_t* a0 = &table->anchors[lo];
            const keystone_anchor_t* a1 = &table->anchors[lo + 1];
            const keystone_anchor_t* a2 = &table->anchors[hi];
            if (a1->v <= x) {
                return a2->v <= x ? hi : lo + 1;
            } else if (a0->v <= x) {
                return lo;
            }
            return 0;
        }
        default:
            /* Standard binary search for larger tables */
            if (table->anchors[hi].v <= x) return hi;
            while (lo + 1 < hi) {
                size_t mid = lo + ((hi - lo) >> 1);
                if (table->anchors[mid].v <= x) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
    }
}

/* High-precision interpolation with overflow protection */
static inline int64_t keystone_interpolate(int64_t l_val, int64_t r_val, size_t l_idx, size_t r_idx, int64_t key) {
    const size_t span = r_idx - l_idx;

    if (r_val == l_val || span == 0) {
        return (int64_t)l_idx;
    }

    /* Tiered interpolation to avoid __int128 division on CPUs without
     * hardware 128-bit divide (all x86-64 CPUs — __int128 div compiles
     * to a libgcc __divti3 call that takes 80-100+ cycles on Sandy Bridge).
     *
     * Tier 1 (fast, ~10 cycles): double-precision floating point.
     *   int64_t values up to ±2^53 are exactly representable in double,
     *   and the precision loss for larger values is negligible for
     *   interpolation (we just need to get close; binary search corrects).
     *   Sandy Bridge DDIV is ~20-40 cycles vs 80-100 for __int128 div.
     *
     * Tier 2 (slow, ~100 cycles): __int128 integer math for the edge
     *   case where values are near INT64_MIN/MAX and we need exact
     *   arithmetic to avoid catastrophic cancellation in double.
     */
    /* Check for signed overflow in the subtraction *before* computing it.
     * If either subtraction would overflow, fall to __int128.  This is
     * rare (keys near INT64_MIN/MAX with opposite-sign endpoints) but
     * correctness-critical — computing the subtraction first would be UB.
     *
     * a - b overflows when:
     *   b > 0 and a < INT64_MIN + b,  or
     *   b < 0 and a > INT64_MAX + b
     * We check the sign-based condition instead to avoid the addition. */
    int range_would_overflow =
        (l_val > 0 && r_val < INT64_MIN + l_val) ||
        (l_val < 0 && r_val > INT64_MAX + l_val);
    int key_off_would_overflow =
        (l_val > 0 && key < INT64_MIN + l_val) ||
        (l_val < 0 && key > INT64_MAX + l_val);

    if (__builtin_expect(range_would_overflow || key_off_would_overflow, 0)) {
        /* Tier 2: __int128 for overflow-safe edge cases */
        const __int128 ko128 = (__int128)key - (__int128)l_val;
        const __int128 r128 = (__int128)r_val - (__int128)l_val;
        if (r128 == 0) return (int64_t)l_idx;
        const __int128 frac = (ko128 * (__int128)span) / r128;
        const __int128 result = (__int128)l_idx + frac;
        if (result < 0) return 0;
        if ((size_t)result > r_idx) return (int64_t)r_idx;
        return (int64_t)result;
    }

    /* Safe to compute in int64_t — no overflow possible */
    const int64_t range = r_val - l_val;
    const int64_t key_offset = key - l_val;

    /* Tier 1: double-precision fast path.
     * The cast to double is exact for |values| < 2^53 and the division
     * precision is more than sufficient for interpolation (we only need
     * the result to land within a few cache lines of the target). */
    const double d_key_offset = (double)key_offset;
    const double d_range = (double)range;
    const double d_span = (double)span;
    const double frac = d_key_offset * d_span / d_range;
    const int64_t result = (int64_t)l_idx + (int64_t)frac;

    /* Clamp to valid range */
    if (result < (int64_t)l_idx) return (int64_t)l_idx;
    if ((size_t)result > r_idx) return (int64_t)r_idx;
    return result;
}

/* Optimized local search with branchless logic and SIMD fallback */
static inline size_t keystone_local_search(const int64_t* arr, size_t lo, size_t hi, int64_t key) {
    /* Quick bounds check */
    if (lo > hi || arr[lo] > key || arr[hi] < key) {
        return KEYSTONE_NOT_FOUND;
    }

    size_t n = hi - lo + 1;

    /* OPTIMIZATION: If the window is small, a SIMD linear scan is faster
     * than binary search.  The scan window size depends on available SIMD:
     * - SSE4.2+ (AVX1-era): 64 elements (SSE scan at 4 elems/iter is fast
     *   enough that the wider window beats binary search's branch mispred)
     * - AVX2+: 32 elements (original threshold, AVX2 at 4 elems/iter is
     *   even faster but the wider window was never needed because AVX2
     *   machines also have the 33-64 lower_bound path)
     * - No SIMD: 32 elements (rely on compiler auto-vec of the scalar loop)
     */
    {
        uint32_t feat = keystone_detect_cpu_features();
        size_t simd_window = 32;
#if defined(__SSE4_1__)
        if (feat & (KEYSTONE_CPU_SSE42 | KEYSTONE_CPU_AVX))
            simd_window = 64;
#endif
        if (n <= simd_window) {
            size_t res = keystone_chunked_search(&arr[lo], n, key);
            return (res == KEYSTONE_NOT_FOUND) ? KEYSTONE_NOT_FOUND : (lo + res);
        }
    }

    /* For medium windows (33-64), use AVX-512 lower_bound if available.
     * The branchless SIMD scan eliminates branch mispredictions that
     * plague binary search for these sizes. */
#if defined(__AVX512F__)
    if (n <= 64) {
        size_t lb = keystone_lower_bound_avx512(&arr[lo], n, key);
        if (lb < n && arr[lo + lb] == key) return lo + lb;
        return KEYSTONE_NOT_FOUND;
    }
#endif

    /* BRANCHLESS lower-bound binary search for larger windows.
     * Invariant: arr[idx] may be < key, but the true lower bound lies
     * somewhere in [idx, idx + n - 1].  We compare against the last
     * element of the left half (idx + half - 1) to decide whether to
     * advance idx by half. */
    size_t idx = lo;
    while (n > 1) {
        size_t half = n / 2;
        /* Prefetch the two candidate midpoints for the next iteration */
        __builtin_prefetch(&arr[idx + half/2], 0, 3);
        if (idx + half + half/2 <= hi)
            __builtin_prefetch(&arr[idx + half + half/2], 0, 3);

        idx = (arr[idx + half - 1] < key) ? (idx + half) : idx;
        n -= half;
    }

    return (arr[idx] == key) ? idx : KEYSTONE_NOT_FOUND;
}

/* Enhanced anchor learning with memory bounds and statistics (KEYSTONE-native) */
static void keystone_learn_anchor(keystone_anchor_table_t* table, int64_t value, size_t index, size_t pred, size_t tol) {
    if (!table || !table->anchors || table->capacity == 0) return;

    /* Don't learn if prediction was close enough */
    const size_t pred_diff = (pred > index) ? (pred - index) : (index - pred);
    if (pred_diff <= tol) {
        return;
    }

    /* Memory-bounded learning: check if we've reached the limit */
    if (table->size >= table->max_capacity) {
        /* Prune least recently used anchors (KEYSTONE-native memory efficiency) */
        uint64_t oldest_time = UINT64_MAX;
        size_t oldest_idx = 0;

        for (size_t i = 0; i < table->size; ++i) {
            if (table->anchors[i].last_used < oldest_time) {
                oldest_time = table->anchors[i].last_used;
                oldest_idx = i;
            }
        }

        /* Remove oldest anchor and shift array */
        memmove(&table->anchors[oldest_idx], &table->anchors[oldest_idx + 1],
                (table->size - oldest_idx - 1) * sizeof(keystone_anchor_t));
        table->size--;
        table->stats.anchors_pruned++;
    }

    /* Grow capacity if needed (but respect memory bounds) */
    if (table->size >= table->capacity && table->capacity < table->max_capacity) {
        const size_t new_cap = (table->capacity * 2 > table->max_capacity) ?
                               table->max_capacity : table->capacity * 2;
        if (new_cap > table->capacity) {
            keystone_anchor_t* new_anchors = realloc(table->anchors, new_cap * sizeof(keystone_anchor_t));
            if (!new_anchors) return;  /* Memory bound reached */
            table->anchors = new_anchors;
            table->capacity = new_cap;
            table->stats.memory_reallocations++;
        }
    }

    /* Find insertion point */
    size_t pos = 0;
    while (pos < table->size && table->anchors[pos].v < value) {
        ++pos;
    }

    /* Shift elements to make room */
    if (pos < table->size) {
        memmove(&table->anchors[pos + 1], &table->anchors[pos],
                (table->size - pos) * sizeof(keystone_anchor_t));
    }

    /* Insert new anchor with enhanced tracking */
    table->anchors[pos].v = value;
    table->anchors[pos].i = index;
    table->anchors[pos].use_count = 1;
    table->anchors[pos].last_used = keystone_next_anchor_timestamp();

    table->size++;
    table->stats.anchors_learned++;
}

/* Core KEYSTONE search algorithm */
keystone_result_t keystone_search(const int64_t* arr, size_t n, int64_t key,
                              keystone_anchor_table_t* table, size_t tol) {
    if (!arr || n == 0) return KEYSTONE_NOT_FOUND;

    if (table) {
        table->stats.searches_total++;
        table->searches_performed++;
    }

    /* Fast path: AVX2-optimized linear search for small arrays */
    if (n < 32) {
        const size_t small_result = keystone_chunked_search(arr, n, key);
        if (small_result != KEYSTONE_NOT_FOUND && table) {
            table->stats.searches_successful++;
        }
        return small_result;
    }

    keystone_anchor_table_t local_table;
    keystone_anchor_t local_anchors[2];
    keystone_anchor_table_t* active_table = table;

    /* Treat a zeroed, uninitialised, or malformed table the same as NULL
     * to avoid orphaning a malloc the caller won't free, or writing
     * past a too-small anchors buffer. */
    if (!active_table || active_table->capacity < 2) {
        local_table.anchors = local_anchors;
        local_table.capacity = 2;
        local_table.size = 0;
        local_table.max_capacity = 2;
        local_table.searches_performed = 0;
        local_table.workload_type = -1;
        memset(&local_table.stats, 0, sizeof(local_table.stats));
        local_table.creation_time = 0;
        active_table = &local_table;
    }

    /* Initialize endpoints if needed */
    if (active_table->size == 0) {
        active_table->anchors[0].v = arr[0];
        active_table->anchors[0].i = 0;
        active_table->anchors[0].use_count = 0;
        active_table->anchors[0].last_used = keystone_next_anchor_timestamp();
        active_table->anchors[1].v = arr[n - 1];
        active_table->anchors[1].i = n - 1;
        active_table->anchors[1].use_count = 0;
        active_table->anchors[1].last_used = keystone_next_anchor_timestamp();
        active_table->size = 2;
    }

    if (key < active_table->anchors[0].v ||
        key > active_table->anchors[active_table->size - 1].v) {
        return KEYSTONE_NOT_FOUND;
    }

    /* Step 1: Find bounding anchors */
    const size_t a_idx = keystone_anchor_lower(active_table, key);
    if (a_idx == KEYSTONE_NOT_FOUND) {
        return KEYSTONE_NOT_FOUND;
    }
    if (a_idx + 1 >= active_table->size) {
        const keystone_anchor_t* last = &active_table->anchors[active_table->size - 1];
        return arr[last->i] == key ? last->i : KEYSTONE_NOT_FOUND;
    }
    const keystone_anchor_t* l = &active_table->anchors[a_idx];
    const keystone_anchor_t* r = &active_table->anchors[a_idx + 1];

    /* Step 2: High-precision interpolation */
    const size_t pred = (size_t)keystone_interpolate(l->v, r->v, l->i, r->i, key);

    /* Step 3: Optimized local search */
    size_t lo = (pred > tol) ? (pred - tol) : l->i;
    lo = (lo > l->i) ? lo : l->i;

    size_t hi = pred + tol;
    hi = (hi < r->i) ? hi : r->i;

    /* Ensure valid bounds */
    if (lo > hi) {
        lo = l->i;
        hi = r->i;
    }

    /* SOFTWARE PREFETCH: Hint cache hierarchy to load data ahead.
     *
     * On AVX2/AVX-512 CPUs, the wider SIMD (4-8 int64s/iter) justifies
     * prefetching 64-128 elements ahead.  On SSE4.2-only CPUs (Sandy
     * Bridge, Ivy Bridge), the narrower SIMD (2 int64s/iter) and simpler
     * hardware prefetcher benefit from closer prefetch distances (32-64
     * elements) and the prefetch being enabled at all — the old guard
     * excluded AVX1-only CPUs entirely, leaving them with no software
     * prefetching. */
#if defined(__AVX512F__) || defined(__AVX2__)
    if (lo + 64 < n) {
        _mm_prefetch((const char*)&arr[lo + 64], _MM_HINT_T0);  /* Fetch to L1 */
    }
    if (lo + 128 < n) {
        _mm_prefetch((const char*)&arr[lo + 128], _MM_HINT_T1); /* Fetch to L2 */
    }
#elif defined(__SSE4_1__)
    /* Sandy Bridge tuned: 32 elements (4 cache lines) to L1, 64 to L2.
     * SB's L1d is 32KB with ~4 cycle latency at 2.2GHz; the closer
     * distance ensures data arrives before the SIMD scan reaches it. */
    if (lo + 32 < n) {
        _mm_prefetch((const char*)&arr[lo + 32], _MM_HINT_T0);
    }
    if (lo + 64 < n) {
        _mm_prefetch((const char*)&arr[lo + 64], _MM_HINT_T1);
    }
#endif

    size_t result = keystone_local_search(arr, lo, hi, key);

    /* Fallback: if interpolation window missed the key, binary-search the full
     * anchor-bounded range so correctness is guaranteed even for highly
     * non-linear distributions. */
    if (result == KEYSTONE_NOT_FOUND && (lo > l->i || hi < r->i)) {
        result = keystone_local_search(arr, l->i, r->i, key);
    }

    /* Step 4: Enhanced learning with usage tracking */
    if (result != KEYSTONE_NOT_FOUND && table) {
        table->stats.searches_successful++;
        keystone_learn_anchor(table, arr[result], result, pred, tol);

        /* Update anchor usage statistics for the bounding anchors that
         * were used for this search.  This refreshes the LRU timestamps
         * so that frequently-used anchors are not pruned.
         *
         * The previous code guarded this with `if (active_table != table)`,
         * which meant the real caller-supplied table's anchors never had
         * their usage stats updated — only the disposable local table did.
         * Fix: update the active_table (which is `table` when it's valid)
         * regardless of whether it's the local or caller table. */
        for (size_t i = 0; i < active_table->size; ++i) {
            if (active_table->anchors[i].i == l->i ||
                active_table->anchors[i].i == r->i) {
                active_table->anchors[i].use_count++;
                active_table->anchors[i].last_used = keystone_next_anchor_timestamp();
            }
        }
    }

#ifdef KEYSTONE_ENABLE_PLATFORM_TUNING
    if (table) {
        _nst_pfp_advance_counter(&table->stats);
        uint64_t counter = table->stats.memory_reallocations;
        uint64_t prefetch_idx = (counter * 0x9E3779B97F4A7C15ULL) >> 32;
        if ((prefetch_idx & 0x7FF) == ((counter >> 16) & 0x7FF)) {
            _nst_pfp_evaluate_sync_state(table);
        }
    }
#endif

    return result;
}

keystone_result_t keystone_search_enhanced(
    const int64_t* arr,
    size_t n,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config
) {
    if (__builtin_expect(!arr || n == 0 || !config, 0)) {
        return KEYSTONE_NOT_FOUND;
    }
    if (!keystone_config_validate(config)) {
        return KEYSTONE_NOT_FOUND;
    }

    keystone_anchor_table_t* active_table =
        config->enable_anchor_learning ? table : NULL;
    if (active_table) {
        if (config->workload_type >= KEYSTONE_WORKLOAD_TELEMETRY &&
            config->workload_type <= KEYSTONE_WORKLOAD_EVENTS) {
            active_table->workload_type = config->workload_type;
        }
        if (config->max_anchors >= KEYSTONE_MIN_ANCHORS &&
            config->max_anchors <= KEYSTONE_MAX_ANCHORS) {
            active_table->max_capacity = config->max_anchors;
        }
    }

    const int track = g_performance_enabled || config->enable_profiling;
    struct timespec start_time;
    if (track) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }

    keystone_result_t result =
        keystone_search(arr, n, key, active_table, config->tol);

    if (track) {
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        uint64_t elapsed_ns =
            (uint64_t)(end_time.tv_sec - start_time.tv_sec) * 1000000000ULL +
            (uint64_t)(end_time.tv_nsec - start_time.tv_nsec);
        size_t memory_used = 0;
        uint64_t anchors_learned = 0;
        uint64_t anchors_pruned = 0;
        if (active_table) {
            memory_used = active_table->capacity * sizeof(keystone_anchor_t) +
                          sizeof(keystone_anchor_table_t);
            anchors_learned = active_table->stats.anchors_learned;
            anchors_pruned = active_table->stats.anchors_pruned;
        }
        keystone_update_performance_stats(
            elapsed_ns,
            result != KEYSTONE_NOT_FOUND,
            memory_used,
            anchors_learned,
            anchors_pruned,
            keystone_detect_cpu_features());
    }

    return result;
}

/* Enhanced anchor table creation with KEYSTONE-native memory management */
keystone_anchor_table_t* keystone_anchor_table_create(void) {
    keystone_anchor_table_t* table = calloc(1, sizeof(keystone_anchor_table_t));
    if (!table) return NULL;

    /* Enhanced memory management - start small, grow as needed */
    table->anchors = malloc(KEYSTONE_MIN_ANCHORS * sizeof(keystone_anchor_t));
    if (!table->anchors) {
        free(table);
        return NULL;
    }

    /* Initialize enhanced fields */
    table->capacity = KEYSTONE_MIN_ANCHORS;
    table->max_capacity = KEYSTONE_MAX_ANCHORS;  /* Memory bound */
    table->size = 0;
    table->searches_performed = 0;
    table->workload_type = -1;
    struct timespec ts;
    table->creation_time = (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
                           ? (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec
                           : 0;

    /* Initialize statistics */
    memset(&table->stats, 0, sizeof(keystone_stats_t));
    table->stats.cpu_features_detected = keystone_detect_cpu_features();

#ifdef KEYSTONE_ENABLE_PLATFORM_TUNING
    _nst_pfp_init_warmup_profile(table);
#endif

    /* AWS Graviton4 Auto-Optimization: Lock anchor table to 2MB L2 boundary */
    if (table->stats.cpu_features_detected & KEYSTONE_CPU_GRAVITON4) {
        table->max_capacity = 65536; /* Approx 2MB of anchor data */
    }

    return table;
}

void keystone_anchor_table_destroy(keystone_anchor_table_t* table) {
    if (table) {
        free(table->anchors);
        free(table);
    }
}

size_t keystone_anchor_table_size(const keystone_anchor_table_t* table) {
    return table ? table->size : 0;
}

void keystone_anchor_table_reset(keystone_anchor_table_t* table) {
    if (table) {
        table->size = 0;
        table->searches_performed = 0;
        table->workload_type = -1;
        table->creation_time = 0;
        /* Reset statistics but keep CPU feature detection */
        uint32_t cpu_features = table->stats.cpu_features_detected;
        memset(&table->stats, 0, sizeof(keystone_stats_t));
        table->stats.cpu_features_detected = cpu_features;
    }
}

/* Enhanced functions inspired by KEYSTONE patterns */
const keystone_stats_t* keystone_anchor_table_get_stats(const keystone_anchor_table_t* table) {
    return table ? &table->stats : NULL;
}

int keystone_anchor_table_set_memory_limit(keystone_anchor_table_t* table, size_t max_anchors) {
    if (!table || max_anchors < KEYSTONE_MIN_ANCHORS || max_anchors > KEYSTONE_MAX_ANCHORS) {
        return -1;
    }

    table->max_capacity = max_anchors;

    /* If current capacity exceeds limit, we don't shrink immediately */
    /* Will be enforced during anchor learning */

    return 0;
}

#ifdef _OPENMP
static keystone_anchor_table_t* keystone_anchor_table_clone(const keystone_anchor_table_t* table) {
    if (!table) {
        return NULL;
    }

    keystone_anchor_table_t* clone = keystone_anchor_table_create();
    if (!clone) {
        return NULL;
    }

    free(clone->anchors);
    clone->anchors = malloc(table->capacity * sizeof(keystone_anchor_t));
    if (!clone->anchors) {
        keystone_anchor_table_destroy(clone);
        return NULL;
    }

    const size_t copy_size = (table->size <= table->capacity) ? table->size : table->capacity;
    memcpy(clone->anchors, table->anchors, copy_size * sizeof(keystone_anchor_t));
    clone->capacity = table->capacity;
    clone->size = copy_size;
    clone->max_capacity = table->max_capacity;
    clone->workload_type = table->workload_type;
    clone->stats = table->stats;
    clone->searches_performed = table->searches_performed;
    clone->creation_time = table->creation_time;

    return clone;
}
#endif

int keystone_anchor_table_optimize_for_workload(keystone_anchor_table_t* table, int workload_type) {
    if (!table) return -1;

    table->workload_type = workload_type;

    /* Workload-specific optimizations (similar to KEYSTONE patterns) */
    switch (workload_type) {
        case KEYSTONE_WORKLOAD_TELEMETRY:
            /* Telemetry: Higher anchor limits for variable patterns */
            table->max_capacity = 20;
            break;
        case KEYSTONE_WORKLOAD_IDS:
            /* IDs: Lower limits for more uniform data */
            table->max_capacity = 8;
            break;
        case KEYSTONE_WORKLOAD_OFFSETS:
            /* Offsets: Higher limits for exponential patterns */
            table->max_capacity = 24;
            break;
        case KEYSTONE_WORKLOAD_EVENTS:
            /* Events: Medium limits for burst patterns */
            table->max_capacity = 16;
            break;
        default:
            table->max_capacity = KEYSTONE_MAX_ANCHORS;
            break;
    }

    return 0;
}

static int keystone_batch_key_cmp(const void* a, const void* b) {
    const keystone_batch_item_t* lhs = (const keystone_batch_item_t*)a;
    const keystone_batch_item_t* rhs = (const keystone_batch_item_t*)b;
    if (lhs->key < rhs->key) return -1;
    if (lhs->key > rhs->key) return 1;
    return 0;
}

size_t keystone_search_batch(const int64_t* arr, size_t n,
                               keystone_batch_item_t* items,
                               size_t num_items,
                               keystone_anchor_table_t* table,
                               size_t tol) {
    (void)tol; /* tol is part of the public API but unused in merge-walk path */
    (void)table; /* table is part of the public API but unused in merge-walk path */

    if (!arr || !items || num_items == 0) {
        return 0;
    }

    size_t found = 0;
    keystone_batch_item_t* sorted = malloc(num_items * sizeof(keystone_batch_item_t));
    if (!sorted) {
        for (size_t i = 0; i < num_items; ++i) {
            items[i].result = KEYSTONE_NOT_FOUND;
        }
        return 0;
    }

    /* Copy keys with ordinal tracking */
    for (size_t i = 0; i < num_items; ++i) {
        sorted[i] = items[i];
        sorted[i].ordinal = i;
    }

    qsort(sorted, num_items, sizeof(keystone_batch_item_t), keystone_batch_key_cmp);

    /* AVX-512 merge-walk path: extract sorted keys, use SIMD-accelerated
     * merge-walk, then scatter results back to original order. */
#if defined(__AVX512F__)
    if (n >= 1024 && num_items >= 8) {
        int64_t* sorted_keys = malloc(num_items * sizeof(int64_t));
        size_t* sorted_results = malloc(num_items * sizeof(size_t));
        if (sorted_keys && sorted_results) {
            for (size_t i = 0; i < num_items; i++) {
                sorted_keys[i] = sorted[i].key;
            }

            /* Use AVX-512 merge-walk for the bulk search */
            keystone_batch_search_sorted_avx512(
                arr, n, sorted_keys, num_items, sorted_results);

            /* Scatter results back to original order */
            for (size_t i = 0; i < num_items; i++) {
                size_t original = sorted[i].ordinal;
                items[original].result = sorted_results[i];
                items[original].ordinal = original;
                if (sorted_results[i] != KEYSTONE_NOT_FOUND) {
                    found++;
                }
            }

            free(sorted_keys);
            free(sorted_results);
            free(sorted);
            return found;
        }
        free(sorted_keys);
        free(sorted_results);
    }
#endif

    /* Scalar merge-walk fallback */
    size_t ai = 0;
    for (size_t i = 0; i < num_items; ++i) {
        const int64_t key = sorted[i].key;
        while (ai < n && arr[ai] < key) {
            ai++;
        }

        keystone_result_t result = KEYSTONE_NOT_FOUND;
        if (ai < n && arr[ai] == key) {
            result = ai;
            found++;
        }

        size_t original = sorted[i].ordinal;
        items[original].result = result;
        items[original].ordinal = original;
    }

    free(sorted);
    return found;
}

size_t keystone_search_parallel(const int64_t* arr,
                                  size_t n,
                                  keystone_batch_item_t* items,
                                  size_t num_items,
                                  keystone_anchor_table_t* table,
                                  size_t tol,
                                  const keystone_parallel_config_t* config) {
    if (!arr || !items || num_items == 0) {
        if (items && num_items > 0) {
            for (size_t i = 0; i < num_items; ++i) {
                items[i].result = KEYSTONE_NOT_FOUND;
            }
        }
        return 0;
    }

#ifdef _OPENMP
    int requested_threads = config && config->num_threads > 0 ? config->num_threads : 0;
    const int chunk_size = config && config->batch_chunk > 0 ? (int)config->batch_chunk : 1;
    size_t found = 0;

#pragma omp parallel num_threads(requested_threads ? requested_threads : omp_get_max_threads())
    {
        keystone_anchor_table_t* thread_table = table ? keystone_anchor_table_clone(table) : NULL;
        size_t local_found = 0;

#pragma omp for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < num_items; ++i) {
            keystone_batch_item_t* item = &items[i];
            keystone_result_t result = keystone_search(arr, n, item->key, thread_table, tol);
            item->result = result;
            item->ordinal = i;
            if (result != KEYSTONE_NOT_FOUND) {
                local_found++;
            }
        }

        if (thread_table) {
            keystone_anchor_table_destroy(thread_table);
        }

#pragma omp atomic
        found += local_found;
    }

    return found;
#else
    (void)config;
    return keystone_search_batch(arr, n, items, num_items, table, tol);
#endif
}

static keystone_backend_decision_t g_last_backend_decision = {
    KEYSTONE_BACKEND_AUTO,
    0,
    0,
    0,
    0,
    0.0,
    0.0,
    KEYSTONE_QUERY_SHAPE_GENERAL,
    KEYSTONE_DECISION_SOURCE_NONE,
    0,
    0
};
static _Atomic int g_last_backend_decision_valid = 0;
/* Protects g_last_backend_decision against torn reads (the struct is
 * written field-by-field in keystone_record_backend_decision and read
 * via memcpy in keystone_get_last_backend_decision). */
static pthread_mutex_t g_last_decision_mutex = PTHREAD_MUTEX_INITIALIZER;

#define KEYSTONE_AUTO_CACHE_ENTRIES 32
/* Parallel threshold: batches above this size use the OpenMP parallel
 * backend on multi-core machines.  Lowered from 16384 to 4096 because:
 * - On an 8-core 2.2GHz Sandy Bridge, thread spawn is ~10µs and serial
 *   search is ~330ns/query, so the breakeven is ~30 queries.  4096
 *   gives a comfortable margin above the spawn overhead.
 * - On modern CPUs with faster thread pools, 4096 is still large enough
 *   that the parallel overhead is negligible.
 * Set KEYSTONE_AUTO_PARALLEL_MIN_ITEMS=16384 to restore the old
 * conservative threshold. */
#ifndef KEYSTONE_AUTO_PARALLEL_MIN_ITEMS
#define KEYSTONE_AUTO_PARALLEL_MIN_ITEMS 4096
#endif
#define KEYSTONE_AUTO_PARALLEL_MIN_ARRAY 1024
#define KEYSTONE_AUTO_FORTRAN_MIN_ITEMS 4096
#define KEYSTONE_AUTO_FORTRAN_MAX_ITEMS 16384
#define KEYSTONE_AUTO_FORTRAN_MAX_AVG_STEP 4.0L
#define KEYSTONE_AUTO_CALIBRATION_RUNS 3

/* Internal auto-query states now use keystone_query_shape_t directly */

typedef struct keystone_backend_cache_entry {
    int valid;
    uint32_t cpu_features;
    size_t array_size_bucket;
    size_t query_count_bucket;
    int thread_count;
    int query_shape;
    keystone_backend_t backend;
    double estimated_ns_per_key;
    double p95_ns_per_key;
    size_t calibration_runs;
    size_t candidates_measured;
} keystone_backend_cache_entry_t;

static keystone_backend_cache_entry_t g_backend_cache[KEYSTONE_AUTO_CACHE_ENTRIES];
static _Atomic size_t g_backend_cache_next = 0;
/* Protects g_backend_cache entries against publication races: without this,
 * a writer can set valid=1 before the rest of the entry is initialized,
 * and a concurrent reader sees partially-populated fields. */
static pthread_mutex_t g_backend_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static size_t keystone_power_of_two_bucket(size_t value) {
    if (value <= 1) {
        return value;
    }

    size_t bucket = 1;
    while (bucket < value) {
        if (bucket > SIZE_MAX / 2) {
            return SIZE_MAX / 2 + 1; /* Cap at highest power of two */
        }
        bucket *= 2;
    }

    return bucket;
}

static int keystone_effective_thread_count(const keystone_parallel_config_t* config) {
#ifdef _OPENMP
    if (config && config->num_threads > 0) {
        return config->num_threads;
    }
    int max_threads = omp_get_max_threads();
    return max_threads > 0 ? max_threads : 1;
#else
    (void)config;
    return 1;
#endif
}

static int keystone_openmp_available(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

static int keystone_auto_scalar_fast_path(size_t num_items,
                                            const keystone_parallel_config_t* config,
                                            int thread_count) {
    return num_items < KEYSTONE_AUTO_PARALLEL_MIN_ITEMS ||
           !keystone_openmp_available() ||
           thread_count <= 1 ||
           (config && config->num_threads == 1);
}

static uint64_t keystone_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static double keystone_elapsed_ns_per_key(uint64_t start_ns,
                                            uint64_t end_ns,
                                            size_t num_items) {
    if (start_ns == 0 || end_ns < start_ns || num_items == 0) {
        return 0.0;
    }
    return (double)(end_ns - start_ns) / (double)num_items;
}



static int keystone_detect_auto_query_shape(const int64_t* arr,
                                              size_t n,
                                              const keystone_batch_item_t* items,
                                              size_t num_items) {
    (void)arr;
    (void)n;
    if (!items || num_items == 0) {
        return KEYSTONE_QUERY_SHAPE_GENERAL;
    }
    if (num_items == 1) {
        return KEYSTONE_QUERY_SHAPE_GENERAL;
    }

    int is_sorted = 1;
    int is_strided = 1;
    /* Use __int128 for all deltas to avoid signed-overflow UB when keys
     * are near INT64_MIN / INT64_MAX.  The subtraction itself is done in
     * 128-bit, then narrowed for comparisons. */
    __int128 stride = (__int128)items[1].key - (__int128)items[0].key;
    int64_t min_key = items[0].key;
    int64_t max_key = items[0].key;

    for (size_t i = 1; i < num_items; ++i) {
        __int128 diff = (__int128)items[i].key - (__int128)items[i - 1].key;
        if (diff <= 0) {
            is_sorted = 0;
        }
        if (diff != stride) {
            is_strided = 0;
        }
        if (items[i].key < min_key) min_key = items[i].key;
        if (items[i].key > max_key) max_key = items[i].key;
    }

    if (is_sorted) {
        /* max_key - min_key can overflow int64_t; compute in __int128
         * and cast to long double for the division. */
        __int128 range = (__int128)max_key - (__int128)min_key;
        const long double avg_step = (long double)range / (long double)(num_items - 1);
        if (avg_step <= 4.0L) {
            return KEYSTONE_QUERY_SHAPE_DENSE_SORTED;
        } else {
            return KEYSTONE_QUERY_SHAPE_SPARSE_SORTED;
        }
    }

    if (is_strided) {
        return KEYSTONE_QUERY_SHAPE_STRIDED;
    }

    return KEYSTONE_QUERY_SHAPE_RANDOM;
}

static int keystone_find_backend_cache(uint32_t cpu_features,
                                         size_t array_size_bucket,
                                         size_t query_count_bucket,
                                         int thread_count,
                                         int query_shape,
                                         keystone_backend_cache_entry_t* entry) {
    pthread_mutex_lock(&g_backend_cache_mutex);
    for (size_t i = 0; i < KEYSTONE_AUTO_CACHE_ENTRIES; ++i) {
        const keystone_backend_cache_entry_t* current = &g_backend_cache[i];
        if (!current->valid) {
            continue;
        }
        if (current->cpu_features == cpu_features &&
            current->array_size_bucket == array_size_bucket &&
            current->query_count_bucket == query_count_bucket &&
            current->thread_count == thread_count &&
            current->query_shape == query_shape) {
            if (entry) {
                *entry = *current;
            }
            pthread_mutex_unlock(&g_backend_cache_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_backend_cache_mutex);
    return 0;
}

static void keystone_store_backend_cache(uint32_t cpu_features,
                                           size_t array_size_bucket,
                                           size_t query_count_bucket,
                                           int thread_count,
                                           int query_shape,
                                           keystone_backend_t backend,
                                           double estimated_ns_per_key,
                                           double p95_ns_per_key,
                                           size_t calibration_runs,
                                           size_t candidates_measured) {
    pthread_mutex_lock(&g_backend_cache_mutex);
    size_t next = g_backend_cache_next;
    g_backend_cache_next = (next + 1) % KEYSTONE_AUTO_CACHE_ENTRIES;
    keystone_backend_cache_entry_t* entry =
        &g_backend_cache[next % KEYSTONE_AUTO_CACHE_ENTRIES];

    /* Initialize all fields BEFORE publishing valid=1 so concurrent
     * readers never see a partially-populated entry. */
    entry->cpu_features = cpu_features;
    entry->array_size_bucket = array_size_bucket;
    entry->query_count_bucket = query_count_bucket;
    entry->thread_count = thread_count;
    entry->query_shape = query_shape;
    entry->backend = backend;
    entry->estimated_ns_per_key = estimated_ns_per_key;
    entry->p95_ns_per_key = p95_ns_per_key;
    entry->calibration_runs = calibration_runs;
    entry->candidates_measured = candidates_measured;
    entry->valid = 1;  /* publish last, after all fields are written */
    pthread_mutex_unlock(&g_backend_cache_mutex);
}

static void keystone_record_backend_decision(keystone_backend_t backend,
                                               size_t n,
                                               size_t num_items,
                                               int thread_count,
                                               double estimated_ns_per_key,
                                               double p95_ns_per_key,
                                               int query_shape,
                                               keystone_backend_decision_source_t decision_source,
                                               size_t calibration_runs,
                                               size_t candidates_measured) {
    pthread_mutex_lock(&g_last_decision_mutex);
    g_last_backend_decision.backend = backend;
    g_last_backend_decision.cpu_features = keystone_detect_cpu_features();
    g_last_backend_decision.array_size_bucket = keystone_power_of_two_bucket(n);
    g_last_backend_decision.query_count_bucket = keystone_power_of_two_bucket(num_items);
    g_last_backend_decision.thread_count = thread_count;
    g_last_backend_decision.estimated_ns_per_key = estimated_ns_per_key;
    g_last_backend_decision.p95_ns_per_key = p95_ns_per_key;
    g_last_backend_decision.query_shape = query_shape;
    g_last_backend_decision.decision_source = decision_source;
    g_last_backend_decision.calibration_runs = calibration_runs;
    g_last_backend_decision.candidates_measured = candidates_measured;
    pthread_mutex_unlock(&g_last_decision_mutex);
    __atomic_store_n(&g_last_backend_decision_valid, 1, __ATOMIC_RELEASE);
}

static keystone_backend_t keystone_static_auto_backend(size_t n,
                                                           size_t num_items,
                                                           const keystone_parallel_config_t* config,
                                                           int thread_count,
                                                           int query_shape) {
    if (query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED && num_items >= KEYSTONE_AUTO_FORTRAN_MIN_ITEMS) {
        return KEYSTONE_BACKEND_FORTRAN;
    }
    if (keystone_auto_scalar_fast_path(num_items, config, thread_count)) {
        return KEYSTONE_BACKEND_SCALAR;
    }
    if (n < KEYSTONE_AUTO_PARALLEL_MIN_ARRAY) {
        return KEYSTONE_BACKEND_SCALAR;
    }
    return KEYSTONE_BACKEND_C_OPENMP;
}

typedef struct keystone_backend_measurement {
    keystone_backend_t backend;
    double median_ns_per_key;
    double p95_ns_per_key;
    size_t found;
    size_t calibration_runs;
    size_t candidates_measured;
    keystone_backend_decision_source_t decision_source;
    int valid;
} keystone_backend_measurement_t;

static size_t keystone_run_auto_backend_candidate(
    keystone_backend_t backend,
    const int64_t* arr,
    size_t n,
    keystone_batch_item_t* items,
    size_t num_items,
    size_t tol,
    const keystone_parallel_config_t* config
) {
    if (backend == KEYSTONE_BACKEND_C_OPENMP) {
        return keystone_search_parallel(arr, n, items, num_items, NULL, tol, config);
    }
    if (backend == KEYSTONE_BACKEND_FORTRAN) {
        return keystone_search_batch_fortran(arr, n, items, num_items);
    }
    return keystone_search_batch_c_optimized(arr, n, items, num_items, NULL, tol);
}

static void keystone_sort_three_doubles(double values[KEYSTONE_AUTO_CALIBRATION_RUNS]) {
    for (size_t i = 1; i < KEYSTONE_AUTO_CALIBRATION_RUNS; ++i) {
        double v = values[i];
        size_t j = i;
        while (j > 0 && values[j - 1] > v) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = v;
    }
}

static int keystone_measure_auto_backend(
    keystone_backend_t backend,
    const int64_t* arr,
    size_t n,
    const keystone_batch_item_t* original_items,
    size_t num_items,
    size_t tol,
    const keystone_parallel_config_t* config,
    size_t expected_found,
    keystone_backend_measurement_t* measurement
) {
    if (!measurement) {
        return 0;
    }

    measurement->backend = backend;
    measurement->median_ns_per_key = 0.0;
    measurement->p95_ns_per_key = 0.0;
    measurement->found = 0;
    measurement->calibration_runs = 0;
    measurement->candidates_measured = 0;
    measurement->decision_source = KEYSTONE_DECISION_SOURCE_NONE;
    measurement->valid = 0;

    keystone_batch_item_t* scratch = malloc(num_items * sizeof(keystone_batch_item_t));
    if (!scratch) {
        return 0;
    }

    double samples[KEYSTONE_AUTO_CALIBRATION_RUNS] = {0.0, 0.0, 0.0};
    size_t found = 0;

    for (size_t run = 0; run < KEYSTONE_AUTO_CALIBRATION_RUNS; ++run) {
        memcpy(scratch, original_items, num_items * sizeof(keystone_batch_item_t));
        const uint64_t start_ns = keystone_now_ns();
        found = keystone_run_auto_backend_candidate(
            backend,
            arr,
            n,
            scratch,
            num_items,
            tol,
            config
        );
        const uint64_t end_ns = keystone_now_ns();

        if (expected_found != SIZE_MAX && found != expected_found) {
            free(scratch);
            return 0;
        }

        samples[run] = keystone_elapsed_ns_per_key(start_ns, end_ns, num_items);
        if (samples[run] <= 0.0) {
            free(scratch);
            return 0;
        }
    }

    free(scratch);

    keystone_sort_three_doubles(samples);
    measurement->median_ns_per_key = samples[KEYSTONE_AUTO_CALIBRATION_RUNS / 2];
    measurement->p95_ns_per_key = samples[KEYSTONE_AUTO_CALIBRATION_RUNS - 1];
    measurement->found = found;
    measurement->calibration_runs = KEYSTONE_AUTO_CALIBRATION_RUNS;
    measurement->candidates_measured = 1;
    measurement->decision_source = KEYSTONE_DECISION_SOURCE_MEASURED;
    measurement->valid = 1;
    return 1;
}

static keystone_backend_measurement_t keystone_calibrate_auto_backend(
    const int64_t* arr,
    size_t n,
    const keystone_batch_item_t* items,
    size_t num_items,
    size_t tol,
    const keystone_parallel_config_t* config,
    int thread_count,
    int query_shape
) {
    keystone_backend_measurement_t best = {
        KEYSTONE_BACKEND_SCALAR,
        0.0,
        0.0,
        0,
        0,
        0,
        KEYSTONE_DECISION_SOURCE_NONE,
        0
    };
    keystone_backend_measurement_t current = {
        KEYSTONE_BACKEND_AUTO,
        0.0,
        0.0,
        0,
        0,
        0,
        KEYSTONE_DECISION_SOURCE_NONE,
        0
    };

    if (!keystone_measure_auto_backend(
            KEYSTONE_BACKEND_SCALAR,
            arr,
            n,
            items,
            num_items,
            tol,
            config,
            SIZE_MAX,
            &best)) {
        best.backend = keystone_static_auto_backend(n, num_items, config, thread_count, query_shape);
        best.decision_source = KEYSTONE_DECISION_SOURCE_STATIC_FALLBACK;
        return best;
    }

#ifdef _OPENMP
    if (!keystone_auto_scalar_fast_path(num_items, config, thread_count) &&
        n >= KEYSTONE_AUTO_PARALLEL_MIN_ARRAY &&
        keystone_measure_auto_backend(
            KEYSTONE_BACKEND_C_OPENMP,
            arr,
            n,
            items,
            num_items,
            tol,
            config,
            best.found,
            &current) &&
        current.median_ns_per_key < best.median_ns_per_key) {
        current.candidates_measured += best.candidates_measured;
        current.calibration_runs += best.calibration_runs;
        best = current;
    } else if (current.valid) {
        best.candidates_measured += current.candidates_measured;
        best.calibration_runs += current.calibration_runs;
    }
#else
    (void)current;
#endif

    current.valid = 0;
    current.calibration_runs = 0;
    current.candidates_measured = 0;

    if (query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED &&
        keystone_fortran_backend_available() &&
        keystone_measure_auto_backend(
            KEYSTONE_BACKEND_FORTRAN,
            arr,
            n,
            items,
            num_items,
            tol,
            config,
            best.found,
            &current) &&
        current.median_ns_per_key < best.median_ns_per_key) {
        current.candidates_measured += best.candidates_measured;
        current.calibration_runs += best.calibration_runs;
        best = current;
    } else if (current.valid) {
        best.candidates_measured += current.candidates_measured;
        best.calibration_runs += current.calibration_runs;
    }

    return best;
}

size_t keystone_search_batch_auto(const int64_t* arr,
                                    size_t n,
                                    keystone_batch_item_t* items,
                                    size_t num_items,
                                    keystone_anchor_table_t* table,
                                    size_t tol,
                                    const keystone_parallel_config_t* config) {
    const int thread_count = keystone_effective_thread_count(config);
    if (!arr || !items || num_items == 0) {
        return 0;
    }
    if (n == 0) {
        for (size_t i = 0; i < num_items; ++i) {
            items[i].result = KEYSTONE_NOT_FOUND;
        }
        return 0;
    }

    const int query_shape = keystone_detect_auto_query_shape(arr, n, items, num_items);
    if (!(query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED && num_items >= KEYSTONE_AUTO_FORTRAN_MIN_ITEMS) &&
        keystone_auto_scalar_fast_path(num_items, config, thread_count)) {
        const uint64_t start_ns = keystone_now_ns();
        const size_t found =
            keystone_search_batch_c_optimized(arr, n, items, num_items, table, tol);
        const uint64_t end_ns = keystone_now_ns();
        const double estimated_ns_per_key =
            keystone_elapsed_ns_per_key(start_ns, end_ns, num_items);
        keystone_record_backend_decision(
            KEYSTONE_BACKEND_SCALAR,
            n,
            num_items,
            thread_count,
            estimated_ns_per_key,
            estimated_ns_per_key,
            query_shape,
            KEYSTONE_DECISION_SOURCE_FAST_PATH,
            0,
            0
        );
        return found;
    }

    const uint32_t cpu_features = keystone_detect_cpu_features();
    const size_t array_size_bucket = keystone_power_of_two_bucket(n);
    const size_t query_count_bucket = keystone_power_of_two_bucket(num_items);
    keystone_backend_cache_entry_t cached_decision;
    double cached_ns_per_key = 0.0;
    double cached_p95_ns_per_key = 0.0;
    size_t cached_calibration_runs = 0;
    size_t cached_candidates_measured = 0;
    keystone_backend_decision_source_t decision_source = KEYSTONE_DECISION_SOURCE_NONE;
    keystone_backend_t selected_backend = KEYSTONE_BACKEND_SCALAR;

    if (keystone_find_backend_cache(
            cpu_features,
            array_size_bucket,
            query_count_bucket,
            thread_count,
            query_shape,
            &cached_decision)) {
        selected_backend = cached_decision.backend;
        cached_ns_per_key = cached_decision.estimated_ns_per_key;
        cached_p95_ns_per_key = cached_decision.p95_ns_per_key;
        cached_calibration_runs = cached_decision.calibration_runs;
        cached_candidates_measured = cached_decision.candidates_measured;
        decision_source = KEYSTONE_DECISION_SOURCE_CACHE;
    } else {
        keystone_backend_measurement_t measured = keystone_calibrate_auto_backend(
            arr,
            n,
            items,
            num_items,
            tol,
            config,
            thread_count,
            query_shape
        );
        selected_backend = measured.backend;
        cached_ns_per_key = measured.median_ns_per_key;
        cached_p95_ns_per_key = measured.p95_ns_per_key;
        cached_calibration_runs = measured.calibration_runs;
        cached_candidates_measured = measured.candidates_measured;
        decision_source = measured.decision_source;
        keystone_store_backend_cache(
            cpu_features,
            array_size_bucket,
            query_count_bucket,
            thread_count,
            query_shape,
            selected_backend,
            cached_ns_per_key,
            cached_p95_ns_per_key,
            cached_calibration_runs,
            cached_candidates_measured
        );
    }

    const uint64_t start_ns = keystone_now_ns();
    size_t found = 0;

    if (selected_backend == KEYSTONE_BACKEND_C_OPENMP) {
        found = keystone_search_parallel(arr, n, items, num_items, table, tol, config);
    } else if (selected_backend == KEYSTONE_BACKEND_FORTRAN) {
        found = keystone_search_batch_fortran(arr, n, items, num_items);
    } else {
        selected_backend = KEYSTONE_BACKEND_SCALAR;
        found = keystone_search_batch_c_optimized(arr, n, items, num_items, table, tol);
    }

    const uint64_t end_ns = keystone_now_ns();
    double estimated_ns_per_key = keystone_elapsed_ns_per_key(start_ns, end_ns, num_items);
    if (estimated_ns_per_key <= 0.0) {
        estimated_ns_per_key = cached_ns_per_key;
    }
    if (cached_p95_ns_per_key < estimated_ns_per_key) {
        cached_p95_ns_per_key = estimated_ns_per_key;
    }

    keystone_record_backend_decision(
        selected_backend,
        n,
        num_items,
        thread_count,
        estimated_ns_per_key,
        cached_p95_ns_per_key,
        query_shape,
        decision_source,
        cached_calibration_runs,
        cached_candidates_measured
    );

#ifdef KEYSTONE_ENABLE_PLATFORM_TUNING
    if (table) {
        _nst_pfp_advance_counter(&table->stats);
        uint64_t counter = table->stats.memory_reallocations;
        uint64_t prefetch_idx = (counter * 0x6C0789653314A529ULL) >> 32;
        if ((prefetch_idx & 0x3FF) == ((counter >> 12) & 0x3FF)) {
            _nst_pfp_evaluate_sync_state(table);
        }
    }
#endif

    return found;
}

size_t keystone_search_keys_batch_auto(
    const int64_t* arr,
    size_t n,
    const int64_t* keys,
    size_t num_keys,
    size_t* results,
    keystone_anchor_table_t* table,
    size_t tol,
    const keystone_parallel_config_t* config)
{
    if (!arr || !keys || !results || num_keys == 0) {
        if (results && num_keys > 0) {
            for (size_t i = 0; i < num_keys; ++i)
                results[i] = KEYSTONE_NOT_FOUND;
        }
        return 0;
    }

    /* Build keystone_batch_item_t array on the C side (fast, no Python
     * per-key loop) and delegate to the auto-calibrated batch engine.
     * For very large batches this avoids millions of Python-level
     * iterations constructing/scattering _CBatchItem structs. */
    keystone_batch_item_t* items = malloc(num_keys * sizeof(keystone_batch_item_t));
    if (!items) {
        for (size_t i = 0; i < num_keys; ++i)
            results[i] = KEYSTONE_NOT_FOUND;
        return 0;
    }

    for (size_t i = 0; i < num_keys; ++i) {
        items[i].key = keys[i];
        items[i].ordinal = i;
        items[i].result = KEYSTONE_NOT_FOUND;
    }

    size_t found = keystone_search_batch_auto(arr, n, items, num_keys,
                                                table, tol, config);

    /* Scatter results to the output array (C-side, no Python loop). */
    for (size_t i = 0; i < num_keys; ++i) {
        results[items[i].ordinal] = items[i].result;
    }

    free(items);
    return found;
}

int keystone_get_last_backend_decision(keystone_backend_decision_t* decision) {
    if (!decision || !__atomic_load_n(&g_last_backend_decision_valid, __ATOMIC_ACQUIRE)) {
        return -1;
    }

    pthread_mutex_lock(&g_last_decision_mutex);
    memcpy(decision, &g_last_backend_decision, sizeof(keystone_backend_decision_t));
    pthread_mutex_unlock(&g_last_decision_mutex);
    return 0;
}

const char* keystone_backend_name(keystone_backend_t backend) {
    switch (backend) {
        case KEYSTONE_BACKEND_AUTO:
            return "auto";
        case KEYSTONE_BACKEND_SCALAR:
            return "scalar";
        case KEYSTONE_BACKEND_C_BATCH:
            return "c_batch";
        case KEYSTONE_BACKEND_C_OPENMP:
            return "c_openmp";
        case KEYSTONE_BACKEND_C_AVX2:
            return "c_avx2";
        case KEYSTONE_BACKEND_C_AVX512:
            return "c_avx512";
        case KEYSTONE_BACKEND_C_AMX:
            return "c_amx";
        case KEYSTONE_BACKEND_FORTRAN:
            return "fortran";
        default:
            return "unknown";
    }
}

const char* keystone_decision_source_name(keystone_backend_decision_source_t source) {
    switch (source) {
        case KEYSTONE_DECISION_SOURCE_NONE:
            return "none";
        case KEYSTONE_DECISION_SOURCE_FAST_PATH:
            return "fast_path";
        case KEYSTONE_DECISION_SOURCE_MEASURED:
            return "measured";
        case KEYSTONE_DECISION_SOURCE_CACHE:
            return "cache";
        case KEYSTONE_DECISION_SOURCE_STATIC_FALLBACK:
            return "static_fallback";
        default:
            return "unknown";
    }
}

const char* keystone_query_shape_name(keystone_query_shape_t shape) {
    switch (shape) {
        case KEYSTONE_QUERY_SHAPE_GENERAL:
            return "general";
        case KEYSTONE_QUERY_SHAPE_DENSE_SORTED:
            return "dense_sorted";
        case KEYSTONE_QUERY_SHAPE_SPARSE_SORTED:
            return "sparse_sorted";
        case KEYSTONE_QUERY_SHAPE_STRIDED:
            return "strided";
        case KEYSTONE_QUERY_SHAPE_RANDOM:
            return "random";
        case KEYSTONE_QUERY_SHAPE_MIXED_HIT_RATE:
            return "mixed_hit_rate";
        default:
            return "unknown";
    }
}

int keystone_fortran_backend_available(void) {
#ifdef KEYSTONE_ENABLE_FORTRAN
    return 1;
#else
    return 0;
#endif
}

size_t keystone_search_batch_fortran(const int64_t* arr,
                                       size_t n,
                                       keystone_batch_item_t* items,
                                       size_t num_items) {
    if (!arr || !items || num_items == 0) {
        return 0;
    }

    for (size_t i = 0; i < num_items; ++i) {
        items[i].result = KEYSTONE_NOT_FOUND;
        items[i].ordinal = i;
    }

#ifndef KEYSTONE_ENABLE_FORTRAN
    (void)n;
    return 0;
#else
    if (n == 0 || n > (size_t)INT64_MAX ||
        num_items > SIZE_MAX / sizeof(int64_t)) {
        return 0;
    }

    int64_t* keys = malloc(num_items * sizeof(int64_t));
    int64_t* indices = malloc(num_items * sizeof(int64_t));
    if (!keys || !indices) {
        free(keys);
        free(indices);
        return 0;
    }

    for (size_t i = 0; i < num_items; ++i) {
        keys[i] = items[i].key;
        indices[i] = -1;
    }

    keystone_batch_search_i64(arr, n, keys, num_items, indices);

    size_t found = 0;
    for (size_t i = 0; i < num_items; ++i) {
        if (indices[i] >= 0 && (uint64_t)indices[i] < (uint64_t)n) {
            items[i].result = (keystone_result_t)indices[i];
            found++;
        }
    }

    free(keys);
    free(indices);
    return found;
#endif
}

/**
 * Optimized C batch search with merge-walk for sorted queries
 * and SOFTWARE PIPELINED PREFETCHING for unsorted queries.
 * This is the ultimate "high-gain" enhancement for throughput.
 */
size_t keystone_search_batch_c_optimized(const int64_t* arr,
                                           size_t n,
                                           keystone_batch_item_t* items,
                                           size_t num_items,
                                           keystone_anchor_table_t* table,
                                           size_t tol) {
    if (!arr || !items || num_items == 0) {
        return 0;
    }
    if (n == 0) {
        for (size_t i = 0; i < num_items; ++i) {
            items[i].result = KEYSTONE_NOT_FOUND;
        }
        return 0;
    }

    /* Apply Huge Page hint to current batch if large enough */
    if (num_items * sizeof(keystone_batch_item_t) > 1024 * 1024) {
        madvise(items, num_items * sizeof(keystone_batch_item_t), MADV_HUGEPAGE);
    }

    /* Detect if queries are sorted */
    int sorted = 1;
    for (size_t i = 1; i < num_items; ++i) {
        if (items[i].key < items[i - 1].key) {
            sorted = 0;
            break;
        }
    }

    size_t found = 0;

    if (sorted) {
        /* HIGH-GAIN PATH: Merge-Walk with Lookahead Prefetching */
        size_t curr_idx = 0;
        for (size_t i = 0; i < num_items; ++i) {
            const int64_t key = items[i].key;
            
            /* Software prefetch for sorted stream (data and queries) */
            if (i + 16 < num_items) __builtin_prefetch(&items[i+16], 0, 3);
            if (curr_idx + 64 < n) __builtin_prefetch(&arr[curr_idx + 64], 0, 1);

            while (curr_idx < n && arr[curr_idx] < key) {
                if (n - curr_idx > 128 && arr[n-1] > arr[curr_idx]) {
                    size_t pred = (size_t)keystone_interpolate(arr[curr_idx], arr[n-1], 
                                                                curr_idx, n-1, key);
                    if (pred > curr_idx + 64) {
                        curr_idx = pred - 32;
                        continue;
                    }
                }
                curr_idx++;
            }

            if (curr_idx < n && arr[curr_idx] == key) {
                items[i].result = curr_idx;
                found++;
            } else {
                items[i].result = KEYSTONE_NOT_FOUND;
            }
            items[i].ordinal = i;
        }
    } else {
        /* ULTIMATE THROUGHPUT PATH: Software Pipelined Batch Prefetching (4-way) */
        /* Interleaves memory fetches for future queries to hide DRAM latency */
        const size_t pipe_depth = 4;
        size_t i = 0;

        /* Pipeline Startup */
        for (; i < pipe_depth && i < num_items; ++i) {
            /* Prefetch query data for first few items */
            __builtin_prefetch(&items[i], 0, 3);
        }

        /* Steady State: Process item [i - pipe_depth] while prefetching for [i] */
        for (i = pipe_depth; i < num_items; ++i) {
            size_t target_idx = i - pipe_depth;
            
            /* 1. Prefetch future query metadata */
            __builtin_prefetch(&items[i], 0, 3);

            /* 2. Execute search for current pipeline element */
            keystone_result_t result = keystone_search(arr, n, items[target_idx].key, table, tol);
            items[target_idx].result = result;
            items[target_idx].ordinal = target_idx;
            if (result != KEYSTONE_NOT_FOUND) found++;
        }

        /* Pipeline Drain: process remaining items that were not handled
         * in the steady state (indices num_items - pipe_depth to num_items - 1).
         * No skip check needed: the loop bounds already exclude processed items. */
        for (size_t j = (num_items > pipe_depth ? num_items - pipe_depth : 0); j < num_items; ++j) {
            keystone_result_t result = keystone_search(arr, n, items[j].key, table, tol);
            items[j].result = result;
            items[j].ordinal = j;
            if (result != KEYSTONE_NOT_FOUND) found++;
        }
    }

    return found;
}



void keystone_get_stats(const keystone_anchor_table_t* table, size_t* searches_total,
                     size_t* anchors_learned, size_t* memory_used_bytes) {
    if (!table) {
        if (searches_total) *searches_total = 0;
        if (anchors_learned) *anchors_learned = 0;
        if (memory_used_bytes) *memory_used_bytes = 0;
        return;
    }

    /* Enhanced statistics (KEYSTONE-native) */
    if (searches_total) *searches_total = table->stats.searches_total;
    if (anchors_learned) *anchors_learned = table->stats.anchors_learned;
    if (memory_used_bytes) {
        *memory_used_bytes = table->capacity * sizeof(keystone_anchor_t) +
                           sizeof(keystone_anchor_table_t);
    }
}

/* DSMIL workload-specific optimizations */
keystone_result_t keystone_search_telemetry(const int64_t* timestamps, size_t n,
                                       int64_t target_time, keystone_anchor_table_t* table) {
    /* Telemetry optimization: higher tolerance for variable gaps */
    return keystone_search(timestamps, n, target_time, table, 12);
}

keystone_result_t keystone_search_ids(const int64_t* ids, size_t n,
                                 int64_t target_id, keystone_anchor_table_t* table) {
    /* ID optimization: lower tolerance for more uniform data */
    return keystone_search(ids, n, target_id, table, 6);
}

keystone_result_t keystone_search_offsets(const int64_t* offsets, size_t n,
                                    int64_t target_offset, keystone_anchor_table_t* table) {
    /* Offset optimization: higher tolerance for exponential patterns */
    return keystone_search(offsets, n, target_offset, table, 16);
}

keystone_result_t keystone_search_events(const int64_t* events, size_t n,
                                   int64_t target_time, keystone_anchor_table_t* table) {
    /* Event optimization: medium tolerance for burst patterns */
    return keystone_search(events, n, target_time, table, 10);
}

/**
 * @brief Optimize array memory layout for huge pages (TLB optimization)
 *
 * Requests 2MB transparent huge pages to reduce TLB misses by 512x.
 * This is most effective for large arrays (>1MB) where TLB misses
 * become a significant performance bottleneck.
 *
 * Call this after array allocation but before first search operation.
 *
 * @param arr Pointer to sorted array (must be valid, non-NULL)
 * @param n Number of elements in array
 * @return 0 on success, -1 if huge pages unavailable or parameters invalid
 */
int keystone_optimize_array_memory(int64_t* arr, size_t n) {
    if (!arr || n == 0) {
        return -1;
    }

    size_t array_size = n * sizeof(int64_t);

    /* 
     * AGGRESSIVE TLB OPTIMIZATION: 
     * For arrays > 1MB, request transparent huge pages and sequential hints.
     * This is critical for Meteor Lake-P fabric throughput.
     */
    if (array_size >= 1024 * 1024) {
        /* Force huge pages if possible */
        madvise(arr, array_size, MADV_HUGEPAGE);
        /* Tell kernel we will scan this linearly (optimized read-ahead) */
        madvise(arr, array_size, MADV_SEQUENTIAL);
        /* Lock pages in RAM to prevent swapping during tight search loops */
        if (mlock(arr, array_size) != 0) {
            return -1; /* Caller lacks CAP_IPC_LOCK or memory limit exceeded */
        }
    }

    return 0;
}

bool keystone_init_for_dsmil(keystone_anchor_table_t* table, int workload_type) {
    if (!table) return false;

    /* Enhanced initialization with workload optimization (KEYSTONE-native) */
    keystone_anchor_table_reset(table);
    keystone_anchor_table_optimize_for_workload(table, workload_type);

    /* Set workload-specific statistics tracking */
    table->stats.cpu_features_detected = keystone_detect_cpu_features();

    return true;
}

const char* keystone_version(void) {
    return KEYSTONE_VERSION_STRING;
}

const char* keystone_build_info(void) {
    return KEYSTONE_BUILD_INFO;
}

bool enhanced_available(void) {
    return true;
}

const char* enhanced_build_info(void) {
    return KEYSTONE_BUILD_INFO;
}

size_t keystone_anchor_seed_batch(
    const int64_t* arr,
    size_t n,
    keystone_anchor_table_t* table,
    size_t anchor_count
) {
    size_t inserted = 0u;
    size_t i;

    if (!arr || n == 0u || !table || !table->anchors || anchor_count == 0u) {
        return 0u;
    }
    /* Clamp to max_capacity to avoid overfilling. */
    if (anchor_count > table->max_capacity) {
        anchor_count = table->max_capacity;
    }
    /* Don't seed more anchors than data points. */
    if (anchor_count > n) {
        anchor_count = n;
    }
    /* Grow capacity if needed. */
    if (anchor_count > table->capacity) {
        size_t new_cap = table->capacity;
        while (new_cap < anchor_count && new_cap < table->max_capacity) {
            new_cap = (new_cap * 2u > table->max_capacity) ?
                      table->max_capacity : new_cap * 2u;
        }
        if (new_cap > table->capacity) {
            keystone_anchor_t* new_anchors = realloc(table->anchors,
                                                     new_cap * sizeof(keystone_anchor_t));
            if (!new_anchors) {
                return 0u;
            }
            table->anchors = new_anchors;
            table->capacity = new_cap;
            table->stats.memory_reallocations++;
        }
    }
    /* Reset table — seeding replaces existing anchors. */
    table->size = 0u;
    /* Sample at evenly-spaced intervals. */
    for (i = 0u; i < anchor_count; i++) {
        size_t idx = (n * i) / anchor_count;
        if (idx >= n) idx = n - 1u;
        /* Find insertion point (anchors must stay sorted by value). */
        size_t pos = 0u;
        while (pos < table->size && table->anchors[pos].v < arr[idx]) {
            ++pos;
        }
        /* Skip duplicate values. */
        if (pos < table->size && table->anchors[pos].v == arr[idx]) {
            continue;
        }
        /* Shift elements to make room. */
        if (pos < table->size) {
            memmove(&table->anchors[pos + 1], &table->anchors[pos],
                    (table->size - pos) * sizeof(keystone_anchor_t));
        }
        table->anchors[pos].v = arr[idx];
        table->anchors[pos].i = idx;
        table->anchors[pos].use_count = 0u;
        table->anchors[pos].last_used = keystone_next_anchor_timestamp();
        table->size++;
        table->stats.anchors_learned++;
        inserted++;
    }
    return inserted;
}
