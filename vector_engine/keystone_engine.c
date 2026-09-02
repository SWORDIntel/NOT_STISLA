/*
 * keystone_engine.c - Main vector engine implementation
 *
 * Hardware-agnostic vector store with LSH coarse index and SIMD
 * exact rerank. Dispatches to the best available backend at runtime
 * with compile-time guards for each SIMD level.
 */
#define _POSIX_C_SOURCE 200112L
#include "keystone_vector_engine.h"
#include "kernel_dispatch.h"
#include "lsh.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* CUDA soft-load support */
#ifdef __unix__
#include <dlfcn.h>
#endif

/* ---------------------------------------------------------------
 * Engine structure (opaque to callers)
 * --------------------------------------------------------------- */
struct keystone_vec_engine {
    keystone_config_t cfg;

    /* Vector storage: contiguous aligned array */
    float    *vectors;          /* capacity * dim, 32-byte aligned */
    uint64_t *ids;              /* capacity */
    uint32_t  count;            /* current number of stored vectors */
    uint32_t  capacity;
    uint32_t  dim;

    /* ID lookup: sorted array of (id, index) for fast retrieval */
    uint64_t *sorted_ids;       /* sorted copy of ids */
    uint32_t *sorted_indices;   /* corresponding indices into vectors[] */
    int       ids_sorted;

    /* LSH coarse index */
    keystone_lsh_index_t *lsh;
    int      lsh_finalized;

    /* Selected backend */
    const keystone_kernels_t *kernels;
    keystone_backend_t backend;

    /* CUDA soft-load state */
    void *cuda_handle;
    int (*cuda_init)(void);
    int (*cuda_batch_dist)(const float *, const float *, uint32_t, uint32_t, float *);
    void (*cuda_cleanup)(void);
    int (*cuda_available)(void);
    int cuda_initialized;

#ifdef KEYSTONE_HAVE_VPU
    int vpu_connected;
    char vpu_socket[108];
#endif
};

/* ---------------------------------------------------------------
 * CPU feature detection
 * --------------------------------------------------------------- */
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__)
#include <cpuid.h>
/* Use inline asm for xgetbv to avoid needing -mxsave compiler flag */
static inline unsigned long long ks_xgetbv(unsigned int index) {
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((unsigned long long)edx << 32) | eax;
}
#endif
#endif

uint32_t keystone_vec_detect_features(void) {
    uint32_t flags = 0;

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (edx & (1 << 20)) flags |= KEYSTONE_F_SSE42;
        if (ecx & (1 << 28)) {
            /* Check OSXSAVE */
            if (ecx & (1 << 27)) {
                unsigned long long xcr = ks_xgetbv(0);
                if ((xcr & 0x6) == 0x6) {
                    flags |= KEYSTONE_F_AVX;
                    /* Check AVX2 and AVX-512 via leaf 7 */
                    unsigned int eax7, ebx7, ecx7, edx7;
                    if (__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7)) {
                        if (ebx7 & (1 << 5)) flags |= KEYSTONE_F_AVX2;
                        if (ebx7 & (1 << 16)) {
                            if ((xcr & 0xE6) == 0xE6) {
                                flags |= KEYSTONE_F_AVX512F;
                            }
                        }
                    }
                }
            }
        }
    }
#endif /* __GNUC__ */
#elif defined(__arm__) || defined(__aarch64__)
    flags |= KEYSTONE_F_NEON; /* NEON is mandatory on aarch64, common on armv7+ */
#endif

    /* VPU detection: check if default socket exists */
#ifdef KEYSTONE_HAVE_VPU
    if (keystone_vec_vpu_available(NULL)) {
        flags |= KEYSTONE_F_VPU;
    }
#endif

    return flags;
}

const char *keystone_vec_backend_name(keystone_backend_t b) {
    switch (b) {
        case KEYSTONE_BACKEND_CUDA:   return "cuda";
        case KEYSTONE_BACKEND_AVX512: return "avx512";
        case KEYSTONE_BACKEND_AVX2:   return "avx2";
        case KEYSTONE_BACKEND_AVX:    return "avx";
        case KEYSTONE_BACKEND_SSE42:  return "sse42";
        case KEYSTONE_BACKEND_NEON:   return "neon";
        case KEYSTONE_BACKEND_VPU:    return "vpu";
        case KEYSTONE_BACKEND_SCALAR:
        default:                      return "scalar";
    }
}

/* ---------------------------------------------------------------
 * CUDA soft-load
 * --------------------------------------------------------------- */
static int cuda_try_init(keystone_vec_engine_t *e) {
#ifdef __unix__
    /* Try to dlopen the CUDA module */
    const char *libname = "libkeystone_cuda.so";
    void *h = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        /* Try alternate path */
        h = dlopen("./libkeystone_cuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!h) return 0; /* CUDA not available — not an error */

    /* Resolve symbols */
    e->cuda_init = (int (*)(void))dlsym(h, "keystone_cuda_init");
    e->cuda_batch_dist = (int (*)(const float *, const float *, uint32_t, uint32_t, float *))
                         dlsym(h, "keystone_cuda_batch_dist");
    e->cuda_cleanup = (void (*)(void))dlsym(h, "keystone_cuda_cleanup");
    e->cuda_available = (int (*)(void))dlsym(h, "keystone_cuda_available");

    if (!e->cuda_init || !e->cuda_batch_dist || !e->cuda_cleanup) {
        dlclose(h);
        h = NULL;
        return 0;
    }

    e->cuda_handle = h;

    /* Initialize CUDA */
    if (e->cuda_init() != 0) {
        dlclose(h);
        e->cuda_handle = NULL;
        e->cuda_init = NULL;
        e->cuda_batch_dist = NULL;
        e->cuda_cleanup = NULL;
        return 0;
    }

    e->cuda_initialized = 1;
    return 1;
#else
    (void)e;
    return 0;
#endif
}

static void cuda_teardown(keystone_vec_engine_t *e) {
#ifdef __unix__
    if (e->cuda_cleanup && e->cuda_initialized) {
        e->cuda_cleanup();
    }
    if (e->cuda_handle) {
        dlclose(e->cuda_handle);
        e->cuda_handle = NULL;
    }
    e->cuda_init = NULL;
    e->cuda_batch_dist = NULL;
    e->cuda_cleanup = NULL;
    e->cuda_initialized = 0;
#else
    (void)e;
#endif
}

/* ---------------------------------------------------------------
 * Backend selection
 * --------------------------------------------------------------- */
static keystone_error_t select_backend(keystone_vec_engine_t *e) {
    uint32_t f = keystone_vec_detect_features();
    int force = e->cfg.force_backend;
    (void)f; /* Used only in #if blocks below */

    /* Level 1: CUDA — soft-loaded, never required */
    if (force < 0 || force == KEYSTONE_BACKEND_CUDA) {
        if (cuda_try_init(e)) {
            e->backend = KEYSTONE_BACKEND_CUDA;
            e->kernels = NULL; /* CUDA uses its own path, not kernel table */
            return KEYSTONE_OK;
        }
        /* CUDA absent or init failed — silently continue */
    }

#ifdef KEYSTONE_HAVE_VPU
    /* Level 7: VPU — guarded, optional */
    if ((force < 0 || force == KEYSTONE_BACKEND_VPU) && e->cfg.vpu_enabled) {
        const char *sock = e->cfg.vpu_socket_path[0] ? e->cfg.vpu_socket_path : NULL;
        if (keystone_vpu_kernel_connect(sock) == 0) {
            e->backend = KEYSTONE_BACKEND_VPU;
            e->kernels = &keystone_kernels_vpu;
            e->vpu_connected = 1;
            if (e->kernels->set_metric) e->kernels->set_metric(e->cfg.metric);
            return KEYSTONE_OK;
        }
        /* VPU not available — silently continue to CPU */
    }
#endif

    /* Levels 2-6: CPU SIMD, highest compiled-and-present wins.
     * We use dlsym to resolve kernel tables that were compiled with
     * different ISA flags in separate translation units. This allows
     * the core engine (compiled without ISA flags) to dispatch to
     * SIMD kernels that were compiled with -mavx, -mavx2, etc. */
    const keystone_kernels_t *k = &keystone_kernels_scalar;
    keystone_backend_t b = KEYSTONE_BACKEND_SCALAR;

    if (force < 0) {
        /* Try each backend in priority order via dlsym */
#ifdef __unix__
        if (f & KEYSTONE_F_AVX512F) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, "keystone_kernels_avx512");
            if (pk) { k = pk; b = KEYSTONE_BACKEND_AVX512; }
        }
        if (b == KEYSTONE_BACKEND_SCALAR && (f & KEYSTONE_F_AVX2)) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, "keystone_kernels_avx2");
            if (pk) { k = pk; b = KEYSTONE_BACKEND_AVX2; }
        }
        if (b == KEYSTONE_BACKEND_SCALAR && (f & KEYSTONE_F_AVX)) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, "keystone_kernels_avx");
            if (pk) { k = pk; b = KEYSTONE_BACKEND_AVX; }
        }
        if (b == KEYSTONE_BACKEND_SCALAR && (f & KEYSTONE_F_SSE42)) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, "keystone_kernels_sse42");
            if (pk) { k = pk; b = KEYSTONE_BACKEND_SSE42; }
        }
        if (b == KEYSTONE_BACKEND_SCALAR && (f & KEYSTONE_F_NEON)) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, "keystone_kernels_neon");
            if (pk) { k = pk; b = KEYSTONE_BACKEND_NEON; }
        }
#else
        /* Non-Unix: use compile-time guards only */
#if defined(__AVX512F__)
        if (f & KEYSTONE_F_AVX512F) { k = &keystone_kernels_avx512; b = KEYSTONE_BACKEND_AVX512; }
        else
#endif
#if defined(__AVX2__)
        if (f & KEYSTONE_F_AVX2)    { k = &keystone_kernels_avx2;   b = KEYSTONE_BACKEND_AVX2; }
        else
#endif
#if defined(__AVX__)
        if (f & KEYSTONE_F_AVX)     { k = &keystone_kernels_avx;    b = KEYSTONE_BACKEND_AVX; }
        else
#endif
#if defined(__SSE4_2__)
        if (f & KEYSTONE_F_SSE42)   { k = &keystone_kernels_sse42;  b = KEYSTONE_BACKEND_SSE42; }
        else
#endif
#if defined(__ARM_NEON)
        if (f & KEYSTONE_F_NEON)    { k = &keystone_kernels_neon;   b = KEYSTONE_BACKEND_NEON; }
        else
#endif
        { /* scalar — always available */ }
#endif /* __unix__ */
    } else {
        /* Forced backend selection (for testing) */
#ifdef __unix__
        const char *sym_names[8] = {NULL};
        sym_names[KEYSTONE_BACKEND_SCALAR] = "keystone_kernels_scalar";
        sym_names[KEYSTONE_BACKEND_SSE42]  = "keystone_kernels_sse42";
        sym_names[KEYSTONE_BACKEND_AVX]    = "keystone_kernels_avx";
        sym_names[KEYSTONE_BACKEND_AVX2]   = "keystone_kernels_avx2";
        sym_names[KEYSTONE_BACKEND_AVX512] = "keystone_kernels_avx512";
        sym_names[KEYSTONE_BACKEND_NEON]   = "keystone_kernels_neon";
        if (force >= 0 && force <= KEYSTONE_BACKEND_AVX512 && force != KEYSTONE_BACKEND_VPU && force != KEYSTONE_BACKEND_CUDA && sym_names[force]) {
            const keystone_kernels_t *pk = (const keystone_kernels_t *)
                dlsym(RTLD_DEFAULT, sym_names[force]);
            if (pk) { k = pk; b = (keystone_backend_t)force; }
        }
#else
        switch (force) {
#if defined(__AVX512F__)
            case KEYSTONE_BACKEND_AVX512: k = &keystone_kernels_avx512; b = KEYSTONE_BACKEND_AVX512; break;
#endif
#if defined(__AVX2__)
            case KEYSTONE_BACKEND_AVX2:   k = &keystone_kernels_avx2;   b = KEYSTONE_BACKEND_AVX2; break;
#endif
#if defined(__AVX__)
            case KEYSTONE_BACKEND_AVX:    k = &keystone_kernels_avx;    b = KEYSTONE_BACKEND_AVX; break;
#endif
#if defined(__SSE4_2__)
            case KEYSTONE_BACKEND_SSE42:  k = &keystone_kernels_sse42;  b = KEYSTONE_BACKEND_SSE42; break;
#endif
#if defined(__ARM_NEON)
            case KEYSTONE_BACKEND_NEON:   k = &keystone_kernels_neon;   b = KEYSTONE_BACKEND_NEON; break;
#endif
            default: k = &keystone_kernels_scalar; b = KEYSTONE_BACKEND_SCALAR; break;
        }
#endif /* __unix__ */
    }

    e->kernels = k;
    e->backend = b;
    if (k->set_metric) k->set_metric(e->cfg.metric);
    return KEYSTONE_OK;
}

/* ---------------------------------------------------------------
 * Lifecycle
 * --------------------------------------------------------------- */
keystone_error_t keystone_vec_engine_create(const keystone_config_t *cfg,
                                             keystone_vec_engine_t **out) {
    if (!cfg || !out) return KEYSTONE_ERR_NULL;
    if (cfg->dim == 0) return KEYSTONE_ERR_DIM;
    if (cfg->capacity == 0) return KEYSTONE_ERR_PARAM;

    keystone_vec_engine_t *e = (keystone_vec_engine_t *)calloc(1, sizeof(*e));
    if (!e) return KEYSTONE_ERR_OOM;

    e->cfg = *cfg;
    e->dim = cfg->dim;
    e->capacity = cfg->capacity;

    /* Allocate aligned vector storage (32-byte for AVX) */
#if defined(_MSC_VER)
    e->vectors = (float *)_aligned_malloc((size_t)e->capacity * e->dim * sizeof(float), 32);
#else
    if (posix_memalign((void **)&e->vectors, 32, (size_t)e->capacity * e->dim * sizeof(float)) != 0) {
        e->vectors = NULL;
    }
#endif
    if (!e->vectors) { free(e); return KEYSTONE_ERR_OOM; }

    e->ids = (uint64_t *)malloc((size_t)e->capacity * sizeof(uint64_t));
    if (!e->ids) { free(e->vectors); free(e); return KEYSTONE_ERR_OOM; }

    e->sorted_ids = (uint64_t *)malloc((size_t)e->capacity * sizeof(uint64_t));
    e->sorted_indices = (uint32_t *)malloc((size_t)e->capacity * sizeof(uint32_t));
    if (!e->sorted_ids || !e->sorted_indices) {
        free(e->vectors); free(e->ids); free(e->sorted_ids); free(e->sorted_indices); free(e);
        return KEYSTONE_ERR_OOM;
    }

    e->count = 0;
    e->ids_sorted = 0;
    e->lsh_finalized = 0;

    /* Create LSH index */
    uint32_t num_tables = cfg->lsh_num_tables > 0 ? cfg->lsh_num_tables : 8;
    uint32_t hash_bits = cfg->lsh_hash_bits > 0 ? cfg->lsh_hash_bits : 12;
    uint32_t probes = cfg->lsh_probes > 0 ? cfg->lsh_probes : 4;

    keystone_error_t rc = keystone_lsh_create(&e->lsh, e->dim, num_tables, hash_bits, probes);
    if (rc != KEYSTONE_OK) {
        free(e->vectors); free(e->ids); free(e->sorted_ids); free(e->sorted_indices); free(e);
        return rc;
    }

    /* Select best available backend */
    rc = select_backend(e);
    if (rc != KEYSTONE_OK) {
        keystone_lsh_destroy(e->lsh);
        free(e->vectors); free(e->ids); free(e->sorted_ids); free(e->sorted_indices); free(e);
        return rc;
    }

    *out = e;
    return KEYSTONE_OK;
}

void keystone_vec_engine_destroy(keystone_vec_engine_t *e) {
    if (!e) return;

#ifdef KEYSTONE_HAVE_VPU
    if (e->vpu_connected) {
        keystone_vpu_kernel_disconnect();
        e->vpu_connected = 0;
    }
#endif

    if (e->cuda_initialized) {
        cuda_teardown(e);
    }

    if (e->lsh) keystone_lsh_destroy(e->lsh);
    free(e->vectors);
    free(e->ids);
    free(e->sorted_ids);
    free(e->sorted_indices);
    free(e);
}

keystone_backend_t keystone_vec_engine_get_backend(const keystone_vec_engine_t *e) {
    if (!e) return KEYSTONE_BACKEND_SCALAR;
    return e->backend;
}

/* ---------------------------------------------------------------
 * ID lookup (binary search on sorted ids)
 * --------------------------------------------------------------- */
static void sort_ids(keystone_vec_engine_t *e) {
    if (e->ids_sorted || e->count == 0) return;
    for (uint32_t i = 0; i < e->count; i++) {
        e->sorted_ids[i] = e->ids[i];
        e->sorted_indices[i] = i;
    }
    /* Simple insertion sort (n is typically large, but we only sort once) */
    for (uint32_t i = 1; i < e->count; i++) {
        uint64_t key = e->sorted_ids[i];
        uint32_t idx = e->sorted_indices[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && e->sorted_ids[j] > key) {
            e->sorted_ids[j + 1] = e->sorted_ids[j];
            e->sorted_indices[j + 1] = e->sorted_indices[j];
            j--;
        }
        e->sorted_ids[j + 1] = key;
        e->sorted_indices[j + 1] = idx;
    }
    e->ids_sorted = 1;
}

static int32_t find_id(keystone_vec_engine_t *e, uint64_t id) {
    if (!e->ids_sorted) sort_ids(e);
    int32_t lo = 0, hi = (int32_t)e->count - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (e->sorted_ids[mid] == id) return (int32_t)e->sorted_indices[mid];
        if (e->sorted_ids[mid] < id) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ---------------------------------------------------------------
 * Vector I/O
 * --------------------------------------------------------------- */
keystone_error_t keystone_vec_upsert_batch(keystone_vec_engine_t *e,
                                            const uint64_t *ids,
                                            const float *vectors,
                                            uint32_t n) {
    if (!e || !ids || !vectors) return KEYSTONE_ERR_NULL;
    if (n == 0) return KEYSTONE_OK;

    for (uint32_t i = 0; i < n; i++) {
        if (e->count >= e->capacity) return KEYSTONE_ERR_CAPACITY;

        int32_t existing = find_id(e, ids[i]);
        uint32_t idx;
        if (existing >= 0) {
            idx = (uint32_t)existing;
        } else {
            idx = e->count++;
            e->ids[idx] = ids[i];
            e->ids_sorted = 0;
            e->lsh_finalized = 0;
        }

        /* Copy vector data */
        memcpy(&e->vectors[(size_t)idx * e->dim], &vectors[(size_t)i * e->dim],
               (size_t)e->dim * sizeof(float));

        /* Normalize if using cosine metric */
        if (e->cfg.metric == KEYSTONE_METRIC_COSINE && e->kernels && e->kernels->normalize_batch) {
            e->kernels->normalize_batch(&e->vectors[(size_t)idx * e->dim], 1, e->dim);
        }

        /* Insert into LSH if new */
        if (existing < 0) {
            keystone_lsh_insert(e->lsh, &e->vectors[(size_t)idx * e->dim], idx);
        }
    }
    return KEYSTONE_OK;
}

/* ---------------------------------------------------------------
 * Search
 * --------------------------------------------------------------- */

/* Partial sort: keep top-k smallest distances */
static void topk_update(keystone_vec_result_t *results, uint32_t k,
                        uint32_t *count, uint64_t id, float dist) {
    if (*count < k) {
        /* Insert into partially filled array */
        uint32_t pos = *count;
        while (pos > 0 && results[pos - 1].distance > dist) {
            results[pos] = results[pos - 1];
            pos--;
        }
        results[pos].id = id;
        results[pos].distance = dist;
        (*count)++;
    } else if (dist < results[k - 1].distance) {
        /* Replace worst and re-sort */
        uint32_t pos = k - 1;
        while (pos > 0 && results[pos - 1].distance > dist) {
            results[pos] = results[pos - 1];
            pos--;
        }
        results[pos].id = id;
        results[pos].distance = dist;
    }
}

static keystone_error_t search_cpu(keystone_vec_engine_t *e,
                                    const float *query,
                                    uint32_t k,
                                    keystone_vec_result_t *results) {
    /* Try LSH first for sub-linear search */
    if (e->lsh_finalized && e->lsh) {
        uint32_t max_cands = e->cfg.rerank_k > 0 ? e->cfg.rerank_k : 256;
        if (max_cands > e->count) max_cands = e->count;

        uint32_t stack_cands[512];
        uint32_t *cand_indices = (max_cands <= 512)
            ? stack_cands
            : (uint32_t *)malloc(max_cands * sizeof(uint32_t));
        if (!cand_indices) return KEYSTONE_ERR_OOM;

        uint32_t n_cands = 0;
        keystone_lsh_query(e->lsh, query, cand_indices, max_cands, &n_cands);

        if (n_cands > 0) {
            /* Rerank candidates with exact distance */
            uint32_t count = 0;
            for (uint32_t i = 0; i < n_cands; i++) {
                uint32_t idx = cand_indices[i];
                float dist = e->kernels->dist1(query, &e->vectors[(size_t)idx * e->dim], e->dim);
                topk_update(results, k, &count, e->ids[idx], dist);
            }
            if (cand_indices != stack_cands) free(cand_indices);
            return KEYSTONE_OK;
        }
        if (cand_indices != stack_cands) free(cand_indices);
        /* LSH found nothing — fall through to brute force */
    }

    /* Brute force scan */
    uint32_t count = 0;
    for (uint32_t i = 0; i < e->count; i++) {
        float dist = e->kernels->dist1(query, &e->vectors[(size_t)i * e->dim], e->dim);
        topk_update(results, k, &count, e->ids[i], dist);
    }
    return KEYSTONE_OK;
}

static keystone_error_t search_cuda(keystone_vec_engine_t *e,
                                     const float *query,
                                     uint32_t k,
                                     keystone_vec_result_t *results) {
#ifdef __unix__
    if (!e->cuda_batch_dist || !e->cuda_initialized) {
        /* CUDA died — fall back to CPU */
        cuda_teardown(e);
        select_backend(e);
        return search_cpu(e, query, k, results);
    }

    /* Compute all distances on GPU */
    float *distances = (float *)malloc(e->count * sizeof(float));
    if (!distances) return KEYSTONE_ERR_OOM;

    int rc = e->cuda_batch_dist(query, e->vectors, e->count, e->dim, distances);
    if (rc != 0) {
        /* CUDA error — degrade to CPU */
        free(distances);
        cuda_teardown(e);
        select_backend(e);
        return search_cpu(e, query, k, results);
    }

    /* Find top-k on CPU (small compared to distance computation) */
    uint32_t count = 0;
    for (uint32_t i = 0; i < e->count; i++) {
        topk_update(results, k, &count, e->ids[i], distances[i]);
    }
    free(distances);
    return KEYSTONE_OK;
#else
    return search_cpu(e, query, k, results);
#endif
}

keystone_error_t keystone_vec_search(keystone_vec_engine_t *e,
                                      const float *query,
                                      uint32_t k,
                                      keystone_vec_result_t *results) {
    if (!e || !query || !results) return KEYSTONE_ERR_NULL;
    if (k == 0) return KEYSTONE_ERR_PARAM;
    if (e->count == 0) return KEYSTONE_OK;

    /* Normalize query if cosine */
    if (e->cfg.metric == KEYSTONE_METRIC_COSINE) {
        float stack_qcopy[1024];
        float *qcopy = (e->dim <= 1024)
            ? stack_qcopy
            : (float *)malloc(e->dim * sizeof(float));
        if (!qcopy) return KEYSTONE_ERR_OOM;
        memcpy(qcopy, query, e->dim * sizeof(float));
        if (e->kernels && e->kernels->normalize_batch) {
            e->kernels->normalize_batch(qcopy, 1, e->dim);
        }
        keystone_error_t rc;
        if (e->backend == KEYSTONE_BACKEND_CUDA) {
            rc = search_cuda(e, qcopy, k, results);
        } else {
            rc = search_cpu(e, qcopy, k, results);
        }
        if (qcopy != stack_qcopy) free(qcopy);
        return rc;
    }

    if (e->backend == KEYSTONE_BACKEND_CUDA) {
        return search_cuda(e, query, k, results);
    }
    return search_cpu(e, query, k, results);
}

keystone_error_t keystone_vec_search_batch(keystone_vec_engine_t *e,
                                            const float *queries,
                                            uint32_t nq,
                                            uint32_t k,
                                            keystone_vec_result_t *results) {
    if (!e || !queries || !results) return KEYSTONE_ERR_NULL;
    if (k == 0 || nq == 0) return KEYSTONE_ERR_PARAM;
    if (e->count == 0) return KEYSTONE_OK;

    /* Ensure LSH is finalized for batch search */
    if (!e->lsh_finalized) {
        keystone_lsh_finalize(e->lsh);
        e->lsh_finalized = 1;
    }

#ifdef _OPENMP
    int nthreads = e->cfg.omp_threads > 0 ? e->cfg.omp_threads : omp_get_max_threads();
    if (nthreads > (int)nq) nthreads = (int)nq;
    #pragma omp parallel for num_threads(nthreads)
#endif
    for (uint32_t q = 0; q < nq; q++) {
        const float *query = &queries[(size_t)q * e->dim];
        keystone_vec_result_t *out = &results[(size_t)q * k];
        keystone_vec_search(e, query, k, out);
    }

    return KEYSTONE_OK;
}

/* ---------------------------------------------------------------
 * Introspection
 * --------------------------------------------------------------- */
uint32_t keystone_vec_engine_dim(const keystone_vec_engine_t *e) {
    return e ? e->dim : 0;
}

uint64_t keystone_vec_engine_count(const keystone_vec_engine_t *e) {
    return e ? e->count : 0;
}
