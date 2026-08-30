/*
 * keystone_vector_engine.h - KEYSTONE Vector Engine
 *
 * Hardware-agnostic vector similarity search with graceful fallback:
 *
 *   1. CUDA GPU present + init OK?      -> CUDA batch path
 *   2. AVX-512 present?                 -> AVX-512 SIMD path
 *   3. AVX2 present?                    -> AVX2 SIMD path
 *   4. AVX present?                     -> AVX SIMD path (256-bit YMM)
 *   5. SSE4.2 present?                  -> SSE4.2 SIMD path
 *   6. NEON present (ARM)?              -> NEON SIMD path
 *   7. VPU present (Myriad X)?          -> VPU accelerated path [guarded]
 *   8. None of the above?               -> Scalar C path (ALWAYS compiled)
 *
 * VPU INTEGRATION:
 *   The VPU (Intel Myriad X) code is COMPILE-TIME guarded by
 *   KEYSTONE_HAVE_VPU and RUNTIME detected via socket probe.
 *   If KEYSTONE_HAVE_VPU is not defined at compile time, no VPU
 *   code is compiled. If compiled but the VPU socket is not present
 *   at runtime, the engine silently falls back to CPU SIMD.
 *   The VPU can accelerate both embedding generation (external
 *   process) and batch similarity computation (direct kernel).
 *
 * EMBEDDING SOURCE AGNOSTIC:
 *   This library stores and searches float32 arrays. It does not
 *   know or care where vectors came from. Vectors arrive via
 *   upsert_batch() from any source.
 */

#ifndef KEYSTONE_VECTOR_ENGINE_H
#define KEYSTONE_VECTOR_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KEYSTONE_VEC_VERSION_MAJOR 2
#define KEYSTONE_VEC_VERSION_MINOR 1

/* ---------------------------------------------------------------
 * Error codes
 * --------------------------------------------------------------- */
typedef enum {
    KEYSTONE_OK              = 0,
    KEYSTONE_ERR_NULL        = -1,
    KEYSTONE_ERR_PARAM       = -2,
    KEYSTONE_ERR_DIM         = -3,
    KEYSTONE_ERR_OOM         = -4,
    KEYSTONE_ERR_CAPACITY    = -5,
    KEYSTONE_ERR_IO          = -6,
    KEYSTONE_ERR_INTERNAL    = -7,
    KEYSTONE_ERR_BACKEND     = -8   /* backend init failed, degraded */
} keystone_error_t;

/* ---------------------------------------------------------------
 * Backend identifiers - mirrors the fallback chain.
 * VPU is level 7, between NEON and scalar.
 * --------------------------------------------------------------- */
typedef enum {
    KEYSTONE_BACKEND_SCALAR  = 0,   /* level 8: portable baseline    */
    KEYSTONE_BACKEND_VPU     = 1,   /* level 7: Myriad X [guarded]   */
    KEYSTONE_BACKEND_NEON    = 2,   /* level 6: ARM SIMD             */
    KEYSTONE_BACKEND_SSE42   = 3,   /* level 5                       */
    KEYSTONE_BACKEND_AVX     = 4,   /* level 4: YMM, float only      */
    KEYSTONE_BACKEND_AVX2    = 5,   /* level 3                       */
    KEYSTONE_BACKEND_AVX512  = 6,   /* level 2                       */
    KEYSTONE_BACKEND_CUDA    = 7    /* level 1: GPU                  */
} keystone_backend_t;

/* ---------------------------------------------------------------
 * CPU feature flags (runtime detection)
 * --------------------------------------------------------------- */
typedef enum {
    KEYSTONE_F_SSE42    = 1 << 0,
    KEYSTONE_F_AVX      = 1 << 1,
    KEYSTONE_F_AVX2     = 1 << 2,
    KEYSTONE_F_AVX512F  = 1 << 3,
    KEYSTONE_F_NEON     = 1 << 4,
    KEYSTONE_F_VPU      = 1 << 5    /* Myriad X socket reachable     */
} keystone_cpu_features_t;

/* ---------------------------------------------------------------
 * Distance metrics
 * --------------------------------------------------------------- */
typedef enum {
    KEYSTONE_METRIC_L2      = 0,
    KEYSTONE_METRIC_DOT     = 1,
    KEYSTONE_METRIC_COSINE  = 2
} keystone_metric_t;

/* ---------------------------------------------------------------
 * Engine configuration
 *
 * VPU fields are present but only used when KEYSTONE_HAVE_VPU
 * is defined at compile time AND the VPU socket is reachable
 * at runtime. Otherwise they are ignored.
 * --------------------------------------------------------------- */
typedef struct {
    uint32_t    dim;                /* vector dimension (e.g. 384)   */
    uint32_t    capacity;           /* max number of vectors         */
    keystone_metric_t metric;

    /* LSH coarse-index parameters */
    uint32_t    lsh_num_tables;     /* e.g. 8                        */
    uint32_t    lsh_hash_bits;      /* e.g. 12 -> 4096 buckets       */
    uint32_t    lsh_probes;         /* multiprobe count, e.g. 4      */
    uint32_t    rerank_k;           /* candidates to exact-rerank    */

    /* Threading */
    int         omp_threads;        /* 0 = let OpenMP decide         */

    /* Backend override: pass -1 for automatic dispatch.
     * Pass a specific backend ID to force it (useful for testing). */
    int         force_backend;

    /* VPU configuration (ignored if KEYSTONE_HAVE_VPU not defined).
     * If vpu_socket_path is NULL or empty, uses default path. */
    char        vpu_socket_path[108];
    int         vpu_enabled;        /* 0 = skip VPU even if present  */
} keystone_config_t;

/* Opaque engine handle */
typedef struct keystone_vec_engine keystone_vec_engine_t;

/* ---------------------------------------------------------------
 * Search result
 * --------------------------------------------------------------- */
typedef struct {
    uint64_t    id;
    float       distance;
} keystone_vec_result_t;

/* ---------------------------------------------------------------
 * Runtime hardware detection
 *
 * Returns feature flags detected at runtime. Always succeeds;
 * returns 0 on any machine with no SIMD.
 * --------------------------------------------------------------- */
uint32_t keystone_vec_detect_features(void);

/* Returns human-readable name of a backend. Never returns NULL. */
const char *keystone_vec_backend_name(keystone_backend_t b);

/* ---------------------------------------------------------------
 * Lifecycle
 * --------------------------------------------------------------- */
keystone_error_t keystone_vec_engine_create(const keystone_config_t *cfg,
                                             keystone_vec_engine_t **out);

void             keystone_vec_engine_destroy(keystone_vec_engine_t *e);

/* Which backend is this engine using right now? After a CUDA or VPU
 * runtime failure this reports the CPU backend it fell back to. */
keystone_backend_t keystone_vec_engine_get_backend(const keystone_vec_engine_t *e);

/* ---------------------------------------------------------------
 * Vector I/O - the primary input path. Hardware-agnostic.
 *
 * Vectors are float32 arrays of engine->dim, row-major.
 * --------------------------------------------------------------- */
keystone_error_t keystone_vec_upsert_batch(keystone_vec_engine_t *e,
                                            const uint64_t *ids,
                                            const float *vectors,
                                            uint32_t n);

/* Batch search. OpenMP-parallel across queries when available. */
keystone_error_t keystone_vec_search_batch(keystone_vec_engine_t *e,
                                            const float *queries,
                                            uint32_t nq,
                                            uint32_t k,
                                            keystone_vec_result_t *results);

/* Single-query convenience wrapper. */
keystone_error_t keystone_vec_search(keystone_vec_engine_t *e,
                                      const float *query,
                                      uint32_t k,
                                      keystone_vec_result_t *results);

/* ---------------------------------------------------------------
 * Persistence
 * --------------------------------------------------------------- */
keystone_error_t keystone_vec_save(const keystone_vec_engine_t *e,
                                    const char *path);
keystone_error_t keystone_vec_load(keystone_vec_engine_t **e,
                                    const char *path);

/* ---------------------------------------------------------------
 * Introspection
 * --------------------------------------------------------------- */
uint32_t keystone_vec_engine_dim(const keystone_vec_engine_t *e);
uint64_t keystone_vec_engine_count(const keystone_vec_engine_t *e);

/* ---------------------------------------------------------------
 * VPU-specific functions (only compiled with KEYSTONE_HAVE_VPU).
 * These are NO-OPs (return KEYSTONE_ERR_BACKEND) if VPU support
 * is not compiled in.
 * --------------------------------------------------------------- */
#ifdef KEYSTONE_HAVE_VPU
keystone_error_t keystone_vec_vpu_connect(keystone_vec_engine_t *e,
                                           const char *socket_path);
keystone_error_t keystone_vec_vpu_disconnect(keystone_vec_engine_t *e);
int keystone_vec_vpu_available(const char *socket_path);
#else
/* Stub implementations when VPU is not compiled in */
static inline keystone_error_t keystone_vec_vpu_connect(
    keystone_vec_engine_t *e, const char *socket_path) {
    (void)e; (void)socket_path;
    return KEYSTONE_ERR_BACKEND;
}
static inline keystone_error_t keystone_vec_vpu_disconnect(
    keystone_vec_engine_t *e) {
    (void)e;
    return KEYSTONE_ERR_BACKEND;
}
static inline int keystone_vec_vpu_available(const char *socket_path) {
    (void)socket_path;
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* KEYSTONE_VECTOR_ENGINE_H */
