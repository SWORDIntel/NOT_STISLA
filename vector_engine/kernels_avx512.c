/*
 * kernels_avx512.c - KEYSTONE Vector Engine AVX-512 kernel translation unit
 *
 * Compiled with -mavx512f -mavx512dq.
 * Uses 512-bit ZMM registers (16x float32 per iteration):
 *   _mm512_loadu_ps, _mm512_mul_ps, _mm512_add_ps, _mm512_sub_ps
 *
 * Vector dimension 384: 384 / 16 = 24 iterations per dot product.
 * 384 mod 16 = 0 — clean fit, no remainder (scalar tail kept for
 * correctness on non-384 dimensions).
 */

#include <immintrin.h>
#include <math.h>

#include "kernel_dispatch.h"

/* ---------------------------------------------------------------
 * Per-thread metric selector
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_avx512_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * SIMD dot product (AVX-512, 16x float32 per iteration)
 * --------------------------------------------------------------- */
static inline float avx512_dot(const float *a, const float *b, uint32_t dim)
{
    __m512 acc = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, vb));
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < dim; i++)
        result += a[i] * b[i];
    return result;
}

/* ---------------------------------------------------------------
 * SIMD squared L2 distance (AVX-512)
 * --------------------------------------------------------------- */
static inline float avx512_l2_sq(const float *a, const float *b, uint32_t dim)
{
    __m512 acc = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d  = _mm512_sub_ps(va, vb);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(d, d));
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

/* ---------------------------------------------------------------
 * SIMD sum of squares (AVX-512) — used by normalize_batch
 * --------------------------------------------------------------- */
static inline float avx512_sumsq(const float *vec, uint32_t dim)
{
    __m512 acc = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(vec + i);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, va));
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < dim; i++)
        result += vec[i] * vec[i];
    return result;
}

/* ---------------------------------------------------------------
 * dist1 — single-pair distance, dispatched on g_metric
 * --------------------------------------------------------------- */
static float avx512_dist1(const float *a, const float *b, uint32_t dim)
{
    switch (g_metric) {
    case KEYSTONE_METRIC_L2:
        return avx512_l2_sq(a, b, dim);
    case KEYSTONE_METRIC_DOT:
        return -avx512_dot(a, b, dim);
    case KEYSTONE_METRIC_COSINE:
    default:
        return 1.0f - avx512_dot(a, b, dim);
    }
}

/* ---------------------------------------------------------------
 * dist_batch — loop over candidates, each uses AVX-512 inner loop
 * --------------------------------------------------------------- */
static void avx512_dist_batch(const float *q,
                              const float *cands,
                              uint32_t n_cands,
                              uint32_t dim,
                              float *out)
{
    for (uint32_t c = 0; c < n_cands; c++) {
        const float *cand = cands + (size_t)c * dim;
        out[c] = avx512_dist1(q, cand, dim);
    }
}

/* ---------------------------------------------------------------
 * normalize_batch — AVX-512 for norm (sum of squares), scalar divide
 * --------------------------------------------------------------- */
static void avx512_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;

        float sumsq = avx512_sumsq(vec, dim);
        float norm  = sqrtf(sumsq);
        if (norm > 0.0f) {
            float inv = 1.0f / norm;
            for (uint32_t j = 0; j < dim; j++)
                vec[j] *= inv;
        }
    }
}

/* ---------------------------------------------------------------
 * Exported kernel table
 * --------------------------------------------------------------- */
const keystone_kernels_t keystone_kernels_avx512 = {
    .dist1          = avx512_dist1,
    .dist_batch     = avx512_dist_batch,
    .normalize_batch = avx512_normalize_batch,
};
