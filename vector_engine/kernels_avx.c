/*
 * kernels_avx.c - KEYSTONE Vector Engine AVX kernel translation unit
 *
 * Compiled with -mavx ONLY (no -mavx2, no -mfma).
 * Uses 256-bit YMM registers with float-only AVX intrinsics:
 *   _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, _mm256_sub_ps
 *
 * Vector dimension 384: 384 / 8 = 48 iterations per dot product.
 * 384 mod 8 = 0 — clean fit, no remainder (a scalar tail is kept
 * for correctness on non-384 dimensions).
 *
 * This is the PRIMARY SIMD path on hardware with base AVX but no AVX2.
 */

#include <immintrin.h>
#include <math.h>

#include "kernel_dispatch.h"

/* ---------------------------------------------------------------
 * Per-thread metric selector
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_avx_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * Horizontal reduction of a 256-bit vector to a single float.
 * Uses only base-AVX + SSE3 hadd (no AVX2 required).
 * --------------------------------------------------------------- */
static inline float avx_hsum256(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* ---------------------------------------------------------------
 * SIMD dot product (AVX, 8x float32 per iteration)
 * --------------------------------------------------------------- */
static inline float avx_dot(const float *a, const float *b, uint32_t dim)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float result = avx_hsum256(acc);
    for (; i < dim; i++)
        result += a[i] * b[i];
    return result;
}

/* ---------------------------------------------------------------
 * SIMD squared L2 distance (AVX)
 * --------------------------------------------------------------- */
static inline float avx_l2_sq(const float *a, const float *b, uint32_t dim)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    float result = avx_hsum256(acc);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

/* ---------------------------------------------------------------
 * dist1 — single-pair distance, dispatched on g_metric
 * --------------------------------------------------------------- */
static float avx_dist1(const float *a, const float *b, uint32_t dim)
{
    switch (g_metric) {
    case KEYSTONE_METRIC_L2:
        return avx_l2_sq(a, b, dim);
    case KEYSTONE_METRIC_DOT:
        return -avx_dot(a, b, dim);
    case KEYSTONE_METRIC_COSINE:
    default:
        return 1.0f - avx_dot(a, b, dim);
    }
}

/* ---------------------------------------------------------------
 * dist_batch — loop over candidates, each uses AVX inner loop
 * --------------------------------------------------------------- */
static void avx_dist_batch(const float *q,
                           const float *cands,
                           uint32_t n_cands,
                           uint32_t dim,
                           float *out)
{
    for (uint32_t c = 0; c < n_cands; c++) {
        const float *cand = cands + (size_t)c * dim;
        out[c] = avx_dist1(q, cand, dim);
    }
}

/* ---------------------------------------------------------------
 * normalize_batch — AVX for norm (sum of squares), scalar divide
 * --------------------------------------------------------------- */
static void avx_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;

        __m256 acc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(vec + i);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, va));
        }
        float sumsq = avx_hsum256(acc);
        for (; i < dim; i++)
            sumsq += vec[i] * vec[i];

        float norm = sqrtf(sumsq);
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
const keystone_kernels_t keystone_kernels_avx = {
    .dist1          = avx_dist1,
    .dist_batch     = avx_dist_batch,
    .normalize_batch = avx_normalize_batch,
};
