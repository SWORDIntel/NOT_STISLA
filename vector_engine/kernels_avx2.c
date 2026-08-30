/*
 * kernels_avx2.c - KEYSTONE Vector Engine AVX2+FMA kernel translation unit
 *
 * Compiled with -mavx2 -mfma.
 * Uses 256-bit YMM registers with fused multiply-add:
 *   _mm256_fmadd_ps  (3-operand, faster than separate mul+add)
 *
 * Vector dimension 384: 384 / 8 = 48 FMA iterations per dot product.
 * 384 mod 8 = 0 — clean fit, no remainder (scalar tail kept for
 * correctness on non-384 dimensions).
 */

#include <immintrin.h>
#include <math.h>

#include "kernel_dispatch.h"

/* ---------------------------------------------------------------
 * Per-thread metric selector
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_avx2_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * Horizontal reduction of a 256-bit vector to a single float.
 * --------------------------------------------------------------- */
static inline float avx2_hsum256(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* ---------------------------------------------------------------
 * SIMD dot product (AVX2 + FMA, 8x float32 per iteration)
 * Uses _mm256_fmadd_ps: acc = va * vb + acc
 * --------------------------------------------------------------- */
static inline float avx2_dot(const float *a, const float *b, uint32_t dim)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float result = avx2_hsum256(acc);
    for (; i < dim; i++)
        result += a[i] * b[i];
    return result;
}

/* ---------------------------------------------------------------
 * SIMD squared L2 distance (AVX2 + FMA)
 * d = a - b; acc = d * d + acc
 * --------------------------------------------------------------- */
static inline float avx2_l2_sq(const float *a, const float *b, uint32_t dim)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float result = avx2_hsum256(acc);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

/* ---------------------------------------------------------------
 * dist1 — single-pair distance, dispatched on g_metric
 * --------------------------------------------------------------- */
static float avx2_dist1(const float *a, const float *b, uint32_t dim)
{
    switch (g_metric) {
    case KEYSTONE_METRIC_L2:
        return avx2_l2_sq(a, b, dim);
    case KEYSTONE_METRIC_DOT:
        return -avx2_dot(a, b, dim);
    case KEYSTONE_METRIC_COSINE:
    default:
        return 1.0f - avx2_dot(a, b, dim);
    }
}

/* ---------------------------------------------------------------
 * dist_batch — loop over candidates, each uses AVX2+FMA inner loop
 * --------------------------------------------------------------- */
static void avx2_dist_batch(const float *q,
                            const float *cands,
                            uint32_t n_cands,
                            uint32_t dim,
                            float *out)
{
    for (uint32_t c = 0; c < n_cands; c++) {
        const float *cand = cands + (size_t)c * dim;
        out[c] = avx2_dist1(q, cand, dim);
    }
}

/* ---------------------------------------------------------------
 * normalize_batch — FMA for norm (sum of squares), scalar divide
 * --------------------------------------------------------------- */
static void avx2_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;

        __m256 acc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(vec + i);
            acc = _mm256_fmadd_ps(va, va, acc);
        }
        float sumsq = avx2_hsum256(acc);
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
const keystone_kernels_t keystone_kernels_avx2 = {
    .dist1          = avx2_dist1,
    .dist_batch     = avx2_dist_batch,
    .normalize_batch = avx2_normalize_batch,
};
