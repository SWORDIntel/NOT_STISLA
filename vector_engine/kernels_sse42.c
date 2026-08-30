/*
 * kernels_sse42.c - KEYSTONE Vector Engine SSE4.2 kernel backend
 *
 * Compiled with -msse4.2. Uses 128-bit XMM registers for 4x float32
 * parallelism. For the canonical KEYSTONE dimension of 384, 384 / 4 = 96
 * iterations with no remainder, so the main loop is clean. A scalar
 * tail handles arbitrary dimensions that are not a multiple of 4.
 *
 * Metric dispatch mirrors the scalar backend: a thread-local g_metric
 * is set once via keystone_sse42_set_metric() and checked in dist1.
 */

#include "kernel_dispatch.h"

#include <smmintrin.h>   /* SSE4.2 intrinsics */
#include <math.h>

/* ---------------------------------------------------------------
 * Thread-local metric. Default cosine.
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_sse42_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * dist1 - SSE4.2 vectorized distance.
 *
 *   COSINE : 1.0 - dot(a,b)
 *   L2     : sum of squared differences
 *   DOT    : -dot(a,b)
 *
 * Processes 4 floats per iteration. Scalar tail handles dim % 4.
 * --------------------------------------------------------------- */
static float sse42_dist1(const float *a, const float *b, uint32_t dim)
{
    uint32_t i = 0;
    __m128 v_acc = _mm_setzero_ps();

    switch (g_metric) {
    case KEYSTONE_METRIC_L2: {
        for (; i + 4 <= dim; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 d  = _mm_sub_ps(va, vb);
            v_acc = _mm_add_ps(v_acc, _mm_mul_ps(d, d));
        }
        /* horizontal sum */
        float buf[4];
        _mm_storeu_ps(buf, v_acc);
        float acc = buf[0] + buf[1] + buf[2] + buf[3];
        /* scalar tail */
        for (; i < dim; i++) {
            float d = a[i] - b[i];
            acc += d * d;
        }
        return acc;
    }
    case KEYSTONE_METRIC_DOT: {
        for (; i + 4 <= dim; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            v_acc = _mm_add_ps(v_acc, _mm_mul_ps(va, vb));
        }
        float buf[4];
        _mm_storeu_ps(buf, v_acc);
        float acc = buf[0] + buf[1] + buf[2] + buf[3];
        for (; i < dim; i++)
            acc += a[i] * b[i];
        return -acc;
    }
    case KEYSTONE_METRIC_COSINE:
    default: {
        for (; i + 4 <= dim; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            v_acc = _mm_add_ps(v_acc, _mm_mul_ps(va, vb));
        }
        float buf[4];
        _mm_storeu_ps(buf, v_acc);
        float acc = buf[0] + buf[1] + buf[2] + buf[3];
        for (; i < dim; i++)
            acc += a[i] * b[i];
        return 1.0f - acc;
    }
    }
}

/* ---------------------------------------------------------------
 * dist_batch - SSE4.2 vectorized batch distance.
 * --------------------------------------------------------------- */
static void sse42_dist_batch(const float *q,
                             const float *cands,
                             uint32_t n_cands,
                             uint32_t dim,
                             float *out)
{
    for (uint32_t i = 0; i < n_cands; i++)
        out[i] = sse42_dist1(q, cands + (size_t)i * dim, dim);
}

/* ---------------------------------------------------------------
 * normalize_batch - SSE4.2 in-place L2 normalization.
 *
 * Norm computation is vectorized (4-wide dot of vector with itself).
 * Division is vectorized when the norm is non-zero. Zero-norm
 * vectors are left untouched.
 * --------------------------------------------------------------- */
static void sse42_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;

        uint32_t i = 0;
        __m128 v_acc = _mm_setzero_ps();
        for (; i + 4 <= dim; i += 4) {
            __m128 va = _mm_loadu_ps(vec + i);
            v_acc = _mm_add_ps(v_acc, _mm_mul_ps(va, va));
        }
        float buf[4];
        _mm_storeu_ps(buf, v_acc);
        float sum = buf[0] + buf[1] + buf[2] + buf[3];
        for (; i < dim; i++)
            sum += vec[i] * vec[i];

        float norm = sqrtf(sum);
        if (norm > 0.0f) {
            __m128 v_inv = _mm_set1_ps(1.0f / norm);
            i = 0;
            for (; i + 4 <= dim; i += 4) {
                __m128 va = _mm_loadu_ps(vec + i);
                _mm_storeu_ps(vec + i, _mm_mul_ps(va, v_inv));
            }
            for (; i < dim; i++)
                vec[i] *= 1.0f / norm;
        }
    }
}

/* ---------------------------------------------------------------
 * Exported kernel table
 * --------------------------------------------------------------- */
const keystone_kernels_t keystone_kernels_sse42 = {
    .dist1           = sse42_dist1,
    .dist_batch      = sse42_dist_batch,
    .normalize_batch = sse42_normalize_batch,
};
