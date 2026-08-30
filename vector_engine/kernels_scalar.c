/*
 * kernels_scalar.c - KEYSTONE Vector Engine scalar kernel backend
 *
 * Portable C11 implementation of the kernel ABI. No SIMD intrinsics,
 * no ISA flags. This is the ALWAYS-linked baseline that every other
 * backend falls back to.
 *
 * Metric dispatch: the engine sets the active metric once at creation
 * time via keystone_scalar_set_metric(). dist1 checks the thread-local
 * g_metric and computes the appropriate distance. This avoids changing
 * the fixed ABI (which has no metric parameter) while still supporting
 * all three metrics (cosine, L2, dot) from a single translation unit.
 */

#include "kernel_dispatch.h"

#include <math.h>

/* ---------------------------------------------------------------
 * Thread-local metric. Default cosine. The engine calls the setter
 * once after creating the backend so all subsequent dist1 calls use
 * the right formula without an ABI change.
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_scalar_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * dist1 - distance between two vectors of length dim.
 *
 *   COSINE : 1.0 - dot(a,b)   (assumes pre-normalized vectors)
 *   L2     : sum of squared differences
 *   DOT    : -dot(a,b)
 * --------------------------------------------------------------- */
static float scalar_dist1(const float *a, const float *b, uint32_t dim)
{
    switch (g_metric) {
    case KEYSTONE_METRIC_L2: {
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float d = a[i] - b[i];
            acc += d * d;
        }
        return acc;
    }
    case KEYSTONE_METRIC_DOT: {
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            acc += a[i] * b[i];
        return -acc;
    }
    case KEYSTONE_METRIC_COSINE:
    default: {
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            acc += a[i] * b[i];
        return 1.0f - acc;
    }
    }
}

/* ---------------------------------------------------------------
 * dist_batch - compute dist1 for one query against n_cands candidates.
 * Candidates are row-major: cands[i * dim + j].
 * --------------------------------------------------------------- */
static void scalar_dist_batch(const float *q,
                              const float *cands,
                              uint32_t n_cands,
                              uint32_t dim,
                              float *out)
{
    for (uint32_t i = 0; i < n_cands; i++)
        out[i] = scalar_dist1(q, cands + (size_t)i * dim, dim);
}

/* ---------------------------------------------------------------
 * normalize_batch - in-place L2 normalization for cosine metric.
 * Each of n vectors (dim floats apart) is divided by its L2 norm.
 * Zero-norm vectors are left untouched (avoid div-by-zero).
 * --------------------------------------------------------------- */
static void scalar_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            sum += vec[i] * vec[i];
        float norm = sqrtf(sum);
        if (norm > 0.0f) {
            float inv = 1.0f / norm;
            for (uint32_t i = 0; i < dim; i++)
                vec[i] *= inv;
        }
    }
}

/* ---------------------------------------------------------------
 * Exported kernel table
 * --------------------------------------------------------------- */
const keystone_kernels_t keystone_kernels_scalar = {
    .dist1           = scalar_dist1,
    .dist_batch      = scalar_dist_batch,
    .normalize_batch = scalar_normalize_batch,
};
