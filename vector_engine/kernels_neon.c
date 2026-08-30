/*
 * kernels_neon.c - KEYSTONE Vector Engine ARM NEON kernel backend
 *
 * 128-bit NEON SIMD implementation of the kernel ABI. Processes 4
 * float32 lanes per iteration using vld1q_f32 / vmulq_f32 / vaddq_f32.
 * For the canonical 384-dim embedding this is 96 vectorized iterations
 * plus a scalar tail (384 % 4 == 0, so no tail in the common case).
 *
 * If the build target does not expose NEON this translation unit is
 * empty (just a comment) so it links cleanly on non-ARM hosts.
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include "kernel_dispatch.h"
#include <arm_neon.h>
#include <math.h>

/* ---------------------------------------------------------------
 * Thread-local metric. Default cosine. The engine calls the setter
 * once after creating the backend so all subsequent dist1 calls use
 * the right formula without an ABI change.
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_neon_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * neon_dot - vectorized dot product of two dim-length float arrays.
 * Processes 4 floats per iteration with NEON intrinsics, then folds
 * the 4-lane accumulator down to a single float via vget_lane_f32.
 * --------------------------------------------------------------- */
static float neon_dot(const float *a, const float *b, uint32_t dim)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    /* Fast path: 4 floats per iteration. For dim=384 this is 96 iters. */
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vaddq_f32(acc, vmulq_f32(va, vb));
    }

    /* Horizontal sum of the 4-lane accumulator. */
    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float s = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

    /* Scalar tail for dims not divisible by 4 (384 is, so usually 0). */
    for (; i < dim; i++)
        s += a[i] * b[i];

    return s;
}

/* ---------------------------------------------------------------
 * neon_l2 - vectorized squared L2 distance between two dim-length
 * float arrays. Same 4-lane structure as neon_dot.
 * --------------------------------------------------------------- */
static float neon_l2(const float *a, const float *b, uint32_t dim)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vaddq_f32(acc, vmulq_f32(diff, diff));
    }

    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t sum2 = vadd_f32(lo, hi);
    float s = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

    for (; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }

    return s;
}

/* ---------------------------------------------------------------
 * dist1 - distance between two vectors of length dim.
 *
 *   COSINE : 1.0 - dot(a,b)   (assumes pre-normalized vectors)
 *   L2     : sum of squared differences
 *   DOT    : -dot(a,b)
 * --------------------------------------------------------------- */
static float neon_dist1(const float *a, const float *b, uint32_t dim)
{
    if (!a || !b || dim == 0)
        return 0.0f;

    switch (g_metric) {
    case KEYSTONE_METRIC_L2:
        return neon_l2(a, b, dim);
    case KEYSTONE_METRIC_DOT:
        return -neon_dot(a, b, dim);
    case KEYSTONE_METRIC_COSINE:
    default:
        return 1.0f - neon_dot(a, b, dim);
    }
}

/* ---------------------------------------------------------------
 * dist_batch - compute dist1 for one query against n_cands candidates.
 * Candidates are row-major: cands[i * dim + j].
 * --------------------------------------------------------------- */
static void neon_dist_batch(const float *q,
                            const float *cands,
                            uint32_t n_cands,
                            uint32_t dim,
                            float *out)
{
    if (!q || !cands || !out || dim == 0 || n_cands == 0)
        return;

    for (uint32_t i = 0; i < n_cands; i++)
        out[i] = neon_dist1(q, cands + (size_t)i * dim, dim);
}

/* ---------------------------------------------------------------
 * normalize_batch - in-place L2 normalization for cosine metric.
 * Each of n vectors (dim floats apart) is divided by its L2 norm.
 * Vectors with norm < 1e-12 are left untouched (avoid div-by-zero).
 * --------------------------------------------------------------- */
static void neon_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    if (!vectors || dim == 0 || n == 0)
        return;

    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;

        /* Vectorized sum of squares. */
        float32x4_t acc = vdupq_n_f32(0.0f);
        uint32_t i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vv = vld1q_f32(vec + i);
            acc = vaddq_f32(acc, vmulq_f32(vv, vv));
        }

        float32x2_t lo = vget_low_f32(acc);
        float32x2_t hi = vget_high_f32(acc);
        float32x2_t sum2 = vadd_f32(lo, hi);
        float sum = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

        for (; i < dim; i++)
            sum += vec[i] * vec[i];

        float norm = sqrtf(sum);
        if (norm < 1e-12f)
            continue;

        float inv = 1.0f / norm;
        float32x4_t vinv = vdupq_n_f32(inv);
        i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t vv = vld1q_f32(vec + i);
            vst1q_f32(vec + i, vmulq_f32(vv, vinv));
        }
        for (; i < dim; i++)
            vec[i] *= inv;
    }
}

/* ---------------------------------------------------------------
 * Exported kernel table
 * --------------------------------------------------------------- */
const keystone_kernels_t keystone_kernels_neon = {
    .dist1           = neon_dist1,
    .dist_batch      = neon_dist_batch,
    .normalize_batch = neon_normalize_batch,
};

#else  /* !(defined(__ARM_NEON) || defined(__ARM_NEON__)) */

/*
 * No NEON on this build target. This translation unit is intentionally
 * empty so the file links cleanly on non-ARM hosts. The scalar backend
 * (always compiled) provides the fallback.
 */

#endif /* __ARM_NEON */
