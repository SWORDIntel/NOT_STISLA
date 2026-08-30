/*
 * kernel_dispatch.h - Internal kernel ABI and dispatch table
 *
 * Every backend (scalar, SSE4.2, AVX, AVX2, AVX-512, NEON, VPU, CUDA)
 * implements the same kernel ABI. The engine holds a const pointer to
 * the selected backend's kernel table and calls through function
 * pointers — dispatch is a single pointer swap, not a per-call branch.
 */
#ifndef KEYSTONE_KERNEL_DISPATCH_H
#define KEYSTONE_KERNEL_DISPATCH_H

#include "keystone_vector_engine.h"

/* Kernel ABI — every backend implements these. */
typedef struct {
    /* Distance between one query and one stored vector.
     * For cosine: returns 1.0 - dot(a,b) (lower = more similar).
     * For L2: returns squared L2 distance.
     * For dot: returns negative dot product. */
    float (*dist1)(const float *a, const float *b, uint32_t dim);

    /* Batch distance: compute dist1 for query q against n_cands
     * candidate vectors (each dim floats apart, row-major). */
    void (*dist_batch)(const float *q,
                       const float *cands,
                       uint32_t n_cands,
                       uint32_t dim,
                       float *out);

    /* Normalize a batch of vectors in-place (for cosine metric).
     * Divides each vector by its L2 norm. */
    void (*normalize_batch)(float *vectors, uint32_t n, uint32_t dim);

    /* Set the distance metric for this kernel (thread-local). */
    void (*set_metric)(keystone_metric_t m);
} keystone_kernels_t;

/* One extern per backend. Scalar is ALWAYS linked. */
extern const keystone_kernels_t keystone_kernels_scalar;

#ifdef __SSE4_2__
extern const keystone_kernels_t keystone_kernels_sse42;
#endif
#ifdef __AVX__
extern const keystone_kernels_t keystone_kernels_avx;
#endif
#ifdef __AVX2__
extern const keystone_kernels_t keystone_kernels_avx2;
#endif
#ifdef __AVX512F__
extern const keystone_kernels_t keystone_kernels_avx512;
#endif
#ifdef __ARM_NEON
extern const keystone_kernels_t keystone_kernels_neon;
#endif
#ifdef KEYSTONE_HAVE_VPU
extern const keystone_kernels_t keystone_kernels_vpu;
/* VPU kernel connection functions */
int keystone_vpu_kernel_connect(const char *socket_path);
void keystone_vpu_kernel_disconnect(void);
int keystone_vpu_kernel_is_connected(void);
#endif

/* CUDA is soft-loaded via dlopen, so no extern — the engine calls
 * it through function pointers resolved at runtime. */

#endif /* KEYSTONE_KERNEL_DISPATCH_H */
