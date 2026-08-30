/*
 * keystone_cuda.cu - KEYSTONE Vector Engine CUDA GPU kernel backend
 *
 * Batch cosine/L2/dot similarity on an NVIDIA GPU. Each thread computes
 * the distance between the query and one candidate vector. For the
 * canonical 384-dim embedding this is a single dot product per thread;
 * the kernel uses a small per-thread loop over the 384 floats.
 *
 * Exports both the kernel ABI table (keystone_kernels_cuda) and a set
 * of plain-C lifecycle functions (keystone_cuda_init / _batch_dist /
 * _cleanup / _available) so the engine can dlopen() this .so and bind
 * either surface. Any CUDA failure returns non-zero and the caller
 * falls back to a CPU SIMD backend.
 *
 * Compile with: nvcc -arch=sm_XX -Xcompiler -fPIC -shared
 */

#include "kernel_dispatch.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------
 * Thread-local metric. Default cosine. The engine calls the setter
 * once after creating the backend so all subsequent dist1 calls use
 * the right formula without an ABI change.
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

extern "C" void keystone_cuda_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * CUDA kernel: one thread per candidate.
 *
 * Each thread loads its candidate row and accumulates the dot product
 * (and squared-difference sum for L2) against the query. dim is
 * expected to be 384 for the standard embedding; the loop handles any
 * dim. Results are written to out[blockIdx.x * blockDim.x + threadIdx.x].
 * --------------------------------------------------------------- */
__global__ void keystone_cuda_dist_kernel(const float * __restrict__ query,
                                          const float * __restrict__ cands,
                                          uint32_t n_cands,
                                          uint32_t dim,
                                          int metric,
                                          float * __restrict__ out)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cands)
        return;

    const float *cand = cands + (size_t)idx * dim;

    if (metric == KEYSTONE_METRIC_L2) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float d = query[i] - cand[i];
            acc += d * d;
        }
        out[idx] = acc;
    } else {
        /* COSINE and DOT both need the dot product. */
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            acc += query[i] * cand[i];

        if (metric == KEYSTONE_METRIC_DOT)
            out[idx] = -acc;
        else /* KEYSTONE_METRIC_COSINE */
            out[idx] = 1.0f - acc;
    }
}

/* ---------------------------------------------------------------
 * GPU scratch state. Allocated once in keystone_cuda_init and reused
 * across batch calls. d_query holds the current query vector; d_cands
 * and d_out hold the candidate matrix and output distances. The
 * buffers are grown on demand in keystone_cuda_batch_dist.
 * --------------------------------------------------------------- */
static struct {
    int      ready;
    int      device;
    float   *d_query;   /* dim floats            */
    float   *d_cands;   /* n_cands * dim floats  */
    float   *d_out;     /* n_cands floats        */
    size_t   cands_cap; /* floats allocated in d_cands */
    size_t   out_cap;   /* floats allocated in d_out   */
    uint32_t query_dim; /* dim allocated in d_query    */
} g_cuda = { .ready = 0, .device = -1,
             .d_query = NULL, .d_cands = NULL, .d_out = NULL,
             .cands_cap = 0, .out_cap = 0, .query_dim = 0 };

/* ---------------------------------------------------------------
 * CUDA error -> non-zero int. Every entry point uses this so the
 * caller can treat any failure as "fall back to CPU".
 * --------------------------------------------------------------- */
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t _e = (call);                              \
        if (_e != cudaSuccess)                                \
            return -1;                                        \
    } while (0)

/* ---------------------------------------------------------------
 * Lifecycle
 * --------------------------------------------------------------- */

extern "C" int keystone_cuda_available(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return 0;
    return 1;
}

extern "C" int keystone_cuda_init(void)
{
    if (g_cuda.ready)
        return 0;

    if (!keystone_cuda_available())
        return -1;

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    /* Pre-allocate a query buffer for the standard 384-dim embedding.
     * It will be reallocated if a larger dim is ever used. */
    CUDA_CHECK(cudaMalloc((void **)&g_cuda.d_query, 384 * sizeof(float)));
    g_cuda.query_dim = 384;

    g_cuda.device = dev;
    g_cuda.ready  = 1;
    return 0;
}

extern "C" void keystone_cuda_cleanup(void)
{
    if (g_cuda.d_query) {
        cudaFree(g_cuda.d_query);
        g_cuda.d_query = NULL;
    }
    if (g_cuda.d_cands) {
        cudaFree(g_cuda.d_cands);
        g_cuda.d_cands = NULL;
    }
    if (g_cuda.d_out) {
        cudaFree(g_cuda.d_out);
        g_cuda.d_out = NULL;
    }
    g_cuda.cands_cap = 0;
    g_cuda.out_cap   = 0;
    g_cuda.query_dim = 0;
    g_cuda.device    = -1;
    g_cuda.ready     = 0;

    if (cudaDeviceSynchronize() == cudaSuccess) {
        cudaDeviceReset();
    }
}

/* ---------------------------------------------------------------
 * Ensure the candidate/output scratch buffers are large enough for
 * n_cands * dim floats and n_cands floats respectively. Grows only.
 * Returns 0 on success, -1 on allocation failure.
 * --------------------------------------------------------------- */
static int cuda_ensure_capacity(uint32_t n_cands, uint32_t dim)
{
    size_t cands_need = (size_t)n_cands * dim;
    if (cands_need > g_cuda.cands_cap) {
        if (g_cuda.d_cands)
            cudaFree(g_cuda.d_cands);
        if (cudaMalloc((void **)&g_cuda.d_cands, cands_need * sizeof(float)) != cudaSuccess) {
            g_cuda.d_cands = NULL;
            g_cuda.cands_cap = 0;
            return -1;
        }
        g_cuda.cands_cap = cands_need;
    }

    if (n_cands > g_cuda.out_cap) {
        if (g_cuda.d_out)
            cudaFree(g_cuda.d_out);
        if (cudaMalloc((void **)&g_cuda.d_out, n_cands * sizeof(float)) != cudaSuccess) {
            g_cuda.d_out = NULL;
            g_cuda.out_cap = 0;
            return -1;
        }
        g_cuda.out_cap = n_cands;
    }

    if (dim > g_cuda.query_dim) {
        if (g_cuda.d_query)
            cudaFree(g_cuda.d_query);
        if (cudaMalloc((void **)&g_cuda.d_query, dim * sizeof(float)) != cudaSuccess) {
            g_cuda.d_query = NULL;
            g_cuda.query_dim = 0;
            return -1;
        }
        g_cuda.query_dim = dim;
    }

    return 0;
}

/* ---------------------------------------------------------------
 * Batch distance on the GPU.
 *
 * Uploads the query and candidate matrix, launches one thread per
 * candidate, downloads the resulting distances into out[]. Returns 0
 * on success, non-zero on any CUDA failure (caller falls back to CPU).
 * --------------------------------------------------------------- */
extern "C" int keystone_cuda_batch_dist(const float *query,
                                        const float *cands,
                                        uint32_t n_cands,
                                        uint32_t dim,
                                        float *out)
{
    if (!query || !cands || !out || dim == 0 || n_cands == 0)
        return -1;

    if (!g_cuda.ready)
        return -1;

    if (cuda_ensure_capacity(n_cands, dim) != 0)
        return -1;

    /* Upload query. */
    CUDA_CHECK(cudaMemcpy(g_cuda.d_query, query,
                          dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Upload candidate matrix. */
    CUDA_CHECK(cudaMemcpy(g_cuda.d_cands, cands,
                          (size_t)n_cands * dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Launch: one thread per candidate, 256 threads per block. */
    const int block = 256;
    int grid = (int)((n_cands + block - 1) / block);

    keystone_cuda_dist_kernel<<<grid, block>>>(g_cuda.d_query,
                                               g_cuda.d_cands,
                                               n_cands,
                                               dim,
                                               (int)g_metric,
                                               g_cuda.d_out);

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess)
        return -1;

    /* Wait for the kernel to finish before downloading. */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download distances. */
    CUDA_CHECK(cudaMemcpy(out, g_cuda.d_out,
                          n_cands * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return 0;
}

/* ---------------------------------------------------------------
 * CPU-side dist1 fallback. The GPU is only worthwhile for batches;
 * a single query pays upload/download latency, so dist1 runs on the
 * CPU with a plain scalar loop. This keeps the kernel ABI usable for
 * the engine's single-vector code paths.
 * --------------------------------------------------------------- */
static float cuda_scalar_dist1(const float *a, const float *b, uint32_t dim)
{
    if (!a || !b || dim == 0)
        return 0.0f;

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
 * dist_batch wrapper for the kernel ABI.
 *
 * Tries the GPU path first; on any failure falls back to the scalar
 * CPU loop so the caller always gets correct results. This matches the
 * engine's graceful-degradation contract.
 * --------------------------------------------------------------- */
static void cuda_dist_batch(const float *q,
                            const float *cands,
                            uint32_t n_cands,
                            uint32_t dim,
                            float *out)
{
    if (!q || !cands || !out || dim == 0 || n_cands == 0)
        return;

    if (keystone_cuda_batch_dist(q, cands, n_cands, dim, out) == 0)
        return;

    /* GPU failed: scalar fallback. */
    for (uint32_t i = 0; i < n_cands; i++)
        out[i] = cuda_scalar_dist1(q, cands + (size_t)i * dim, dim);
}

/* ---------------------------------------------------------------
 * normalize_batch - in-place L2 normalization (CPU scalar).
 * Normalization is cheap and not worth a GPU round-trip.
 * Vectors with norm < 1e-12 are left untouched.
 * --------------------------------------------------------------- */
static void cuda_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    if (!vectors || dim == 0 || n == 0)
        return;

    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            sum += vec[i] * vec[i];
        float norm = sqrtf(sum);
        if (norm < 1e-12f)
            continue;
        float inv = 1.0f / norm;
        for (uint32_t i = 0; i < dim; i++)
            vec[i] *= inv;
    }
}

/* ---------------------------------------------------------------
 * Exported kernel table.
 *
 * This lives in a separate .so that the engine dlopen()s at runtime.
 * dist1 and normalize_batch are CPU scalar (single op / cheap op);
 * dist_batch goes through the GPU path with a scalar fallback.
 * --------------------------------------------------------------- */
extern "C" const keystone_kernels_t keystone_kernels_cuda = {
    .dist1           = cuda_scalar_dist1,
    .dist_batch      = cuda_dist_batch,
    .normalize_batch = cuda_normalize_batch,
};
