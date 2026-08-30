/*
 * lsh.h - Locality-Sensitive Hashing coarse index for vector engine
 *
 * Random hyperplane LSH: hashes 384-dim float32 vectors to int64
 * bucket keys via random Gaussian projection. Vectors with high
 * cosine similarity land in the same bucket with high probability.
 *
 * The bucket keys are then looked up using KEYSTONE's existing
 * interpolation search (keystone_anchor_table_t) for O(1) bucket
 * retrieval, and candidates are exact-reranked using the SIMD kernels.
 */
#ifndef KEYSTONE_LSH_H
#define KEYSTONE_LSH_H

#include "keystone_vector_engine.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A single LSH hash table (one random projection matrix) */
typedef struct {
    /* Projection matrix: lsh_hash_bits x dim, row-major float32 */
    float *projection;
    uint32_t hash_bits;
    uint32_t dim;

    /* Bucket map: bucket_key -> list of vector indices.
     * Implemented as a sorted array of (bucket_key, offset, count)
     * triples for KEYSTONE-accelerated lookup. */
    int64_t *bucket_keys;    /* sorted unique bucket keys */
    uint32_t *bucket_offsets;/* offset into indices array */
    uint32_t *bucket_counts; /* number of vectors in bucket */
    uint32_t *indices;       /* vector indices, grouped by bucket */
    uint32_t n_buckets;
    uint32_t n_vectors;
    uint32_t bucket_capacity;
} keystone_lsh_table_t;

/* LSH index: collection of hash tables for multiprobe */
typedef struct {
    keystone_lsh_table_t *tables;
    uint32_t num_tables;
    uint32_t hash_bits;
    uint32_t dim;
    uint32_t probes;         /* multiprobe count */
} keystone_lsh_index_t;

/* Create LSH index with given parameters */
keystone_error_t keystone_lsh_create(keystone_lsh_index_t **idx,
                                      uint32_t dim,
                                      uint32_t num_tables,
                                      uint32_t hash_bits,
                                      uint32_t probes);

void keystone_lsh_destroy(keystone_lsh_index_t *idx);

/* Compute LSH bucket key for a vector in one table.
 * Returns a hash_bits-bit integer packed into int64. */
int64_t keystone_lsh_hash(const keystone_lsh_index_t *idx,
                           uint32_t table_idx,
                           const float *vector);

/* Insert a vector (by its index in the main store) into all LSH tables */
keystone_error_t keystone_lsh_insert(keystone_lsh_index_t *idx,
                                      const float *vector,
                                      uint32_t vec_index);

/* Batch insert: insert n vectors into all LSH tables */
keystone_error_t keystone_lsh_insert_batch(keystone_lsh_index_t *idx,
                                            const float *vectors,
                                            uint32_t n,
                                            uint32_t start_index);

/* Query: find candidate vector indices for a query vector.
 * Returns up to max_candidates indices in out_indices, and the
 * actual count in out_count. Uses multiprobe to explore nearby buckets. */
keystone_error_t keystone_lsh_query(const keystone_lsh_index_t *idx,
                                     const float *query,
                                     uint32_t *out_indices,
                                     uint32_t max_candidates,
                                     uint32_t *out_count);

/* Finalize: sort all bucket keys for KEYSTONE-accelerated lookup */
keystone_error_t keystone_lsh_finalize(keystone_lsh_index_t *idx);

/* Clear all tables (for rebuild) */
void keystone_lsh_clear(keystone_lsh_index_t *idx);

#ifdef __cplusplus
}
#endif

#endif /* KEYSTONE_LSH_H */
