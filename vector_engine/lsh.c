/*
 * lsh.c - Locality-Sensitive Hashing coarse index implementation
 *
 * Random hyperplane LSH for cosine similarity. Each hash table uses
 * a random Gaussian projection matrix (hash_bits x dim). The sign of
 * each projection dot product gives one bit of the hash. Vectors with
 * high cosine similarity collide in the same bucket with probability
 * p = 1 - theta/pi, where theta is the angle between them.
 */
#include "lsh.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Box-Muller transform for Gaussian random numbers */
static float gaussian_random(void) {
    /* Use static state — not thread-safe but LSH projection is
     * generated once at creation time, not concurrently */
    static int have_spare = 0;
    static float spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s < 1e-12f);

    float mul = sqrtf(-2.0f * logf(s) / s);
    spare = v * mul;
    have_spare = 1;
    return u * mul;
}

/* Generate a random Gaussian projection matrix */
static float *generate_projection(uint32_t hash_bits, uint32_t dim) {
    float *proj = (float *)malloc((size_t)hash_bits * dim * sizeof(float));
    if (!proj) return NULL;
    for (uint32_t i = 0; i < hash_bits * dim; i++) {
        proj[i] = gaussian_random() / sqrtf((float)dim);
    }
    return proj;
}

/* Compute one hash bit: sign of dot(vector, projection_row) with 4-way unrolling */
static int sign_dot(const float *vec, const float *proj_row, uint32_t dim) {
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    uint32_t i = 0;
    uint32_t limit = dim & ~3u;
    for (; i < limit; i += 4) {
        dot0 += vec[i + 0] * proj_row[i + 0];
        dot1 += vec[i + 1] * proj_row[i + 1];
        dot2 += vec[i + 2] * proj_row[i + 2];
        dot3 += vec[i + 3] * proj_row[i + 3];
    }
    float dot = (dot0 + dot1) + (dot2 + dot3);
    for (; i < dim; i++) {
        dot += vec[i] * proj_row[i];
    }
    return (dot >= 0.0f) ? 1 : 0;
}

typedef struct {
    int64_t key;
    uint32_t offset;
    uint32_t count;
} lsh_bucket_entry_t;

static int cmp_bucket_entries(const void* a, const void* b) {
    const lsh_bucket_entry_t* ea = (const lsh_bucket_entry_t*)a;
    const lsh_bucket_entry_t* eb = (const lsh_bucket_entry_t*)b;
    if (ea->key < eb->key) return -1;
    if (ea->key > eb->key) return 1;
    return 0;
}

keystone_error_t keystone_lsh_create(keystone_lsh_index_t **idx,
                                      uint32_t dim,
                                      uint32_t num_tables,
                                      uint32_t hash_bits,
                                      uint32_t probes) {
    if (!idx || dim == 0 || num_tables == 0 || hash_bits == 0 || hash_bits > 48) {
        return KEYSTONE_ERR_PARAM;
    }

    keystone_lsh_index_t *lsh = (keystone_lsh_index_t *)calloc(1, sizeof(*lsh));
    if (!lsh) return KEYSTONE_ERR_OOM;

    lsh->num_tables = num_tables;
    lsh->hash_bits = hash_bits;
    lsh->dim = dim;
    lsh->probes = probes;

    lsh->tables = (keystone_lsh_table_t *)calloc(num_tables, sizeof(keystone_lsh_table_t));
    if (!lsh->tables) {
        free(lsh);
        return KEYSTONE_ERR_OOM;
    }

    for (uint32_t t = 0; t < num_tables; t++) {
        keystone_lsh_table_t *tbl = &lsh->tables[t];
        tbl->hash_bits = hash_bits;
        tbl->dim = dim;
        tbl->projection = generate_projection(hash_bits, dim);
        if (!tbl->projection) {
            keystone_lsh_destroy(lsh);
            return KEYSTONE_ERR_OOM;
        }
        /* Initial bucket capacity — will grow as needed */
        tbl->bucket_capacity = 4096;
        tbl->bucket_keys = (int64_t *)malloc(tbl->bucket_capacity * sizeof(int64_t));
        tbl->bucket_offsets = (uint32_t *)malloc(tbl->bucket_capacity * sizeof(uint32_t));
        tbl->bucket_counts = (uint32_t *)malloc(tbl->bucket_capacity * sizeof(uint32_t));
        /* Indices array: starts at 64K entries, grows with need */
        tbl->indices = (uint32_t *)malloc(65536 * sizeof(uint32_t));
        if (!tbl->bucket_keys || !tbl->bucket_offsets || !tbl->bucket_counts || !tbl->indices) {
            keystone_lsh_destroy(lsh);
            return KEYSTONE_ERR_OOM;
        }
        tbl->n_buckets = 0;
        tbl->n_vectors = 0;
    }

    *idx = lsh;
    return KEYSTONE_OK;
}

void keystone_lsh_destroy(keystone_lsh_index_t *idx) {
    if (!idx) return;
    for (uint32_t t = 0; t < idx->num_tables; t++) {
        keystone_lsh_table_t *tbl = &idx->tables[t];
        free(tbl->projection);
        free(tbl->bucket_keys);
        free(tbl->bucket_offsets);
        free(tbl->bucket_counts);
        free(tbl->indices);
    }
    free(idx->tables);
    free(idx);
}

void keystone_lsh_clear(keystone_lsh_index_t *idx) {
    if (!idx) return;
    for (uint32_t t = 0; t < idx->num_tables; t++) {
        keystone_lsh_table_t *tbl = &idx->tables[t];
        tbl->n_buckets = 0;
        tbl->n_vectors = 0;
    }
}

int64_t keystone_lsh_hash(const keystone_lsh_index_t *idx,
                           uint32_t table_idx,
                           const float *vector) {
    if (!idx || table_idx >= idx->num_tables || !vector) return 0;

    const keystone_lsh_table_t *tbl = &idx->tables[table_idx];
    int64_t key = 0;
    for (uint32_t b = 0; b < tbl->hash_bits; b++) {
        const float *row = &tbl->projection[(size_t)b * tbl->dim];
        if (sign_dot(vector, row, tbl->dim)) {
            key |= (int64_t)1 << b;
        }
    }
    return key;
}

/* Find or create a bucket in a table. Returns bucket index. */
static uint32_t find_or_create_bucket(keystone_lsh_table_t *tbl, int64_t key) {
    /* Linear search for now — replaced with sorted lookup after finalize */
    for (uint32_t i = 0; i < tbl->n_buckets; i++) {
        if (tbl->bucket_keys[i] == key) return i;
    }

    /* New bucket — check capacity */
    if (tbl->n_buckets >= tbl->bucket_capacity) {
        uint32_t new_cap = tbl->bucket_capacity * 2;
        int64_t *new_keys = (int64_t *)realloc(tbl->bucket_keys, new_cap * sizeof(int64_t));
        uint32_t *new_offs = (uint32_t *)realloc(tbl->bucket_offsets, new_cap * sizeof(uint32_t));
        uint32_t *new_counts = (uint32_t *)realloc(tbl->bucket_counts, new_cap * sizeof(uint32_t));
        if (!new_keys || !new_offs || !new_counts) {
            free(new_keys); free(new_offs); free(new_counts);
            return (uint32_t)-1;
        }
        tbl->bucket_keys = new_keys;
        tbl->bucket_offsets = new_offs;
        tbl->bucket_counts = new_counts;
        tbl->bucket_capacity = new_cap;
    }

    uint32_t idx = tbl->n_buckets++;
    tbl->bucket_keys[idx] = key;
    tbl->bucket_offsets[idx] = tbl->n_vectors;
    tbl->bucket_counts[idx] = 0;
    return idx;
}

keystone_error_t keystone_lsh_insert(keystone_lsh_index_t *idx,
                                      const float *vector,
                                      uint32_t vec_index) {
    if (!idx || !vector) return KEYSTONE_ERR_NULL;

    for (uint32_t t = 0; t < idx->num_tables; t++) {
        keystone_lsh_table_t *tbl = &idx->tables[t];
        int64_t key = keystone_lsh_hash(idx, t, vector);
        uint32_t bucket = find_or_create_bucket(tbl, key);
        if (bucket == (uint32_t)-1) return KEYSTONE_ERR_OOM;

        /* Append vec_index to bucket */
        uint32_t offset = tbl->bucket_offsets[bucket] + tbl->bucket_counts[bucket];
        /* Check indices array capacity — needs to hold n_vectors+1 entries */
        size_t needed = (size_t)tbl->n_vectors + 1;
        size_t capacity = (size_t)tbl->bucket_capacity * 16;
        if (needed > capacity) {
            /* Grow to accommodate, with 2x growth factor */
            size_t new_cap = needed * 2;
            uint32_t *new_indices = (uint32_t *)realloc(tbl->indices, new_cap * sizeof(uint32_t));
            if (!new_indices) return KEYSTONE_ERR_OOM;
            tbl->indices = new_indices;
            /* Update capacity tracking — use n_vectors as the real capacity marker */
        }
        tbl->indices[offset] = vec_index;
        tbl->bucket_counts[bucket]++;
        tbl->n_vectors++;
    }
    return KEYSTONE_OK;
}

keystone_error_t keystone_lsh_insert_batch(keystone_lsh_index_t *idx,
                                            const float *vectors,
                                            uint32_t n,
                                            uint32_t start_index) {
    if (!idx || !vectors) return KEYSTONE_ERR_NULL;
    for (uint32_t i = 0; i < n; i++) {
        const float *vec = &vectors[(size_t)i * idx->dim];
        keystone_error_t rc = keystone_lsh_insert(idx, vec, start_index + i);
        if (rc != KEYSTONE_OK) return rc;
    }
    return KEYSTONE_OK;
}

/* Comparison for sorting bucket keys (used by qsort if needed) */
static int cmp_bucket_keys(const void *a, const void *b) {
    int64_t ka = *(const int64_t *)a;
    int64_t kb = *(const int64_t *)b;
    if (ka < kb) return -1;
    if (ka > kb) return 1;
    return 0;
}
__attribute__((unused)) static int (*ks_cmp)(const void *, const void *) = cmp_bucket_keys;

keystone_error_t keystone_lsh_finalize(keystone_lsh_index_t *idx) {
    if (!idx) return KEYSTONE_ERR_NULL;

    for (uint32_t t = 0; t < idx->num_tables; t++) {
        keystone_lsh_table_t *tbl = &idx->tables[t];
        if (tbl->n_buckets <= 1) continue;

        uint32_t n = tbl->n_buckets;
        lsh_bucket_entry_t *entries = (lsh_bucket_entry_t *)malloc(n * sizeof(lsh_bucket_entry_t));
        if (!entries) {
            return KEYSTONE_ERR_OOM;
        }
        for (uint32_t i = 0; i < n; i++) {
            entries[i].key = tbl->bucket_keys[i];
            entries[i].offset = tbl->bucket_offsets[i];
            entries[i].count = tbl->bucket_counts[i];
        }

        /* O(N log N) quicksort on packed contiguous struct array */
        qsort(entries, n, sizeof(lsh_bucket_entry_t), cmp_bucket_entries);

        for (uint32_t i = 0; i < n; i++) {
            tbl->bucket_keys[i] = entries[i].key;
            tbl->bucket_offsets[i] = entries[i].offset;
            tbl->bucket_counts[i] = entries[i].count;
        }
        free(entries);
    }
    return KEYSTONE_OK;
}

/* Binary search for a bucket key in sorted array */
static int32_t find_bucket(const keystone_lsh_table_t *tbl, int64_t key) {
    int32_t lo = 0, hi = (int32_t)tbl->n_buckets - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (tbl->bucket_keys[mid] == key) return mid;
        if (tbl->bucket_keys[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Multiprobe: generate neighboring bucket keys by flipping bits */
static void multiprobe_keys(int64_t base_key, uint32_t hash_bits,
                            uint32_t probes, int64_t *out_keys, uint32_t *out_count) {
    *out_count = 0;
    if (probes == 0) return;

    /* Flip the lowest-significance bits first (simplest multiprobe) */
    for (uint32_t p = 0; p < probes && p < hash_bits; p++) {
        out_keys[*out_count] = base_key ^ ((int64_t)1 << p);
        (*out_count)++;
    }
}

keystone_error_t keystone_lsh_query(const keystone_lsh_index_t *idx,
                                     const float *query,
                                     uint32_t *out_indices,
                                     uint32_t max_candidates,
                                     uint32_t *out_count) {
    if (!idx || !query || !out_indices || !out_count) return KEYSTONE_ERR_NULL;

    *out_count = 0;
    uint32_t collected = 0;
    uint64_t seen_filter = 0;

    /* For each table, find the bucket and collect candidates */
    for (uint32_t t = 0; t < idx->num_tables; t++) {
        const keystone_lsh_table_t *tbl = &idx->tables[t];
        int64_t base_key = keystone_lsh_hash(idx, t, query);

        /* Generate probe keys (base + multiprobe neighbors) */
        int64_t probe_keys[64];
        uint32_t n_probes = 0;
        probe_keys[n_probes++] = base_key;
        multiprobe_keys(base_key, tbl->hash_bits, idx->probes,
                        probe_keys + 1, &n_probes);

        for (uint32_t p = 0; p < n_probes; p++) {
            int32_t bucket = find_bucket(tbl, probe_keys[p]);
            if (bucket < 0) continue;

            uint32_t offset = tbl->bucket_offsets[bucket];
            uint32_t count = tbl->bucket_counts[bucket];
            for (uint32_t i = 0; i < count && collected < max_candidates; i++) {
                /* Dedup: 64-bit quick bloom filter check before linear search */
                uint32_t vec_idx = tbl->indices[offset + i];
                uint64_t bit = (uint64_t)1 << (vec_idx & 63);
                if (!(seen_filter & bit)) {
                    seen_filter |= bit;
                    out_indices[collected++] = vec_idx;
                } else {
                    int found = 0;
                    for (uint32_t j = 0; j < collected; j++) {
                        if (out_indices[j] == vec_idx) { found = 1; break; }
                    }
                    if (!found) {
                        out_indices[collected++] = vec_idx;
                    }
                }
            }
            if (collected >= max_candidates) break;
        }
        if (collected >= max_candidates) break;
    }

    *out_count = collected;
    return KEYSTONE_OK;
}
