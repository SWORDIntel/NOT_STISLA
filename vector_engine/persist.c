/*
 * persist.c - Save/load vector engine state to/from disk
 *
 * Binary format:
 *   [Header: magic, version, dim, count, capacity, metric, config...]
 *   [IDs: count * uint64]
 *   [Vectors: count * dim * float32]
 *   [LSH: num_tables, hash_bits, probes, then per-table projection + buckets]
 */
#include "keystone_vector_engine.h"
#include "kernel_dispatch.h"
#include "lsh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KVE_MAGIC 0x4B564556u  /* "KVEV" */
#define KVE_VERSION 1

/* We need access to the engine struct internals.
 * This is a bit of a hack — we duplicate the struct layout here.
 * In production, the struct would be in a private header. */
struct keystone_vec_engine {
    keystone_config_t cfg;
    float    *vectors;
    uint64_t *ids;
    uint32_t  count;
    uint32_t  capacity;
    uint32_t  dim;
    uint64_t *sorted_ids;
    uint32_t *sorted_indices;
    int       ids_sorted;
    keystone_lsh_index_t *lsh;
    int       lsh_finalized;
    const keystone_kernels_t *kernels;
    keystone_backend_t backend;
    void *cuda_handle;
    int (*cuda_init)(void);
    int (*cuda_batch_dist)(const float *, const float *, uint32_t, uint32_t, float *);
    void (*cuda_cleanup)(void);
    int (*cuda_available)(void);
    int cuda_initialized;
#ifdef KEYSTONE_HAVE_VPU
    int vpu_connected;
    char vpu_socket[108];
#endif
};

keystone_error_t keystone_vec_save(const keystone_vec_engine_t *e,
                                    const char *path) {
    if (!e || !path) return KEYSTONE_ERR_NULL;

    FILE *f = fopen(path, "wb");
    if (!f) return KEYSTONE_ERR_IO;

    /* Header */
    uint32_t magic = KVE_MAGIC;
    uint32_t version = KVE_VERSION;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1) goto io_err;
    if (fwrite(&version, sizeof(version), 1, f) != 1) goto io_err;
    if (fwrite(&e->dim, sizeof(e->dim), 1, f) != 1) goto io_err;
    if (fwrite(&e->count, sizeof(e->count), 1, f) != 1) goto io_err;
    if (fwrite(&e->capacity, sizeof(e->capacity), 1, f) != 1) goto io_err;
    if (fwrite(&e->cfg, sizeof(e->cfg), 1, f) != 1) goto io_err;

    /* IDs */
    if (e->count > 0) {
        if (fwrite(e->ids, sizeof(uint64_t), e->count, f) != e->count) goto io_err;
        /* Vectors */
        if (fwrite(e->vectors, sizeof(float), (size_t)e->count * e->dim, f)
            != (size_t)e->count * e->dim) goto io_err;
    }

    /* LSH state */
    if (e->lsh) {
        uint32_t has_lsh = 1;
        if (fwrite(&has_lsh, sizeof(has_lsh), 1, f) != 1) goto io_err;
        if (fwrite(&e->lsh->num_tables, sizeof(e->lsh->num_tables), 1, f) != 1) goto io_err;
        if (fwrite(&e->lsh->hash_bits, sizeof(e->lsh->hash_bits), 1, f) != 1) goto io_err;
        if (fwrite(&e->lsh->dim, sizeof(e->lsh->dim), 1, f) != 1) goto io_err;
        if (fwrite(&e->lsh->probes, sizeof(e->lsh->probes), 1, f) != 1) goto io_err;

        for (uint32_t t = 0; t < e->lsh->num_tables; t++) {
            keystone_lsh_table_t *tbl = &e->lsh->tables[t];
            /* Projection matrix */
            if (fwrite(tbl->projection, sizeof(float),
                       (size_t)tbl->hash_bits * tbl->dim, f)
                != (size_t)tbl->hash_bits * tbl->dim) goto io_err;
            /* Bucket data */
            if (fwrite(&tbl->n_buckets, sizeof(tbl->n_buckets), 1, f) != 1) goto io_err;
            if (fwrite(&tbl->n_vectors, sizeof(tbl->n_vectors), 1, f) != 1) goto io_err;
            if (tbl->n_buckets > 0) {
                if (fwrite(tbl->bucket_keys, sizeof(int64_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto io_err;
                if (fwrite(tbl->bucket_offsets, sizeof(uint32_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto io_err;
                if (fwrite(tbl->bucket_counts, sizeof(uint32_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto io_err;
            }
            if (tbl->n_vectors > 0) {
                if (fwrite(tbl->indices, sizeof(uint32_t), tbl->n_vectors, f)
                    != tbl->n_vectors) goto io_err;
            }
        }
    } else {
        uint32_t has_lsh = 0;
        if (fwrite(&has_lsh, sizeof(has_lsh), 1, f) != 1) goto io_err;
    }

    fclose(f);
    return KEYSTONE_OK;

io_err:
    fclose(f);
    return KEYSTONE_ERR_IO;
}

keystone_error_t keystone_vec_load(keystone_vec_engine_t **out,
                                    const char *path) {
    if (!out || !path) return KEYSTONE_ERR_NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return KEYSTONE_ERR_IO;

    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != KVE_MAGIC) {
        fclose(f); return KEYSTONE_ERR_IO;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != KVE_VERSION) {
        fclose(f); return KEYSTONE_ERR_IO;
    }

    uint32_t dim, count, capacity;
    if (fread(&dim, sizeof(dim), 1, f) != 1) { fclose(f); return KEYSTONE_ERR_IO; }
    if (fread(&count, sizeof(count), 1, f) != 1) { fclose(f); return KEYSTONE_ERR_IO; }
    if (fread(&capacity, sizeof(capacity), 1, f) != 1) { fclose(f); return KEYSTONE_ERR_IO; }

    keystone_config_t cfg;
    if (fread(&cfg, sizeof(cfg), 1, f) != 1) { fclose(f); return KEYSTONE_ERR_IO; }

    /* Create engine with loaded config */
    keystone_error_t rc = keystone_vec_engine_create(&cfg, out);
    if (rc != KEYSTONE_OK) { fclose(f); return rc; }

    keystone_vec_engine_t *e = *out;

    /* Read IDs and vectors */
    if (count > 0) {
        if (fread(e->ids, sizeof(uint64_t), count, f) != count) {
            keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
        }
        if (fread(e->vectors, sizeof(float), (size_t)count * dim, f)
            != (size_t)count * dim) {
            keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
        }
        e->count = count;
        e->ids_sorted = 0;
    }

    /* Read LSH state */
    uint32_t has_lsh;
    if (fread(&has_lsh, sizeof(has_lsh), 1, f) != 1) {
        keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
    }

    if (has_lsh && e->lsh) {
        /* Destroy existing LSH and rebuild from saved state */
        keystone_lsh_destroy(e->lsh);
        e->lsh = NULL;

        uint32_t num_tables, hash_bits, lsh_dim, probes;
        if (fread(&num_tables, sizeof(num_tables), 1, f) != 1 ||
            fread(&hash_bits, sizeof(hash_bits), 1, f) != 1 ||
            fread(&lsh_dim, sizeof(lsh_dim), 1, f) != 1 ||
            fread(&probes, sizeof(probes), 1, f) != 1) {
            keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
        }

        rc = keystone_lsh_create(&e->lsh, lsh_dim, num_tables, hash_bits, probes);
        if (rc != KEYSTONE_OK) {
            keystone_vec_engine_destroy(e); fclose(f); return rc;
        }

        for (uint32_t t = 0; t < num_tables; t++) {
            keystone_lsh_table_t *tbl = &e->lsh->tables[t];
            if (fread(tbl->projection, sizeof(float),
                      (size_t)hash_bits * lsh_dim, f)
                != (size_t)hash_bits * lsh_dim) {
                keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
            }
            if (fread(&tbl->n_buckets, sizeof(tbl->n_buckets), 1, f) != 1 ||
                fread(&tbl->n_vectors, sizeof(tbl->n_vectors), 1, f) != 1) {
                keystone_vec_engine_destroy(e); fclose(f); return KEYSTONE_ERR_IO;
            }
            if (tbl->n_buckets > 0) {
                if (fread(tbl->bucket_keys, sizeof(int64_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto lsh_err;
                if (fread(tbl->bucket_offsets, sizeof(uint32_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto lsh_err;
                if (fread(tbl->bucket_counts, sizeof(uint32_t), tbl->n_buckets, f)
                    != tbl->n_buckets) goto lsh_err;
            }
            if (tbl->n_vectors > 0) {
                if (fread(tbl->indices, sizeof(uint32_t), tbl->n_vectors, f)
                    != tbl->n_vectors) goto lsh_err;
            }
        }
        e->lsh_finalized = 1;
    }

    fclose(f);
    return KEYSTONE_OK;

lsh_err:
    keystone_vec_engine_destroy(e);
    fclose(f);
    return KEYSTONE_ERR_IO;
}
