#include "../include/keystone_trigram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#define TRIGRAM_INITIAL_BUCKETS 65536

/* High-performance FNV-1a hash for 24-bit trigrams into bucket slots */
static inline size_t hash_trigram_key(uint32_t key, size_t num_buckets) {
    uint32_t h = 2166136261u;
    h ^= (key & 0xFF);
    h *= 16777619u;
    h ^= ((key >> 8) & 0xFF);
    h *= 16777619u;
    h ^= ((key >> 16) & 0xFF);
    h *= 16777619u;
    return (size_t)(h & (num_buckets - 1));
}

static int resize_trigram_hash_table(keystone_trigram_index_t* idx) {
    size_t old_num_buckets = idx->num_buckets;
    keystone_trigram_posting_list_t* old_buckets = idx->buckets;

    size_t new_num_buckets = old_num_buckets * 2;
    keystone_trigram_posting_list_t* new_buckets = (keystone_trigram_posting_list_t*)calloc(
        new_num_buckets, sizeof(keystone_trigram_posting_list_t)
    );
    if (!new_buckets) return -1;

    for (size_t i = 0; i < old_num_buckets; i++) {
        if (old_buckets[i].capacity > 0) {
            uint32_t key = old_buckets[i].trigram_key;
            size_t bucket_idx = hash_trigram_key(key, new_num_buckets);
            while (new_buckets[bucket_idx].capacity > 0) {
                bucket_idx = (bucket_idx + 1) & (new_num_buckets - 1);
            }
            new_buckets[bucket_idx] = old_buckets[i];
        }
    }

    free(old_buckets);
    idx->buckets = new_buckets;
    idx->num_buckets = new_num_buckets;
    return 0;
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

keystone_trigram_index_t* keystone_trigram_index_create(size_t initial_doc_capacity) {
    if (initial_doc_capacity == 0) initial_doc_capacity = 64;

    keystone_trigram_index_t* idx = (keystone_trigram_index_t*)calloc(1, sizeof(keystone_trigram_index_t));
    if (!idx) return NULL;

    idx->num_buckets = TRIGRAM_INITIAL_BUCKETS;
    idx->buckets = (keystone_trigram_posting_list_t*)calloc(idx->num_buckets, sizeof(keystone_trigram_posting_list_t));
    if (!idx->buckets) {
        free(idx);
        return NULL;
    }

    idx->doc_capacity = initial_doc_capacity;
    idx->docs = (keystone_trigram_doc_t*)calloc(idx->doc_capacity, sizeof(keystone_trigram_doc_t));
    if (!idx->docs) {
        free(idx->buckets);
        free(idx);
        return NULL;
    }

    idx->doc_count = 0;
    idx->unique_trigrams = 0;
    idx->is_finalized = false;
    return idx;
}

void keystone_trigram_index_destroy(keystone_trigram_index_t* idx) {
    if (!idx) return;

    if (idx->buckets) {
        for (size_t i = 0; i < idx->num_buckets; i++) {
            if (idx->buckets[i].doc_ids) {
                free(idx->buckets[i].doc_ids);
            }
        }
        free(idx->buckets);
    }

    if (idx->docs) {
        for (size_t i = 0; i < idx->doc_count; i++) {
            if (idx->docs[i].name) free(idx->docs[i].name);
        }
        free(idx->docs);
    }

    free(idx);
}

size_t keystone_trigram_extract(
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_trigrams,
    size_t max_trigrams
) {
    if (!pattern || pattern_len < 3 || !out_trigrams || max_trigrams == 0) return 0;

    const unsigned char* p = (const unsigned char*)pattern;
    size_t extracted = 0;

    for (size_t i = 0; i <= pattern_len - 3; i++) {
        uint32_t key = ((uint32_t)p[i] << 16) | ((uint32_t)p[i+1] << 8) | (uint32_t)p[i+2];

        /* Deduplicate in output array */
        bool duplicate = false;
        for (size_t j = 0; j < extracted; j++) {
            if (out_trigrams[j] == key) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            out_trigrams[extracted++] = key;
            if (extracted >= max_trigrams) break;
        }
    }

    return extracted;
}

static keystone_trigram_posting_list_t* find_or_create_posting_list(
    keystone_trigram_index_t* idx,
    uint32_t key
) {
    /* Check load factor (resize if >= 70% full) */
    if (idx->unique_trigrams * 10 >= idx->num_buckets * 7) {
        if (resize_trigram_hash_table(idx) != 0) {
            /* If resize fails, continue with current capacity if possible */
        }
    }

    size_t bucket_idx = hash_trigram_key(key, idx->num_buckets);
    size_t mask = idx->num_buckets - 1;
    size_t original = bucket_idx;

    /* Linear probing */
    while (idx->buckets[bucket_idx].capacity > 0 && idx->buckets[bucket_idx].trigram_key != key) {
        bucket_idx = (bucket_idx + 1) & mask;
        if (bucket_idx == original) return NULL; /* Table full */
    }

    keystone_trigram_posting_list_t* plist = &idx->buckets[bucket_idx];
    if (plist->capacity == 0) {
        /* New entry */
        plist->trigram_key = key;
        plist->capacity = 8;
        plist->count = 0;
        plist->doc_ids = (uint32_t*)malloc(plist->capacity * sizeof(uint32_t));
        if (!plist->doc_ids) return NULL;
        idx->unique_trigrams++;
    }

    return plist;
}

static void add_doc_to_posting_list(keystone_trigram_posting_list_t* plist, uint32_t doc_id) {
    /* Check if already appended for this doc */
    if (plist->count > 0 && plist->doc_ids[plist->count - 1] == doc_id) {
        return;
    }

    if (plist->count >= plist->capacity) {
        size_t new_cap = plist->capacity * 2;
        uint32_t* new_ids = (uint32_t*)realloc(plist->doc_ids, new_cap * sizeof(uint32_t));
        if (!new_ids) return;
        plist->doc_ids = new_ids;
        plist->capacity = new_cap;
    }

    plist->doc_ids[plist->count++] = doc_id;
}

int keystone_trigram_index_add_document(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
) {
    if (!idx || !text) return -1;

    uint64_t start_time = get_time_ns();

    if (idx->doc_count >= idx->doc_capacity) {
        size_t new_cap = idx->doc_capacity * 2;
        keystone_trigram_doc_t* new_docs = (keystone_trigram_doc_t*)realloc(idx->docs, new_cap * sizeof(keystone_trigram_doc_t));
        if (!new_docs) return -1;
        idx->docs = new_docs;
        idx->doc_capacity = new_cap;
    }

    uint32_t doc_id = (uint32_t)idx->doc_count;
    keystone_trigram_doc_t* doc = &idx->docs[doc_id];
    doc->id = doc_id;
    doc->name = name ? strdup(name) : NULL;
    doc->content = text;
    doc->content_len = text_len;
    idx->doc_count++;

    if (text_len >= 3) {
        const unsigned char* p = (const unsigned char*)text;
        for (size_t i = 0; i <= text_len - 3; i++) {
            uint32_t key = ((uint32_t)p[i] << 16) | ((uint32_t)p[i+1] << 8) | (uint32_t)p[i+2];
            keystone_trigram_posting_list_t* plist = find_or_create_posting_list(idx, key);
            if (plist) {
                add_doc_to_posting_list(plist, doc_id);
            }
        }
    }

    idx->stats.total_documents = idx->doc_count;
    idx->stats.bytes_indexed += text_len;
    idx->stats.build_time_ns += (get_time_ns() - start_time);

    if (out_doc_id) *out_doc_id = doc_id;
    return 0;
}

static int compare_uint32(const void* a, const void* b) {
    uint32_t u1 = *(const uint32_t*)a;
    uint32_t u2 = *(const uint32_t*)b;
    return (u1 > u2) - (u1 < u2);
}

int keystone_trigram_index_finalize(keystone_trigram_index_t* idx) {
    if (!idx) return -1;

    size_t total_postings = 0;

    for (size_t i = 0; i < idx->num_buckets; i++) {
        keystone_trigram_posting_list_t* plist = &idx->buckets[i];
        if (plist->count > 0) {
            /* Sort doc_ids for faster intersection */
            qsort(plist->doc_ids, plist->count, sizeof(uint32_t), compare_uint32);
            total_postings += plist->count;
        }
    }

    idx->stats.unique_trigrams = idx->unique_trigrams;
    idx->stats.total_postings = total_postings;
    idx->is_finalized = true;
    return 0;
}

static const keystone_trigram_posting_list_t* get_posting_list(
    const keystone_trigram_index_t* idx,
    uint32_t key
) {
    size_t mask = idx->num_buckets - 1;
    size_t bucket_idx = hash_trigram_key(key, idx->num_buckets);
    size_t original = bucket_idx;

    while (idx->buckets[bucket_idx].capacity > 0) {
        if (idx->buckets[bucket_idx].trigram_key == key) {
            return &idx->buckets[bucket_idx];
        }
        bucket_idx = (bucket_idx + 1) & mask;
        if (bucket_idx == original) break;
    }

    return NULL;
}

size_t keystone_trigram_index_get_candidates(
    const keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_candidates,
    size_t max_candidates
) {
    if (!idx || !pattern || !out_candidates || max_candidates == 0) return 0;

    if (pattern_len < 3) {
        /* Pattern too short for trigram filtering: return all document IDs */
        size_t count = idx->doc_count < max_candidates ? idx->doc_count : max_candidates;
        for (size_t i = 0; i < count; i++) {
            out_candidates[i] = (uint32_t)i;
        }
        return count;
    }

    uint32_t query_trigrams[64];
    size_t num_trigrams = keystone_trigram_extract(pattern, pattern_len, query_trigrams, 64);
    if (num_trigrams == 0) return 0;

    /* Fetch posting lists for query trigrams */
    const keystone_trigram_posting_list_t* lists[64];
    size_t num_lists = 0;

    for (size_t i = 0; i < num_trigrams; i++) {
        const keystone_trigram_posting_list_t* plist = get_posting_list(idx, query_trigrams[i]);
        if (!plist || plist->count == 0) {
            /* If any required trigram is missing, no documents can match! */
            return 0;
        }
        lists[num_lists++] = plist;
    }

    /* Sort posting lists by size ascending to minimize intersection work */
    for (size_t i = 0; i < num_lists; i++) {
        for (size_t j = i + 1; j < num_lists; j++) {
            if (lists[j]->count < lists[i]->count) {
                const keystone_trigram_posting_list_t* tmp = lists[i];
                lists[i] = lists[j];
                lists[j] = tmp;
            }
        }
    }

    /* Intersect smallest list against the rest */
    const keystone_trigram_posting_list_t* base = lists[0];
    size_t candidate_count = 0;

    for (size_t i = 0; i < base->count; i++) {
        uint32_t doc_id = base->doc_ids[i];
        bool in_all = true;

        for (size_t l = 1; l < num_lists; l++) {
            const keystone_trigram_posting_list_t* plist = lists[l];
            /* Binary search doc_id in sorted plist->doc_ids */
            uint32_t* found = (uint32_t*)bsearch(&doc_id, plist->doc_ids, plist->count, sizeof(uint32_t), compare_uint32);
            if (!found) {
                in_all = false;
                break;
            }
        }

        if (in_all) {
            out_candidates[candidate_count++] = doc_id;
            if (candidate_count >= max_candidates) break;
        }
    }

    return candidate_count;
}

size_t keystone_trigram_index_search(
    keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_matches,
    size_t max_matches
) {
    if (!idx || !pattern || pattern_len == 0 || !out_matches || max_matches == 0) return 0;

    idx->stats.total_searches++;

    uint32_t* candidates = (uint32_t*)malloc(idx->doc_count * sizeof(uint32_t));
    if (!candidates) return 0;

    size_t num_candidates = keystone_trigram_index_get_candidates(
        idx, pattern, pattern_len, candidates, idx->doc_count
    );

    idx->stats.candidate_docs_evaluated += num_candidates;
    idx->stats.candidate_docs_rejected += (idx->doc_count - num_candidates);

    size_t matches = 0;
    for (size_t i = 0; i < num_candidates; i++) {
        uint32_t doc_id = candidates[i];
        keystone_trigram_doc_t* doc = &idx->docs[doc_id];

        if (doc->content && doc->content_len >= pattern_len) {
            if (memmem(doc->content, doc->content_len, pattern, pattern_len) != NULL) {
                out_matches[matches++] = doc_id;
                if (matches >= max_matches) break;
            }
        }
    }

    free(candidates);
    return matches;
}

void keystone_trigram_index_get_stats(
    const keystone_trigram_index_t* idx,
    keystone_trigram_stats_t* stats
) {
    if (idx && stats) {
        *stats = idx->stats;
    }
}
