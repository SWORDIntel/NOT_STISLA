#include "../include/keystone_trigram.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* SSE4.2 for SIMD-accelerated string scanning */
#ifdef __SSE4_2__
#include <nmmintrin.h>
#define HAVE_SSE42 1
#else
#define HAVE_SSE42 0
#endif

#define TRIGRAM_INITIAL_BUCKETS 65536u
#define TRIGRAM_INITIAL_POSTING_CAPACITY 8u
#define TRIGRAM_QUERY_MAX_UNIQUE 64u
/* 24-bit trigram space = 16M possible values. 2MB bitmap for dedup. */
#define TRIGRAM_BITMAP_BYTES (1u << 21) /* 2MB = 16M bits */
/* Sentinel for empty hash buckets. Outside the 24-bit trigram key range. */
#define TRIGRAM_KEY_EMPTY 0xFFFFFFFFu

typedef struct keystone_trigram_posting_list {
    uint32_t* doc_ids;
    size_t count;
    size_t capacity;
} keystone_trigram_posting_list_t;

typedef struct keystone_trigram_doc {
    uint32_t id;
    char* name;
    char* content;
    size_t content_len;
    bool owns_content;
} keystone_trigram_doc_t;

struct keystone_trigram_index {
    /* Split hash table: bucket_keys holds 4-byte keys (or TRIGRAM_KEY_EMPTY),
     * probed cache-efficiently during lookups. bucket_lists holds the inline
     * posting list structs, only dereferenced on a key match. */
    uint32_t* bucket_keys;
    keystone_trigram_posting_list_t* bucket_lists;
    size_t num_buckets;
    size_t unique_trigrams;

    keystone_trigram_doc_t* docs;
    size_t doc_count;
    size_t doc_capacity;

    /* Per-document trigram dedup bitmap. Allocated once, reused across
     * documents. 2MB covers the full 24-bit trigram space (16M values).
     * doc_seen_touched tracks which bytes were set so clearing is O(unique)
     * instead of O(2MB). This skips redundant hash lookups for trigrams
     * that already appeared in the current document. */
    unsigned char* doc_seen;
    size_t* doc_seen_touched;
    size_t doc_seen_touched_count;
    size_t doc_seen_touched_cap;

    bool is_finalized;
    bool failed;
    int failure_code;
    keystone_trigram_stats_t stats;
};

static bool checked_mul_size(size_t a, size_t b, size_t* out) {
    if (!out) return false;
    if (a != 0u && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool checked_add_size(size_t a, size_t b, size_t* out) {
    if (!out || b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}

static size_t max_document_count(void) {
    size_t max_docs = (size_t)UINT32_MAX;
    if (sizeof(size_t) > sizeof(uint32_t)) max_docs += 1u;
    return max_docs;
}

static void secure_zero(void* ptr, size_t len) {
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    while (ptr && len-- > 0u) {
        *p++ = 0u;
    }
}

static int poison_index(keystone_trigram_index_t* idx, int code) {
    if (idx) {
        idx->failed = true;
        if (idx->failure_code == KEYSTONE_TRIGRAM_OK) {
            idx->failure_code = code;
        }
    }
    return code;
}

static char* duplicate_c_string(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    size_t alloc_len;
    if (!checked_add_size(len, 1u, &alloc_len)) return NULL;
    char* copy = (char*)malloc(alloc_len);
    if (!copy) return NULL;
    memcpy(copy, src, alloc_len);
    return copy;
}

static char* duplicate_content(const char* text, size_t text_len) {
    size_t alloc_len;
    if (!checked_add_size(text_len, 1u, &alloc_len)) return NULL;
    char* copy = (char*)malloc(alloc_len);
    if (!copy) return NULL;
    if (text_len > 0u) memcpy(copy, text, text_len);
    copy[text_len] = '\0';
    return copy;
}

/* Fibonacci hashing: 1 multiply + 1 shift. 0x9E3779B1 = 2^32/phi (golden
 * ratio), giving excellent distribution for power-of-two table sizes. */
static inline size_t hash_trigram_key(uint32_t key, size_t num_buckets) {
    unsigned shift = 32u - (unsigned)__builtin_ctzll((unsigned long long)num_buckets);
    return (size_t)((key * 0x9E3779B1u) >> shift);
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0u;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int resize_trigram_hash_table(keystone_trigram_index_t* idx) {
    if (!idx || !idx->bucket_keys || !idx->bucket_lists || idx->num_buckets == 0u) {
        return KEYSTONE_TRIGRAM_EINVAL;
    }
    if (idx->num_buckets > SIZE_MAX / 2u) {
        return KEYSTONE_TRIGRAM_EOVERFLOW;
    }

    size_t new_num_buckets = idx->num_buckets * 2u;
    size_t keys_bytes;
    if (!checked_mul_size(new_num_buckets, sizeof(uint32_t), &keys_bytes)) {
        return KEYSTONE_TRIGRAM_EOVERFLOW;
    }
    size_t lists_bytes;
    if (!checked_mul_size(new_num_buckets, sizeof(keystone_trigram_posting_list_t), &lists_bytes)) {
        return KEYSTONE_TRIGRAM_EOVERFLOW;
    }

    uint32_t* new_keys = (uint32_t*)malloc(keys_bytes);
    if (!new_keys) return KEYSTONE_TRIGRAM_ENOMEM;
    memset(new_keys, 0xFF, keys_bytes);

    keystone_trigram_posting_list_t* new_lists =
        (keystone_trigram_posting_list_t*)calloc(
            new_num_buckets, sizeof(keystone_trigram_posting_list_t));
    if (!new_lists) {
        free(new_keys);
        return KEYSTONE_TRIGRAM_ENOMEM;
    }

    for (size_t i = 0u; i < idx->num_buckets; i++) {
        if (idx->bucket_keys[i] == TRIGRAM_KEY_EMPTY) continue;

        uint32_t key = idx->bucket_keys[i];
        size_t bucket_idx = hash_trigram_key(key, new_num_buckets);
        while (new_keys[bucket_idx] != TRIGRAM_KEY_EMPTY) {
            bucket_idx = (bucket_idx + 1u) & (new_num_buckets - 1u);
        }
        new_keys[bucket_idx] = key;
        new_lists[bucket_idx] = idx->bucket_lists[i];
    }

    free(idx->bucket_keys);
    free(idx->bucket_lists);
    idx->bucket_keys = new_keys;
    idx->bucket_lists = new_lists;
    idx->num_buckets = new_num_buckets;
    return KEYSTONE_TRIGRAM_OK;
}

keystone_trigram_index_t* keystone_trigram_index_create(size_t initial_doc_capacity) {
    if (initial_doc_capacity == 0u) initial_doc_capacity = 64u;
    if (initial_doc_capacity > max_document_count()) return NULL;

    size_t ignored;
    if (!checked_mul_size(initial_doc_capacity, sizeof(keystone_trigram_doc_t), &ignored)) {
        return NULL;
    }

    keystone_trigram_index_t* idx =
        (keystone_trigram_index_t*)calloc(1u, sizeof(keystone_trigram_index_t));
    if (!idx) return NULL;

    idx->num_buckets = TRIGRAM_INITIAL_BUCKETS;
    size_t keys_bytes;
    if (!checked_mul_size(idx->num_buckets, sizeof(uint32_t), &keys_bytes)) {
        free(idx);
        return NULL;
    }
    idx->bucket_keys = (uint32_t*)malloc(keys_bytes);
    if (!idx->bucket_keys) {
        free(idx);
        return NULL;
    }
    memset(idx->bucket_keys, 0xFF, keys_bytes);

    size_t lists_bytes;
    if (!checked_mul_size(idx->num_buckets, sizeof(keystone_trigram_posting_list_t), &lists_bytes)) {
        free(idx->bucket_keys);
        free(idx);
        return NULL;
    }
    idx->bucket_lists = (keystone_trigram_posting_list_t*)calloc(
        idx->num_buckets, sizeof(keystone_trigram_posting_list_t));
    if (!idx->bucket_lists) {
        free(idx->bucket_keys);
        free(idx);
        return NULL;
    }

    idx->doc_seen = (unsigned char*)calloc(TRIGRAM_BITMAP_BYTES, 1);
    if (!idx->doc_seen) {
        free(idx->bucket_lists);
        free(idx->bucket_keys);
        free(idx);
        return NULL;
    }
    idx->doc_seen_touched_cap = 4096u;
    idx->doc_seen_touched =
        (size_t*)malloc(idx->doc_seen_touched_cap * sizeof(size_t));
    if (!idx->doc_seen_touched) {
        free(idx->doc_seen);
        free(idx->bucket_lists);
        free(idx->bucket_keys);
        free(idx);
        return NULL;
    }

    idx->doc_capacity = initial_doc_capacity;
    idx->docs = (keystone_trigram_doc_t*)calloc(
        idx->doc_capacity, sizeof(keystone_trigram_doc_t));
    if (!idx->docs) {
        free(idx->doc_seen_touched);
        free(idx->doc_seen);
        free(idx->bucket_lists);
        free(idx->bucket_keys);
        free(idx);
        return NULL;
    }

    idx->failure_code = KEYSTONE_TRIGRAM_OK;
    return idx;
}

void keystone_trigram_index_destroy(keystone_trigram_index_t* idx) {
    if (!idx) return;

    if (idx->bucket_lists) {
        for (size_t i = 0u; i < idx->num_buckets; i++) {
            if (idx->bucket_keys[i] != TRIGRAM_KEY_EMPTY) {
                free(idx->bucket_lists[i].doc_ids);
            }
        }
        free(idx->bucket_lists);
    }
    free(idx->bucket_keys);

    free(idx->doc_seen_touched);
    free(idx->doc_seen);

    if (idx->docs) {
        for (size_t i = 0u; i < idx->doc_count; i++) {
            if (idx->docs[i].name) {
                secure_zero(idx->docs[i].name, strlen(idx->docs[i].name));
                free(idx->docs[i].name);
            }
            if (idx->docs[i].owns_content && idx->docs[i].content) {
                secure_zero(idx->docs[i].content, idx->docs[i].content_len);
                free(idx->docs[i].content);
            }
        }
        free(idx->docs);
    }

    secure_zero(idx, sizeof(*idx));
    free(idx);
}

#ifdef __SSE4_2__
/* Each output dword holds a trigram key packed little-endian as
 * (p[i+2] | p[i+1]<<8 | p[i]<<16); 0x80 lanes zero the high byte so
 * pshufb produces a clean 24-bit key. Four shuffles over a single
 * 16-byte load yield the 14 trigrams whose three-byte windows fit
 * entirely within that load (positions 0..13). */
static const unsigned char KEYSHUF_M0[16] = {2,1,0,0x80, 3,2,1,0x80, 4,3,2,0x80, 5,4,3,0x80};
static const unsigned char KEYSHUF_M1[16] = {6,5,4,0x80, 7,6,5,0x80, 8,7,6,0x80, 9,8,7,0x80};
static const unsigned char KEYSHUF_M2[16] = {10,9,8,0x80, 11,10,9,0x80, 12,11,10,0x80, 13,12,11,0x80};
static const unsigned char KEYSHUF_M3[16] = {14,13,12,0x80, 15,14,13,0x80, 0x80,0x80,0x80,0x80, 0x80,0x80,0x80,0x80};

static inline void keystone_extract_trigrams16(const unsigned char* p, uint32_t* out) {
    __m128i v = _mm_loadu_si128((const __m128i*)p);
    _mm_storeu_si128((__m128i*)(out + 0),  _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)KEYSHUF_M0)));
    _mm_storeu_si128((__m128i*)(out + 4),  _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)KEYSHUF_M1)));
    _mm_storeu_si128((__m128i*)(out + 8),  _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)KEYSHUF_M2)));
    _mm_storeu_si128((__m128i*)(out + 12), _mm_shuffle_epi8(v, _mm_loadu_si128((const __m128i*)KEYSHUF_M3)));
}
#endif

size_t keystone_trigram_extract(
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_trigrams,
    size_t max_trigrams
) {
    if (!pattern || pattern_len < 3u || !out_trigrams || max_trigrams == 0u) return 0u;

    const unsigned char* p = (const unsigned char*)pattern;
    size_t extracted = 0u;
    size_t i = 0u;

#ifdef __SSE4_2__
    while (i + 16u <= pattern_len && extracted < max_trigrams) {
        uint32_t keys[16];
        keystone_extract_trigrams16(p + i, keys);
        for (int k = 0; k < 14 && extracted < max_trigrams; k++) {
            uint32_t key = keys[k];
            bool dup = false;
            for (size_t j = 0u; j < extracted; j++) {
                if (out_trigrams[j] == key) { dup = true; break; }
            }
            if (!dup) {
                out_trigrams[extracted++] = key;
            }
        }
        i += 14u;
    }
#endif
    for (; i <= pattern_len - 3u && extracted < max_trigrams; i++) {
        uint32_t key = ((uint32_t)p[i] << 16) |
                       ((uint32_t)p[i + 1u] << 8) |
                       (uint32_t)p[i + 2u];
        bool dup = false;
        for (size_t j = 0u; j < extracted; j++) {
            if (out_trigrams[j] == key) { dup = true; break; }
        }
        if (!dup) {
            out_trigrams[extracted++] = key;
        }
    }

    return extracted;
}

static keystone_trigram_posting_list_t* find_or_create_posting_list(
    keystone_trigram_index_t* idx,
    uint32_t key
) {
    if (!idx || idx->failed) return NULL;

    /* Resize at approximately 70% occupancy. Never continue after a failed
     * resize: silently degrading a candidate index can create false negatives. */
    size_t resize_threshold = (idx->num_buckets / 10u) * 7u;
    if (resize_threshold == 0u) resize_threshold = 1u;
    if (idx->unique_trigrams >= resize_threshold) {
        int rc = resize_trigram_hash_table(idx);
        if (rc != KEYSTONE_TRIGRAM_OK) {
            poison_index(idx, rc);
            return NULL;
        }
    }

    size_t bucket_idx = hash_trigram_key(key, idx->num_buckets);
    size_t mask = idx->num_buckets - 1u;
    size_t original = bucket_idx;

    while (idx->bucket_keys[bucket_idx] != TRIGRAM_KEY_EMPTY &&
           idx->bucket_keys[bucket_idx] != key) {
        bucket_idx = (bucket_idx + 1u) & mask;
        if (bucket_idx == original) {
            poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
            return NULL;
        }
    }

    keystone_trigram_posting_list_t* plist = &idx->bucket_lists[bucket_idx];
    if (idx->bucket_keys[bucket_idx] == TRIGRAM_KEY_EMPTY) {
        size_t bytes;
        if (!checked_mul_size(TRIGRAM_INITIAL_POSTING_CAPACITY, sizeof(uint32_t), &bytes)) {
            poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
            return NULL;
        }
        uint32_t* ids = (uint32_t*)malloc(bytes);
        if (!ids) {
            poison_index(idx, KEYSTONE_TRIGRAM_ENOMEM);
            return NULL;
        }

        plist->doc_ids = ids;
        plist->count = 0u;
        plist->capacity = TRIGRAM_INITIAL_POSTING_CAPACITY;
        idx->bucket_keys[bucket_idx] = key;
        idx->unique_trigrams++;
    }

    return plist;
}

static int add_doc_to_posting_list(
    keystone_trigram_index_t* idx,
    keystone_trigram_posting_list_t* plist,
    uint32_t doc_id
) {
    if (!idx || !plist || !plist->doc_ids || plist->capacity == 0u) {
        return poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
    }

    if (plist->count > 0u && plist->doc_ids[plist->count - 1u] == doc_id) {
        return KEYSTONE_TRIGRAM_OK;
    }

    if (plist->count >= plist->capacity) {
        if (plist->capacity > SIZE_MAX / 2u) {
            return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
        }
        size_t new_cap = plist->capacity * 2u;
        size_t bytes;
        if (!checked_mul_size(new_cap, sizeof(uint32_t), &bytes)) {
            return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
        }
        uint32_t* new_ids = (uint32_t*)realloc(plist->doc_ids, bytes);
        if (!new_ids) return poison_index(idx, KEYSTONE_TRIGRAM_ENOMEM);
        plist->doc_ids = new_ids;
        plist->capacity = new_cap;
    }

    plist->doc_ids[plist->count++] = doc_id;
    return KEYSTONE_TRIGRAM_OK;
}

static int ensure_doc_capacity(keystone_trigram_index_t* idx) {
    if (!idx) return KEYSTONE_TRIGRAM_EINVAL;
    if (idx->doc_count < idx->doc_capacity) return KEYSTONE_TRIGRAM_OK;

    if (idx->doc_capacity > SIZE_MAX / 2u) {
        return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
    }
    size_t new_cap = idx->doc_capacity * 2u;
    size_t max_docs = max_document_count();
    if (new_cap > max_docs) new_cap = max_docs;
    if (new_cap <= idx->doc_capacity) {
        return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
    }

    size_t bytes;
    if (!checked_mul_size(new_cap, sizeof(keystone_trigram_doc_t), &bytes)) {
        return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
    }
    keystone_trigram_doc_t* new_docs =
        (keystone_trigram_doc_t*)realloc(idx->docs, bytes);
    if (!new_docs) return poison_index(idx, KEYSTONE_TRIGRAM_ENOMEM);

    memset(new_docs + idx->doc_capacity, 0,
           (new_cap - idx->doc_capacity) * sizeof(keystone_trigram_doc_t));
    idx->docs = new_docs;
    idx->doc_capacity = new_cap;
    return KEYSTONE_TRIGRAM_OK;
}

static void clear_doc_seen(keystone_trigram_index_t* idx) {
    for (size_t t = 0u; t < idx->doc_seen_touched_count; t++) {
        idx->doc_seen[idx->doc_seen_touched[t]] = 0u;
    }
    idx->doc_seen_touched_count = 0u;
}

/* Per-trigram dedup + posting-list append. Factored so both the SIMD
 * batch path and the scalar tail share one copy of the bitmap logic.
 * `continue` inside the do/while exits the block and advances the
 * enclosing for-loop to the next trigram. */
#define KEYSTONE_EMIT_TRIGRAM(KEY) do { \
    size_t byte_idx = (KEY) >> 3; \
    unsigned char bit_mask = (unsigned char)(1u << ((KEY) & 7u)); \
    unsigned char old = idx->doc_seen[byte_idx]; \
    if (old & bit_mask) continue; \
    if (old == 0u) { \
        if (idx->doc_seen_touched_count >= idx->doc_seen_touched_cap) { \
            size_t new_cap = idx->doc_seen_touched_cap * 2u; \
            size_t new_bytes; \
            if (!checked_mul_size(new_cap, sizeof(size_t), &new_bytes)) { \
                rc = KEYSTONE_TRIGRAM_EOVERFLOW; \
                goto fail_prepared_doc; \
            } \
            size_t* new_touched = \
                (size_t*)realloc(idx->doc_seen_touched, new_bytes); \
            if (!new_touched) { \
                rc = KEYSTONE_TRIGRAM_ENOMEM; \
                goto fail_prepared_doc; \
            } \
            idx->doc_seen_touched = new_touched; \
            idx->doc_seen_touched_cap = new_cap; \
        } \
        idx->doc_seen_touched[idx->doc_seen_touched_count++] = byte_idx; \
    } \
    idx->doc_seen[byte_idx] = old | bit_mask; \
    keystone_trigram_posting_list_t* plist = find_or_create_posting_list(idx, (KEY)); \
    if (!plist) { \
        rc = idx->failure_code ? idx->failure_code : KEYSTONE_TRIGRAM_ESTATE; \
        goto fail_prepared_doc; \
    } \
    rc = add_doc_to_posting_list(idx, plist, doc_id); \
    if (rc != KEYSTONE_TRIGRAM_OK) goto fail_prepared_doc; \
} while (0)

static int add_document_internal(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    bool retain_content,
    uint32_t* out_doc_id
) {
    if (!idx || !text) return KEYSTONE_TRIGRAM_EINVAL;
    if (idx->failed) return idx->failure_code ? idx->failure_code : KEYSTONE_TRIGRAM_ESTATE;
    if (idx->is_finalized) return KEYSTONE_TRIGRAM_ESTATE;
    if (idx->doc_count > (size_t)UINT32_MAX) {
        return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
    }
    if (text_len > SIZE_MAX - idx->stats.bytes_indexed) {
        return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
    }

    int rc = ensure_doc_capacity(idx);
    if (rc != KEYSTONE_TRIGRAM_OK) return rc;

    char* name_copy = NULL;
    char* content_copy = NULL;
    if (name) {
        name_copy = duplicate_c_string(name);
        if (!name_copy) return poison_index(idx, KEYSTONE_TRIGRAM_ENOMEM);
    }
    if (retain_content) {
        content_copy = duplicate_content(text, text_len);
        if (!content_copy) {
            if (name_copy) {
                secure_zero(name_copy, strlen(name_copy));
                free(name_copy);
            }
            return poison_index(idx, KEYSTONE_TRIGRAM_ENOMEM);
        }
    }

    uint64_t start_time = get_time_ns();
    uint32_t doc_id = (uint32_t)idx->doc_count;

    if (text_len >= 3u) {
        const unsigned char* p = (const unsigned char*)text;
        size_t i = 0u;
#ifdef __SSE4_2__
        while (i + 16u <= text_len) {
            uint32_t keys[16];
            keystone_extract_trigrams16(p + i, keys);
            for (int k = 0; k < 14; k++) {
                KEYSTONE_EMIT_TRIGRAM(keys[k]);
            }
            i += 14u;
        }
#endif
        for (; i <= text_len - 3u; i++) {
            uint32_t key = ((uint32_t)p[i] << 16) |
                           ((uint32_t)p[i + 1u] << 8) |
                           (uint32_t)p[i + 2u];
            KEYSTONE_EMIT_TRIGRAM(key);
        }
    }

    clear_doc_seen(idx);

    keystone_trigram_doc_t* doc = &idx->docs[idx->doc_count];
    doc->id = doc_id;
    doc->name = name_copy;
    doc->content = content_copy;
    doc->content_len = text_len;
    doc->owns_content = retain_content;
    idx->doc_count++;

    idx->stats.total_documents = idx->doc_count;
    idx->stats.bytes_indexed += text_len;
    uint64_t end_time = get_time_ns();
    if (end_time >= start_time && UINT64_MAX - idx->stats.build_time_ns >= end_time - start_time) {
        idx->stats.build_time_ns += end_time - start_time;
    } else {
        idx->stats.build_time_ns = UINT64_MAX;
    }

    if (out_doc_id) *out_doc_id = doc_id;
    return KEYSTONE_TRIGRAM_OK;

fail_prepared_doc:
    clear_doc_seen(idx);
    if (content_copy) {
        secure_zero(content_copy, text_len);
        free(content_copy);
    }
    if (name_copy) {
        secure_zero(name_copy, strlen(name_copy));
        free(name_copy);
    }
    return poison_index(idx, rc == KEYSTONE_TRIGRAM_OK ? KEYSTONE_TRIGRAM_ESTATE : rc);
}

int keystone_trigram_index_add_document(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
) {
    return add_document_internal(idx, name, text, text_len, true, out_doc_id);
}

int keystone_trigram_index_add_document_external(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
) {
    return add_document_internal(idx, name, text, text_len, false, out_doc_id);
}

static int compare_uint32(const void* a, const void* b) {
    uint32_t u1 = *(const uint32_t*)a;
    uint32_t u2 = *(const uint32_t*)b;
    return (u1 > u2) - (u1 < u2);
}

/* Counting sort using a caller-provided shared counts array (zeroed and
 * reused across posting lists to eliminate per-list allocation churn). */
static void counting_sort_doc_ids(
    uint32_t* arr, size_t count, size_t max_val, size_t* counts
) {
    if (count <= 1u) return;
    memset(counts, 0, (max_val + 1u) * sizeof(size_t));
    for (size_t i = 0u; i < count; i++) counts[arr[i]]++;
    size_t pos = 0u;
    for (size_t v = 0u; v <= max_val; v++) {
        for (size_t c = 0u; c < counts[v]; c++) arr[pos++] = (uint32_t)v;
    }
}

int keystone_trigram_index_finalize(keystone_trigram_index_t* idx) {
    if (!idx) return KEYSTONE_TRIGRAM_EINVAL;
    if (idx->failed) return idx->failure_code ? idx->failure_code : KEYSTONE_TRIGRAM_ESTATE;
    if (idx->is_finalized) return KEYSTONE_TRIGRAM_OK;

    size_t dc = idx->doc_count;
    size_t* counts = NULL;
    if (dc > 0u && dc < (1u << 20)) {
        size_t counts_bytes;
        if (checked_mul_size(dc + 1u, sizeof(size_t), &counts_bytes)) {
            counts = (size_t*)malloc(counts_bytes);
        }
    }

    size_t total_postings = 0u;
    for (size_t i = 0u; i < idx->num_buckets; i++) {
        if (idx->bucket_keys[i] == TRIGRAM_KEY_EMPTY) continue;
        keystone_trigram_posting_list_t* plist = &idx->bucket_lists[i];
        if (plist->count == 0u) continue;
        if (!plist->doc_ids || plist->count > plist->capacity) {
            free(counts);
            return poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
        }
        if (counts) {
            counting_sort_doc_ids(plist->doc_ids, plist->count, dc, counts);
        } else {
            qsort(plist->doc_ids, plist->count, sizeof(uint32_t), compare_uint32);
        }
        if (plist->count > SIZE_MAX - total_postings) {
            free(counts);
            return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
        }
        total_postings += plist->count;
    }

    free(counts);

    idx->stats.unique_trigrams = idx->unique_trigrams;
    idx->stats.total_postings = total_postings;
    idx->is_finalized = true;
    return KEYSTONE_TRIGRAM_OK;
}

static const keystone_trigram_posting_list_t* get_posting_list(
    const keystone_trigram_index_t* idx,
    uint32_t key
) {
    if (!idx || !idx->bucket_keys || idx->num_buckets == 0u) return NULL;

    size_t mask = idx->num_buckets - 1u;
    size_t bucket_idx = hash_trigram_key(key, idx->num_buckets);
    size_t original = bucket_idx;

    while (idx->bucket_keys[bucket_idx] != TRIGRAM_KEY_EMPTY) {
        if (idx->bucket_keys[bucket_idx] == key) {
            return &idx->bucket_lists[bucket_idx];
        }
        bucket_idx = (bucket_idx + 1u) & mask;
        if (bucket_idx == original) break;
    }

    return NULL;
}

size_t keystone_trigram_index_document_count(const keystone_trigram_index_t* idx) {
    return idx ? idx->doc_count : 0u;
}

#ifdef __SSE4_2__
/* Unsigned lower_bound over arr[pos..cnt) for target, four uint32s at a
 * time. Sign bits are flipped so the signed _mm_cmpgt_epi32 implements
 * an unsigned comparison. Returns the first index >= target and sets
 * *found if target is present at that index. */
static inline size_t simd_lower_bound_u32(
    const uint32_t* arr, size_t pos, size_t cnt, uint32_t target, int* found
) {
    const __m128i sign = _mm_set1_epi32((int)0x80000000u);
    __m128i tgt = _mm_xor_si128(_mm_set1_epi32((int)target), sign);
    size_t i = pos;
    for (; i + 4u <= cnt; i += 4u) {
        __m128i v = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(arr + i)), sign);
        __m128i eq = _mm_cmpeq_epi32(v, tgt);
        __m128i ge = _mm_or_si128(eq, _mm_cmpgt_epi32(v, tgt));
        unsigned int gm = (unsigned int)_mm_movemask_ps(_mm_castsi128_ps(ge));
        if (gm) {
            int bit = __builtin_ctz(gm);
            unsigned int em = (unsigned int)_mm_movemask_ps(_mm_castsi128_ps(eq));
            *found = (int)((em >> bit) & 1u);
            return i + (size_t)bit;
        }
    }
    for (; i < cnt; i++) {
        if (arr[i] == target) { *found = 1; return i; }
        if (arr[i] > target) { *found = 0; return i; }
    }
    *found = 0;
    return cnt;
}
#endif

size_t keystone_trigram_index_get_candidates(
    const keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_candidates,
    size_t max_candidates
) {
    if (!idx || idx->failed || !idx->is_finalized || !pattern ||
        !out_candidates || max_candidates == 0u) {
        return 0u;
    }

    if (pattern_len < 3u) {
        size_t count = idx->doc_count < max_candidates ? idx->doc_count : max_candidates;
        for (size_t i = 0u; i < count; i++) out_candidates[i] = (uint32_t)i;
        return count;
    }

    uint32_t query_trigrams[TRIGRAM_QUERY_MAX_UNIQUE];
    size_t num_trigrams = keystone_trigram_extract(
        pattern, pattern_len, query_trigrams, TRIGRAM_QUERY_MAX_UNIQUE);
    if (num_trigrams == 0u) return 0u;

    const keystone_trigram_posting_list_t* lists[TRIGRAM_QUERY_MAX_UNIQUE];
    size_t num_lists = 0u;
    for (size_t i = 0u; i < num_trigrams; i++) {
        const keystone_trigram_posting_list_t* plist =
            get_posting_list(idx, query_trigrams[i]);
        if (!plist || plist->count == 0u) return 0u;
        lists[num_lists++] = plist;
    }

    for (size_t i = 0u; i < num_lists; i++) {
        for (size_t j = i + 1u; j < num_lists; j++) {
            if (lists[j]->count < lists[i]->count) {
                const keystone_trigram_posting_list_t* tmp = lists[i];
                lists[i] = lists[j];
                lists[j] = tmp;
            }
        }
    }

    /* Galloping intersection: for each doc_id in the smallest list,
     * use exponential+binary search (galloping) to probe larger lists.
     * This is O(n * log(m/n)) instead of O(n * log(m)) for plain bsearch,
     * with a significant win when n << m (common for trigram queries). */
    const keystone_trigram_posting_list_t* base = lists[0];
    size_t candidate_count = 0u;

    /* Track search positions in each list for galloping advancement */
    size_t cursor[TRIGRAM_QUERY_MAX_UNIQUE];
    for (size_t l = 0u; l < num_lists; l++) cursor[l] = 0u;

    for (size_t i = 0u; i < base->count; i++) {
        uint32_t doc_id = base->doc_ids[i];
        bool in_all = true;

        for (size_t l = 1u; l < num_lists; l++) {
            const keystone_trigram_posting_list_t* plist = lists[l];
            const uint32_t* arr = plist->doc_ids;
            size_t cnt = plist->count;
            size_t pos = cursor[l];

#ifdef __SSE4_2__
            /* Small lists: a 4-way SIMD linear scan from the cursor beats
             * the branchy gallop+binary search when the list is short. */
            if (cnt < 64u) {
                int found = 0;
                size_t lo = simd_lower_bound_u32(arr, pos, cnt, doc_id, &found);
                if (found) {
                    cursor[l] = lo + 1u;
                    continue;
                }
                cursor[l] = lo;
                in_all = false;
                break;
            }
#endif

            /* Skip forward: if we've already passed this position, gallop */
            if (pos < cnt && arr[pos] == doc_id) {
                cursor[l] = pos + 1u;
                continue;
            }

            /* Galloping search: exponential probe from cursor, then binary */
            if (pos >= cnt || arr[pos] > doc_id) {
                in_all = false;
                break;
            }

            /* Exponential jump */
            size_t jump = 1u;
            size_t gallop_pos = pos;
            while (gallop_pos + jump < cnt && arr[gallop_pos + jump] <= doc_id) {
                gallop_pos += jump;
                jump <<= 1;
            }

            /* Binary search in [gallop_pos, min(gallop_pos+jump, cnt)) */
            size_t lo = gallop_pos;
            size_t hi = (gallop_pos + jump < cnt) ? gallop_pos + jump : cnt;
            while (lo < hi) {
                size_t mid = lo + ((hi - lo) >> 1);
                if (arr[mid] < doc_id) lo = mid + 1u;
                else hi = mid;
            }

            if (lo < cnt && arr[lo] == doc_id) {
                cursor[l] = lo + 1u;
            } else {
                cursor[l] = lo; /* save position for next gallop */
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

/* SIMD-accelerated substring search.
 * For needles >= 16 bytes the first 16 bytes are compared directly with
 * a single XMM compare, an extremely selective filter. For shorter
 * needles the first and last byte are scanned simultaneously: a
 * candidate position must match both the first byte (at offset 0) and
 * the last byte (at offset needle_len-1), eliminating most false
 * positives before memcmp touches the middle. Falls back to a
 * byte-at-a-time scan when SSE4.2 is unavailable. */
static const void* bounded_memmem(
    const void* haystack,
    size_t haystack_len,
    const void* needle,
    size_t needle_len
) {
    if (!haystack || !needle || needle_len == 0u || haystack_len < needle_len) return NULL;

    const unsigned char* h = (const unsigned char*)haystack;
    const unsigned char* n = (const unsigned char*)needle;
    size_t limit = haystack_len - needle_len;

    if (needle_len == 1u) {
        return memchr(haystack, n[0], haystack_len);
    }

#if HAVE_SSE42
    __m128i first_byte = _mm_set1_epi8((char)n[0]);
    __m128i last_byte = _mm_set1_epi8((char)n[needle_len - 1u]);
    __m128i n16 = (needle_len >= 16u)
        ? _mm_loadu_si128((const __m128i*)n)
        : _mm_setzero_si128();
    size_t i = 0u;
    while (i + 16u <= limit) {
        __m128i chunk_f = _mm_loadu_si128((const __m128i*)(h + i));
        __m128i chunk_l = _mm_loadu_si128((const __m128i*)(h + i + needle_len - 1u));
        unsigned int mask_f = (unsigned int)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk_f, first_byte));
        unsigned int mask_l = (unsigned int)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk_l, last_byte));
        unsigned int mask = mask_f & mask_l;
        while (mask) {
            int bit = __builtin_ctz(mask);
            mask &= mask - 1;
            size_t pos = i + (size_t)bit;
            if (pos > limit) continue;
#ifdef __SSE4_2__
            if (needle_len >= 16u) {
                __m128i h16 = _mm_loadu_si128((const __m128i*)(h + pos));
                if ((unsigned int)_mm_movemask_epi8(_mm_cmpeq_epi8(h16, n16)) != 0xFFFFu)
                    continue;
                if (needle_len > 16u &&
                    memcmp(h + pos + 16u, n + 16u, needle_len - 16u) != 0)
                    continue;
                return h + pos;
            }
#endif
            if (memcmp(h + pos, n, needle_len) == 0)
                return h + pos;
        }
        i += 16u;
    }
    for (; i <= limit; i++) {
#else
    for (size_t i = 0u; i <= limit; i++) {
#endif
        if (h[i] == n[0] && memcmp(h + i, n, needle_len) == 0)
            return h + i;
    }
    return NULL;
}

size_t keystone_trigram_index_search(
    keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_matches,
    size_t max_matches
) {
    if (!idx || idx->failed || !idx->is_finalized || !pattern || pattern_len == 0u ||
        !out_matches || max_matches == 0u) {
        return 0u;
    }

    idx->stats.total_searches++;
    if (idx->doc_count == 0u) return 0u;

    size_t candidate_bytes;
    if (!checked_mul_size(idx->doc_count, sizeof(uint32_t), &candidate_bytes)) return 0u;
    uint32_t* candidates = (uint32_t*)malloc(candidate_bytes);
    if (!candidates) return 0u;

    size_t num_candidates = keystone_trigram_index_get_candidates(
        idx, pattern, pattern_len, candidates, idx->doc_count);

    idx->stats.candidate_docs_evaluated += num_candidates;
    idx->stats.candidate_docs_rejected += idx->doc_count - num_candidates;

    size_t matches = 0u;
    for (size_t i = 0u; i < num_candidates; i++) {
        uint32_t doc_id = candidates[i];
        if ((size_t)doc_id >= idx->doc_count) {
            poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
            break;
        }

        keystone_trigram_doc_t* doc = &idx->docs[doc_id];
        if (!doc->owns_content || !doc->content || doc->content_len < pattern_len) continue;

        if (bounded_memmem(doc->content, doc->content_len, pattern, pattern_len)) {
            out_matches[matches++] = doc_id;
            if (matches >= max_matches) break;
        }
    }

    secure_zero(candidates, candidate_bytes);
    free(candidates);
    return matches;
}

void keystone_trigram_index_get_stats(
    const keystone_trigram_index_t* idx,
    keystone_trigram_stats_t* stats
) {
    if (idx && stats) *stats = idx->stats;
}
