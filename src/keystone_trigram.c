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

typedef struct keystone_trigram_posting_list {
    uint32_t trigram_key;
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
    keystone_trigram_posting_list_t* buckets;
    size_t num_buckets;
    size_t unique_trigrams;

    keystone_trigram_doc_t* docs;
    size_t doc_count;
    size_t doc_capacity;

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

/* FNV-1a hash for 24-bit trigrams into a power-of-two bucket table. */
static inline size_t hash_trigram_key(uint32_t key, size_t num_buckets) {
    uint32_t h = 2166136261u;
    h ^= (key & 0xFFu);
    h *= 16777619u;
    h ^= ((key >> 8) & 0xFFu);
    h *= 16777619u;
    h ^= ((key >> 16) & 0xFFu);
    h *= 16777619u;
    return (size_t)h & (num_buckets - 1u);
}

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0u;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int resize_trigram_hash_table(keystone_trigram_index_t* idx) {
    if (!idx || !idx->buckets || idx->num_buckets == 0u) {
        return KEYSTONE_TRIGRAM_EINVAL;
    }
    if (idx->num_buckets > SIZE_MAX / 2u) {
        return KEYSTONE_TRIGRAM_EOVERFLOW;
    }

    size_t new_num_buckets = idx->num_buckets * 2u;
    size_t bytes;
    if (!checked_mul_size(new_num_buckets, sizeof(keystone_trigram_posting_list_t), &bytes)) {
        return KEYSTONE_TRIGRAM_EOVERFLOW;
    }
    (void)bytes;

    keystone_trigram_posting_list_t* new_buckets =
        (keystone_trigram_posting_list_t*)calloc(
            new_num_buckets, sizeof(keystone_trigram_posting_list_t));
    if (!new_buckets) return KEYSTONE_TRIGRAM_ENOMEM;

    for (size_t i = 0u; i < idx->num_buckets; i++) {
        if (idx->buckets[i].capacity == 0u) continue;

        uint32_t key = idx->buckets[i].trigram_key;
        size_t bucket_idx = hash_trigram_key(key, new_num_buckets);
        while (new_buckets[bucket_idx].capacity > 0u) {
            bucket_idx = (bucket_idx + 1u) & (new_num_buckets - 1u);
        }
        new_buckets[bucket_idx] = idx->buckets[i];
    }

    free(idx->buckets);
    idx->buckets = new_buckets;
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
    idx->buckets = (keystone_trigram_posting_list_t*)calloc(
        idx->num_buckets, sizeof(keystone_trigram_posting_list_t));
    if (!idx->buckets) {
        free(idx);
        return NULL;
    }

    idx->doc_capacity = initial_doc_capacity;
    idx->docs = (keystone_trigram_doc_t*)calloc(
        idx->doc_capacity, sizeof(keystone_trigram_doc_t));
    if (!idx->docs) {
        free(idx->buckets);
        free(idx);
        return NULL;
    }

    idx->failure_code = KEYSTONE_TRIGRAM_OK;
    return idx;
}

void keystone_trigram_index_destroy(keystone_trigram_index_t* idx) {
    if (!idx) return;

    if (idx->buckets) {
        for (size_t i = 0u; i < idx->num_buckets; i++) {
            free(idx->buckets[i].doc_ids);
        }
        free(idx->buckets);
    }

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

size_t keystone_trigram_extract(
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_trigrams,
    size_t max_trigrams
) {
    if (!pattern || pattern_len < 3u || !out_trigrams || max_trigrams == 0u) return 0u;

    const unsigned char* p = (const unsigned char*)pattern;
    size_t extracted = 0u;

    /* Heap-allocated bitmap dedup: O(n) single pass instead of O(n²).
     * 2MB covers all 16M possible 24-bit trigrams. */
    unsigned char* bitmap = (unsigned char*)calloc(TRIGRAM_BITMAP_BYTES, 1);
    if (!bitmap) {
        /* Fallback: original O(n²) scan if allocation fails */
        for (size_t i = 0u; i <= pattern_len - 3u; i++) {
            uint32_t key = ((uint32_t)p[i] << 16) |
                           ((uint32_t)p[i + 1u] << 8) |
                           (uint32_t)p[i + 2u];
            bool dup = false;
            for (size_t j = 0u; j < extracted; j++) {
                if (out_trigrams[j] == key) { dup = true; break; }
            }
            if (!dup) {
                out_trigrams[extracted++] = key;
                if (extracted >= max_trigrams) break;
            }
        }
        return extracted;
    }

    for (size_t i = 0u; i <= pattern_len - 3u; i++) {
        uint32_t key = ((uint32_t)p[i] << 16) |
                       ((uint32_t)p[i + 1u] << 8) |
                       (uint32_t)p[i + 2u];

        size_t byte_idx = key >> 3;
        unsigned char bit_mask = (unsigned char)(1u << (key & 7u));

        if (!(bitmap[byte_idx] & bit_mask)) {
            bitmap[byte_idx] |= bit_mask;
            out_trigrams[extracted++] = key;
            if (extracted >= max_trigrams) break;
        }
    }

    free(bitmap);
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

    while (idx->buckets[bucket_idx].capacity > 0u &&
           idx->buckets[bucket_idx].trigram_key != key) {
        bucket_idx = (bucket_idx + 1u) & mask;
        if (bucket_idx == original) {
            poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
            return NULL;
        }
    }

    keystone_trigram_posting_list_t* plist = &idx->buckets[bucket_idx];
    if (plist->capacity == 0u) {
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

        plist->trigram_key = key;
        plist->doc_ids = ids;
        plist->count = 0u;
        plist->capacity = TRIGRAM_INITIAL_POSTING_CAPACITY;
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
        for (size_t i = 0u; i <= text_len - 3u; i++) {
            uint32_t key = ((uint32_t)p[i] << 16) |
                           ((uint32_t)p[i + 1u] << 8) |
                           (uint32_t)p[i + 2u];
            keystone_trigram_posting_list_t* plist = find_or_create_posting_list(idx, key);
            if (!plist) {
                rc = idx->failure_code ? idx->failure_code : KEYSTONE_TRIGRAM_ESTATE;
                goto fail_prepared_doc;
            }
            rc = add_doc_to_posting_list(idx, plist, doc_id);
            if (rc != KEYSTONE_TRIGRAM_OK) goto fail_prepared_doc;
        }
    }

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

/* Counting sort for uint32 posting lists. O(n + k) where k = doc_count.
 * Faster than qsort's O(n log n) when doc_count is bounded and known. */
static void counting_sort_doc_ids(uint32_t* arr, size_t count, size_t max_val) {
    if (count <= 1u) return;
    if (max_val == 0u) max_val = count;

    /* Use counting sort for small ranges, qsort for large */
    if (max_val <= (1u << 20)) {
        /* Counting sort */
        size_t* counts = (size_t*)calloc(max_val + 1u, sizeof(size_t));
        if (!counts) {
            /* Fallback to qsort if allocation fails */
            qsort(arr, count, sizeof(uint32_t), compare_uint32);
            return;
        }
        for (size_t i = 0u; i < count; i++) counts[arr[i]]++;
        size_t pos = 0u;
        for (size_t v = 0u; v <= max_val; v++) {
            for (size_t c = 0u; c < counts[v]; c++) arr[pos++] = (uint32_t)v;
        }
        free(counts);
    } else {
        qsort(arr, count, sizeof(uint32_t), compare_uint32);
    }
}

int keystone_trigram_index_finalize(keystone_trigram_index_t* idx) {
    if (!idx) return KEYSTONE_TRIGRAM_EINVAL;
    if (idx->failed) return idx->failure_code ? idx->failure_code : KEYSTONE_TRIGRAM_ESTATE;
    if (idx->is_finalized) return KEYSTONE_TRIGRAM_OK;

    size_t total_postings = 0u;
    for (size_t i = 0u; i < idx->num_buckets; i++) {
        keystone_trigram_posting_list_t* plist = &idx->buckets[i];
        if (plist->count == 0u) continue;
        if (!plist->doc_ids || plist->count > plist->capacity) {
            return poison_index(idx, KEYSTONE_TRIGRAM_ESTATE);
        }
        counting_sort_doc_ids(plist->doc_ids, plist->count, idx->doc_count);
        if (plist->count > SIZE_MAX - total_postings) {
            return poison_index(idx, KEYSTONE_TRIGRAM_EOVERFLOW);
        }
        total_postings += plist->count;
    }

    idx->stats.unique_trigrams = idx->unique_trigrams;
    idx->stats.total_postings = total_postings;
    idx->is_finalized = true;
    return KEYSTONE_TRIGRAM_OK;
}

static const keystone_trigram_posting_list_t* get_posting_list(
    const keystone_trigram_index_t* idx,
    uint32_t key
) {
    if (!idx || !idx->buckets || idx->num_buckets == 0u) return NULL;

    size_t mask = idx->num_buckets - 1u;
    size_t bucket_idx = hash_trigram_key(key, idx->num_buckets);
    size_t original = bucket_idx;

    while (idx->buckets[bucket_idx].capacity > 0u) {
        if (idx->buckets[bucket_idx].trigram_key == key) {
            return &idx->buckets[bucket_idx];
        }
        bucket_idx = (bucket_idx + 1u) & mask;
        if (bucket_idx == original) break;
    }

    return NULL;
}

size_t keystone_trigram_index_document_count(const keystone_trigram_index_t* idx) {
    return idx ? idx->doc_count : 0u;
}

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
 * Uses SSE4.2 PCMPESTRI to find first-byte candidates 16 at a time,
 * then memcmp to confirm full match. Falls back to byte-at-a-time
 * scan when SSE4.2 is unavailable. */
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
        /* Single-byte: use memchr which glibc already SIMD-optimizes */
        return memchr(haystack, n[0], haystack_len);
    }

#if HAVE_SSE42
    /* Load first byte into all 16 lanes of an XMM register */
    __m128i first_byte = _mm_set1_epi8((char)n[0]);
    size_t i = 0u;
    /* Process 16 bytes at a time with SSE4.2 PCMPESTRI */
    while (i + 16u <= limit) {
        __m128i chunk = _mm_loadu_si128((const __m128i*)(h + i));
        __m128i eq = _mm_cmpeq_epi8(chunk, first_byte);
        unsigned int mask = (unsigned int)_mm_movemask_epi8(eq);

        while (mask) {
            int bit = __builtin_ctz(mask);
            mask &= mask - 1; /* clear lowest set bit */
            size_t pos = i + (size_t)bit;
            if (pos <= limit && memcmp(h + pos, n, needle_len) == 0)
                return h + pos;
        }
        i += 16u;
    }
    /* Scalar tail */
    for (; i <= limit; i++) {
#else
    /* No SSE4.2: full scalar scan */
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
