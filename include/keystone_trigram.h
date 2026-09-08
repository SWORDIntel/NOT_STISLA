/**
 * KEYSTONE Trigram Content Index (trcg) Engine
 *
 * Inspired by Microsoft tgrep (trigram-indexed search).
 * Indexes 3-byte character sequences (trigrams) from text documents or log chunks
 * into inverted posting lists. Provides sub-linear candidate file rejection by
 * intersecting posting lists before performing full pattern/string verification.
 */

#ifndef KEYSTONE_TRIGRAM_H
#define KEYSTONE_TRIGRAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Single posting list entry structure for a 24-bit trigram.
 */
typedef struct keystone_trigram_posting_list {
    uint32_t trigram_key;    /* 24-bit packed trigram: (b0 << 16) | (b1 << 8) | b2 */
    uint32_t* doc_ids;       /* Sorted array of document IDs containing this trigram */
    size_t count;            /* Number of document IDs */
    size_t capacity;         /* Allocated capacity */
} keystone_trigram_posting_list_t;

/**
 * Statistics for the trigram index.
 */
typedef struct keystone_trigram_stats {
    size_t total_documents;
    size_t unique_trigrams;
    size_t total_postings;
    size_t bytes_indexed;
    uint64_t build_time_ns;
    uint64_t total_searches;
    uint64_t candidate_docs_evaluated;
    uint64_t candidate_docs_rejected;
} keystone_trigram_stats_t;

/**
 * Document record stored in the index for verification.
 */
typedef struct keystone_trigram_doc {
    uint32_t id;
    char* name;             /* Optional document/file identifier */
    const char* content;    /* Reference to text content (or NULL if external) */
    size_t content_len;
} keystone_trigram_doc_t;

/**
 * Trigram Content Index opaque handle.
 */
typedef struct keystone_trigram_index {
    keystone_trigram_posting_list_t* buckets; /* Hash table of posting lists */
    size_t num_buckets;
    size_t unique_trigrams;

    keystone_trigram_doc_t* docs;
    size_t doc_count;
    size_t doc_capacity;

    bool is_finalized;
    keystone_trigram_stats_t stats;
} keystone_trigram_index_t;

/**
 * @brief Create a new trigram content index.
 * @param initial_doc_capacity Expected number of documents.
 * @return Pointer to new index, or NULL on memory failure.
 */
keystone_trigram_index_t* keystone_trigram_index_create(size_t initial_doc_capacity);

/**
 * @brief Destroy a trigram content index and free all allocated memory.
 */
void keystone_trigram_index_destroy(keystone_trigram_index_t* idx);

/**
 * @brief Add a text document to the trigram index.
 * @param idx Trigram index.
 * @param name Document name or path (optional, can be NULL).
 * @param text Document content buffer.
 * @param text_len Length of text content.
 * @param out_doc_id Pointer to receive assigned document ID (optional, can be NULL).
 * @return 0 on success, negative error code on failure.
 */
int keystone_trigram_index_add_document(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
);

/**
 * @brief Finalize the index (sorts posting lists and prepares fast lookup tables).
 * Must be called after adding documents and before running queries.
 */
int keystone_trigram_index_finalize(keystone_trigram_index_t* idx);

/**
 * @brief Extract trigram keys from a text pattern.
 * @param pattern Text string or literal pattern.
 * @param pattern_len Length of pattern.
 * @param out_trigrams Output array for 24-bit trigram keys.
 * @param max_trigrams Allocated size of out_trigrams array.
 * @return Number of unique trigrams extracted.
 */
size_t keystone_trigram_extract(
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_trigrams,
    size_t max_trigrams
);

/**
 * @brief Intersect posting lists to return candidate document IDs.
 * @param idx Finalized trigram index.
 * @param pattern Query text or literal pattern.
 * @param pattern_len Length of query pattern.
 * @param out_candidates Pre-allocated array to receive candidate document IDs.
 * @param max_candidates Capacity of out_candidates array.
 * @return Number of candidate document IDs placed in out_candidates.
 */
size_t keystone_trigram_index_get_candidates(
    const keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_candidates,
    size_t max_candidates
);

/**
 * @brief Execute a trigram-accelerated search with full string verification.
 * First filters candidate documents using trigram posting list intersection, then
 * verifies matches against document content.
 *
 * @param idx Finalized trigram index.
 * @param pattern Query pattern to match.
 * @param pattern_len Length of query pattern.
 * @param out_matches Pre-allocated array to receive matching document IDs.
 * @param max_matches Capacity of out_matches array.
 * @return Total number of matching documents found.
 */
size_t keystone_trigram_index_search(
    keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_matches,
    size_t max_matches
);

/**
 * @brief Get copy of trigram index statistics.
 */
void keystone_trigram_index_get_stats(
    const keystone_trigram_index_t* idx,
    keystone_trigram_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* KEYSTONE_TRIGRAM_H */
