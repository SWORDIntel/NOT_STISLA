/**
 * KEYSTONE Trigram Content Index Engine
 *
 * tgrep-inspired 24-bit trigram candidate indexing. The index is designed as
 * an accelerator: posting-list intersection proposes candidates, while an
 * authoritative caller may perform its own final verification and security
 * filtering before results are exposed.
 */

#ifndef KEYSTONE_TRIGRAM_H
#define KEYSTONE_TRIGRAM_H

/* keystone_trigram.c includes this header before any system headers. Defining
 * the POSIX feature level here keeps clock_gettime/CLOCK_MONOTONIC visible in
 * strict -std=c11 builds as well as GNU dialect builds. Do not overwrite a
 * feature level already selected by the embedding application. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable negative return codes for mutating operations. */
#define KEYSTONE_TRIGRAM_OK          0
#define KEYSTONE_TRIGRAM_EINVAL     -1
#define KEYSTONE_TRIGRAM_ENOMEM     -2
#define KEYSTONE_TRIGRAM_ESTATE     -3
#define KEYSTONE_TRIGRAM_EOVERFLOW  -4

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
 * Opaque trigram index handle.
 *
 * The concrete layout is intentionally private. Callers must not depend on
 * posting-list or document-record internals; this lets KEYSTONE strengthen
 * memory ownership and state validation without ABI-visible structure drift.
 */
typedef struct keystone_trigram_index keystone_trigram_index_t;

/**
 * @brief Create a new trigram content index.
 * @param initial_doc_capacity Expected number of documents. 0 selects a safe
 *        implementation default.
 * @return Pointer to new index, or NULL on invalid capacity / allocation
 *         failure.
 */
keystone_trigram_index_t* keystone_trigram_index_create(size_t initial_doc_capacity);

/**
 * @brief Destroy a trigram content index and free all owned memory.
 *
 * Content added through keystone_trigram_index_add_document() is owned by the
 * index and is best-effort zeroized before release.
 */
void keystone_trigram_index_destroy(keystone_trigram_index_t* idx);

/**
 * @brief Add a text document and retain an owned snapshot for exact search.
 *
 * KEYSTONE copies both the optional name and the text bytes. The caller may
 * therefore release or reuse its input buffer immediately after this function
 * returns. Any allocation/indexing failure poisons the in-progress index and
 * causes finalize/search to fail closed; a partially built posting index is
 * never exposed as authoritative.
 *
 * Documents may only be added before finalize().
 *
 * @param idx Trigram index.
 * @param name Document name or path (optional, can be NULL).
 * @param text Document content buffer.
 * @param text_len Length of text content.
 * @param out_doc_id Pointer to receive assigned document ID (optional).
 * @return KEYSTONE_TRIGRAM_OK on success, otherwise a negative error code.
 */
int keystone_trigram_index_add_document(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
);

/**
 * @brief Add a document for candidate-only indexing without retaining content.
 *
 * The supplied text is consumed synchronously to build trigram postings but is
 * NOT stored by KEYSTONE. This is the preferred mode for QIHSE and other
 * security-sensitive integrations where the authoritative datastore owns the
 * plaintext and KEYSTONE must not create a second classified-data copy.
 *
 * Full keystone_trigram_index_search() cannot verify external-only documents;
 * use get_candidates(), apply the caller's authorization policy, then perform
 * exact verification against the authoritative data source.
 */
int keystone_trigram_index_add_document_external(
    keystone_trigram_index_t* idx,
    const char* name,
    const char* text,
    size_t text_len,
    uint32_t* out_doc_id
);

/**
 * @brief Finalize the index for querying.
 *
 * Finalization fails if any prior document insertion encountered an allocation
 * or indexing failure. Once finalized, the index is immutable.
 */
int keystone_trigram_index_finalize(keystone_trigram_index_t* idx);

/**
 * @brief Extract unique 24-bit trigram keys from a text pattern.
 */
size_t keystone_trigram_extract(
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_trigrams,
    size_t max_trigrams
);

/**
 * @brief Return the number of documents currently registered in the index.
 *
 * This helper allows candidate-only callers to size output storage without
 * reaching into the opaque index layout.
 */
size_t keystone_trigram_index_document_count(const keystone_trigram_index_t* idx);

/**
 * @brief Intersect posting lists to return candidate document IDs.
 *
 * The index must be finalized and healthy. For patterns shorter than three
 * bytes, all document IDs are candidates because trigram rejection is not
 * possible. The returned IDs are candidate hints only; security-sensitive
 * callers MUST authorize and exactly verify them before disclosure.
 */
size_t keystone_trigram_index_get_candidates(
    const keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_candidates,
    size_t max_candidates
);

/**
 * @brief Execute trigram-accelerated exact substring search over owned docs.
 *
 * External-only documents are deliberately skipped because KEYSTONE does not
 * retain their content. Security-sensitive systems should normally use the
 * candidate API and verify in their authoritative datastore instead.
 */
size_t keystone_trigram_index_search(
    keystone_trigram_index_t* idx,
    const char* pattern,
    size_t pattern_len,
    uint32_t* out_matches,
    size_t max_matches
);

/**
 * @brief Get a copy of trigram index statistics.
 */
void keystone_trigram_index_get_stats(
    const keystone_trigram_index_t* idx,
    keystone_trigram_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* KEYSTONE_TRIGRAM_H */
