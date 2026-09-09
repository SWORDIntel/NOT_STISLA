#include "../include/keystone_trigram.h"
#include "test_macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_trigram_extraction(void) {
    printf("Testing trigram extraction...\n");
    const char* text = "hello world";
    uint32_t trigrams[32];
    size_t count = keystone_trigram_extract(text, strlen(text), trigrams, 32);

    TEST_ASSERT(count == 9);

    const char* dup_text = "aaaaa";
    count = keystone_trigram_extract(dup_text, strlen(dup_text), trigrams, 32);
    TEST_ASSERT(count == 1);

    printf("✓ Trigram extraction verified.\n");
}

static void test_owned_snapshot_survives_caller_mutation(void) {
    printf("Testing owned document snapshot...\n");
    keystone_trigram_index_t* idx = keystone_trigram_index_create(2);
    TEST_ASSERT(idx != NULL);

    char mutable_doc[64] = "classified-looking source buffer";
    TEST_ASSERT(keystone_trigram_index_add_document(
                    idx, "owned", mutable_doc, strlen(mutable_doc), NULL) == KEYSTONE_TRIGRAM_OK);

    memset(mutable_doc, 'X', strlen(mutable_doc));
    TEST_ASSERT(keystone_trigram_index_finalize(idx) == KEYSTONE_TRIGRAM_OK);

    uint32_t matches[4];
    size_t count = keystone_trigram_index_search(
        idx, "source buffer", strlen("source buffer"), matches, 4);
    TEST_ASSERT(count == 1);
    TEST_ASSERT(matches[0] == 0);

    keystone_trigram_index_destroy(idx);
    printf("✓ Owned snapshot lifetime verified.\n");
}

static void test_external_candidate_only_mode(void) {
    printf("Testing external candidate-only mode...\n");
    keystone_trigram_index_t* idx = keystone_trigram_index_create(2);
    TEST_ASSERT(idx != NULL);

    char external_doc[64] = "qihse authoritative plaintext record";
    uint32_t doc_id = UINT32_MAX;
    TEST_ASSERT(keystone_trigram_index_add_document_external(
                    idx, "external", external_doc, strlen(external_doc), &doc_id) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(doc_id == 0);
    TEST_ASSERT(keystone_trigram_index_document_count(idx) == 1);

    memset(external_doc, 0, sizeof(external_doc));
    TEST_ASSERT(keystone_trigram_index_finalize(idx) == KEYSTONE_TRIGRAM_OK);

    uint32_t candidates[4];
    size_t candidate_count = keystone_trigram_index_get_candidates(
        idx, "authoritative", strlen("authoritative"), candidates, 4);
    TEST_ASSERT(candidate_count == 1);
    TEST_ASSERT(candidates[0] == 0);

    /* Exact search deliberately cannot inspect external-only plaintext. */
    uint32_t matches[4];
    TEST_ASSERT(keystone_trigram_index_search(
                    idx, "authoritative", strlen("authoritative"), matches, 4) == 0);

    keystone_trigram_index_destroy(idx);
    printf("✓ Candidate-only plaintext separation verified.\n");
}

static void test_finalize_is_security_boundary(void) {
    printf("Testing finalize state boundary...\n");
    keystone_trigram_index_t* idx = keystone_trigram_index_create(1);
    TEST_ASSERT(idx != NULL);

    const char* doc = "immutable after finalize";
    TEST_ASSERT(keystone_trigram_index_add_document(
                    idx, "doc", doc, strlen(doc), NULL) == KEYSTONE_TRIGRAM_OK);

    uint32_t candidates[2];
    TEST_ASSERT(keystone_trigram_index_get_candidates(
                    idx, "immutable", strlen("immutable"), candidates, 2) == 0);

    TEST_ASSERT(keystone_trigram_index_finalize(idx) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(keystone_trigram_index_add_document(
                    idx, "late", "late mutation", strlen("late mutation"), NULL) == KEYSTONE_TRIGRAM_ESTATE);

    keystone_trigram_index_destroy(idx);
    printf("✓ Finalize state boundary verified.\n");
}

static void test_trigram_index_build_and_search(void) {
    printf("Testing trigram index build and search...\n");
    keystone_trigram_index_t* idx = keystone_trigram_index_create(8);
    TEST_ASSERT(idx != NULL);

    const char* doc0 = "The quick brown fox jumps over the lazy dog";
    const char* doc1 = "KEYSTONE interpolation search engine for sorted int64";
    const char* doc2 = "Trigram index accelerates regex search across text files";
    const char* doc3 = "High-performance SIMD search and data ingestion";

    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc0.txt", doc0, strlen(doc0), NULL) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc1.txt", doc1, strlen(doc1), NULL) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc2.txt", doc2, strlen(doc2), NULL) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc3.txt", doc3, strlen(doc3), NULL) == KEYSTONE_TRIGRAM_OK);

    TEST_ASSERT(keystone_trigram_index_finalize(idx) == KEYSTONE_TRIGRAM_OK);
    TEST_ASSERT(keystone_trigram_index_document_count(idx) == 4);

    uint32_t matches[8];
    size_t match_count = keystone_trigram_index_search(idx, "KEYSTONE", 8, matches, 8);
    TEST_ASSERT(match_count == 1);
    TEST_ASSERT(matches[0] == 1);

    match_count = keystone_trigram_index_search(idx, "search", 6, matches, 8);
    TEST_ASSERT(match_count == 3);

    match_count = keystone_trigram_index_search(idx, "nonexistent_pattern", 19, matches, 8);
    TEST_ASSERT(match_count == 0);

    keystone_trigram_stats_t stats;
    keystone_trigram_index_get_stats(idx, &stats);
    TEST_ASSERT(stats.total_documents == 4);
    TEST_ASSERT(stats.total_searches == 3);
    TEST_ASSERT(stats.candidate_docs_rejected > 0);

    keystone_trigram_index_destroy(idx);
    printf("✓ Trigram index build and search verified.\n");
}

int main(void) {
    printf("Running Trigram Index Test Suite\n");
    printf("===============================\n\n");

    test_trigram_extraction();
    test_owned_snapshot_survives_caller_mutation();
    test_external_candidate_only_mode();
    test_finalize_is_security_boundary();
    test_trigram_index_build_and_search();

    printf("\nAll Trigram Index tests passed.\n");
    return 0;
}
