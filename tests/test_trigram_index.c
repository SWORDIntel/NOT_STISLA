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

    /* "hello world" (11 chars) -> 9 character trigrams:
     * "hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld" */
    TEST_ASSERT(count == 9);

    /* Test deduplication */
    const char* dup_text = "aaaaa";
    count = keystone_trigram_extract(dup_text, strlen(dup_text), trigrams, 32);
    /* "aaa" appears 3 times in sliding window, but unique count must be 1 */
    TEST_ASSERT(count == 1);

    printf("✓ Trigram extraction verified.\n");
}

static void test_trigram_index_build_and_search(void) {
    printf("Testing trigram index build and search...\n");
    keystone_trigram_index_t* idx = keystone_trigram_index_create(8);
    TEST_ASSERT(idx != NULL);

    const char* doc0 = "The quick brown fox jumps over the lazy dog";
    const char* doc1 = "KEYSTONE interpolation search engine for sorted int64";
    const char* doc2 = "Trigram index accelerates regex search across text files";
    const char* doc3 = "High-performance SIMD search and data ingestion";

    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc0.txt", doc0, strlen(doc0), NULL) == 0);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc1.txt", doc1, strlen(doc1), NULL) == 0);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc2.txt", doc2, strlen(doc2), NULL) == 0);
    TEST_ASSERT(keystone_trigram_index_add_document(idx, "doc3.txt", doc3, strlen(doc3), NULL) == 0);

    TEST_ASSERT(keystone_trigram_index_finalize(idx) == 0);

    /* Test candidate filtering and search for "KEYSTONE" -> doc1 */
    uint32_t matches[8];
    size_t match_count = keystone_trigram_index_search(idx, "KEYSTONE", 8, matches, 8);
    TEST_ASSERT(match_count == 1);
    TEST_ASSERT(matches[0] == 1);

    /* Search for "search" -> doc1, doc2, doc3 */
    match_count = keystone_trigram_index_search(idx, "search", 6, matches, 8);
    TEST_ASSERT(match_count == 3);

    /* Search for "nonexistent_pattern" -> 0 matches */
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
    test_trigram_index_build_and_search();

    printf("\n🎉 All Trigram Index tests passed!\n");
    return 0;
}
