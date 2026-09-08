#include "../include/keystone_trigram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("====================================================\n");
    printf("  KEYSTONE Trigram Search Benchmark (tgrep-style)   \n");
    printf("====================================================\n\n");

    const size_t NUM_DOCS = 10000;
    const size_t DOC_SIZE = 1024; /* 1KB per doc -> 10MB corpus */
    const char* needle = "CRITICAL_SECURITY_ALERT_KEYSTONE_99";
    size_t needle_len = strlen(needle);

    printf("Generating corpus: %zu documents (%zu KB total)... ", NUM_DOCS, (NUM_DOCS * DOC_SIZE) / 1024);
    fflush(stdout);

    char** docs = (char**)malloc(NUM_DOCS * sizeof(char*));
    for (size_t i = 0; i < NUM_DOCS; i++) {
        docs[i] = (char*)malloc(DOC_SIZE + 1);
        for (size_t j = 0; j < DOC_SIZE; j++) {
            docs[i][j] = 'a' + (rand() % 26);
        }
        docs[i][DOC_SIZE] = '\0';
    }

    /* Inject target needle into 5 documents */
    size_t target_doc_ids[5] = {123, 1456, 4321, 7890, 9500};
    for (int k = 0; k < 5; k++) {
        size_t id = target_doc_ids[k];
        memcpy(docs[id] + 100, needle, needle_len);
    }
    printf("Done.\n");

    /* 1. Baseline Brute-Force Scan */
    printf("\n[1] Running Brute-Force Scan over %zu files...\n", NUM_DOCS);
    double start_time = get_time_sec();
    size_t brute_matches = 0;
    for (size_t i = 0; i < NUM_DOCS; i++) {
        if (memmem(docs[i], DOC_SIZE, needle, needle_len) != NULL) {
            brute_matches++;
        }
    }
    double brute_time = get_time_sec() - start_time;
    printf("    Matches found: %zu\n", brute_matches);
    printf("    Brute-force latency: %.4f ms (%.2f docs/ms)\n", brute_time * 1000.0, NUM_DOCS / (brute_time * 1000.0));

    /* 2. Build Trigram Index */
    printf("\n[2] Building Trigram Index...\n");
    start_time = get_time_sec();
    keystone_trigram_index_t* idx = keystone_trigram_index_create(NUM_DOCS);
    for (size_t i = 0; i < NUM_DOCS; i++) {
        keystone_trigram_index_add_document(idx, NULL, docs[i], DOC_SIZE, NULL);
    }
    keystone_trigram_index_finalize(idx);
    double build_time = get_time_sec() - start_time;
    printf("    Index built in: %.4f s\n", build_time);

    /* 3. Trigram Accelerated Query Search */
    printf("\n[3] Executing Trigram Accelerated Search...\n");
    uint32_t matches[64];
    start_time = get_time_sec();
    size_t tri_matches = keystone_trigram_index_search(idx, needle, needle_len, matches, 64);
    double tri_time = get_time_sec() - start_time;

    printf("    Matches found: %zu\n", tri_matches);
    printf("    Trigram search latency: %.4f ms\n", tri_time * 1000.0);

    keystone_trigram_stats_t stats;
    keystone_trigram_index_get_stats(idx, &stats);
    printf("    Candidate docs evaluated: %zu / %zu (rejection rate: %.2f%%)\n",
           stats.candidate_docs_evaluated, NUM_DOCS,
           ((double)stats.candidate_docs_rejected / NUM_DOCS) * 100.0);

    double speedup = brute_time / tri_time;
    printf("\n====================================================\n");
    printf("  SPEEDUP: %.2fx FASTER search with Trigram Index   \n", speedup);
    printf("====================================================\n");

    /* Cleanup */
    keystone_trigram_index_destroy(idx);
    for (size_t i = 0; i < NUM_DOCS; i++) {
        free(docs[i]);
    }
    free(docs);

    return 0;
}
