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

static const void* brute_memmem(const void* haystack, size_t haystack_len,
                                const void* needle, size_t needle_len) {
    const unsigned char* h = (const unsigned char*)haystack;
    const unsigned char* n = (const unsigned char*)needle;
    if (!haystack || !needle) return NULL;
    if (needle_len == 0u) return haystack;
    if (haystack_len < needle_len) return NULL;
    for (size_t i = 0u; i <= haystack_len - needle_len; i++) {
        if (h[i] == n[0] && memcmp(h + i, n, needle_len) == 0) return h + i;
    }
    return NULL;
}

typedef struct {
    const char* label;
    size_t total_bytes;
    size_t num_docs;
    size_t doc_size;
} bench_config_t;

static void run_benchmark(const bench_config_t* cfg) {
    const char* needle = "CRITICAL_SECURITY_ALERT_KEYSTONE_99";
    size_t needle_len = strlen(needle);

    printf("┌──────────────────────────────────────────────────────────┐\n");
    printf("│  Corpus: %-48s│\n", cfg->label);
    printf("│  Documents: %-8zu  Doc size: %-8zu bytes          │\n",
           cfg->num_docs, cfg->doc_size);
    printf("│  Total corpus: %-8zu MB                                │\n",
           cfg->total_bytes / (1024 * 1024));
    printf("└──────────────────────────────────────────────────────────┘\n\n");

    printf("  Generating corpus... ");
    fflush(stdout);

    char** docs = (char**)malloc(cfg->num_docs * sizeof(char*));
    if (!docs) { printf("FAILED (alloc)\n"); return; }

    for (size_t i = 0; i < cfg->num_docs; i++) {
        docs[i] = (char*)malloc(cfg->doc_size + 1);
        if (!docs[i]) {
            for (size_t j = 0; j < i; j++) free(docs[j]);
            free(docs);
            printf("FAILED (alloc doc %zu)\n", i);
            return;
        }
        for (size_t j = 0; j < cfg->doc_size; j++) {
            docs[i][j] = 'a' + (rand() % 26);
        }
        docs[i][cfg->doc_size] = '\0';
    }

    size_t num_targets = cfg->num_docs > 10000 ? 50 : 5;
    size_t* target_ids = (size_t*)malloc(num_targets * sizeof(size_t));
    for (size_t k = 0; k < num_targets; k++) {
        target_ids[k] = (k + 1) * (cfg->num_docs / (num_targets + 1));
        if (target_ids[k] >= cfg->num_docs) target_ids[k] = cfg->num_docs - 1;
        size_t offset = 100;
        if (offset + needle_len < cfg->doc_size)
            memcpy(docs[target_ids[k]] + offset, needle, needle_len);
    }
    printf("Done. (%zu targets injected)\n", num_targets);

    /* 1. Brute-force */
    printf("  [1] Brute-force scan... ");
    fflush(stdout);
    double t0 = get_time_sec();
    size_t brute_matches = 0;
    for (size_t i = 0; i < cfg->num_docs; i++) {
        if (brute_memmem(docs[i], cfg->doc_size, needle, needle_len))
            brute_matches++;
    }
    double brute_time = get_time_sec() - t0;
    printf("%.2f ms  (%zu matches, %.0f MB/s)\n",
           brute_time * 1000.0, brute_matches,
           (double)cfg->total_bytes / (1024.0 * 1024.0) / brute_time);

    /* 2. Build index */
    printf("  [2] Building trigram index... ");
    fflush(stdout);
    t0 = get_time_sec();
    keystone_trigram_index_t* idx = keystone_trigram_index_create(cfg->num_docs);
    for (size_t i = 0; i < cfg->num_docs; i++) {
        keystone_trigram_index_add_document(idx, NULL, docs[i], cfg->doc_size, NULL);
    }
    keystone_trigram_index_finalize(idx);
    double build_time = get_time_sec() - t0;
    printf("%.2f s  (%.0f MB/s indexing throughput)\n",
           build_time,
           (double)cfg->total_bytes / (1024.0 * 1024.0) / build_time);

    /* 3. Trigram search */
    printf("  [3] Trigram search... ");
    fflush(stdout);
    size_t max_matches = num_targets + 64;
    uint32_t* matches = (uint32_t*)malloc(max_matches * sizeof(uint32_t));
    t0 = get_time_sec();
    size_t tri_matches = keystone_trigram_index_search(idx, needle, needle_len,
                                                       matches, max_matches);
    double tri_time = get_time_sec() - t0;
    printf("%.4f ms  (%zu matches)\n", tri_time * 1000.0, tri_matches);

    keystone_trigram_stats_t stats;
    keystone_trigram_index_get_stats(idx, &stats);
    printf("      Candidates: %zu / %zu (%.2f%% rejected)\n",
           stats.candidate_docs_evaluated, cfg->num_docs,
           cfg->num_docs > 0 ?
           ((double)stats.candidate_docs_rejected / cfg->num_docs) * 100.0 : 0.0);
    printf("      Unique trigrams: %zu  Total postings: %zu\n",
           stats.unique_trigrams, stats.total_postings);

    double speedup = brute_time / tri_time;
    printf("  >> SPEEDUP: %.1fx  (build: %.2fs, search: %.4fms)\n\n",
           speedup, build_time, tri_time * 1000.0);

    keystone_trigram_index_destroy(idx);
    free(matches);
    free(target_ids);
    for (size_t i = 0; i < cfg->num_docs; i++) free(docs[i]);
    free(docs);
}

int main(void) {
    printf("======================================================\n");
    printf("  KEYSTONE Trigram Benchmark — Multi-Corpus Sizes     \n");
    printf("  CPU: Sandy Bridge (SSE4.2 + AVX1)                   \n");
    printf("======================================================\n\n");

    srand(42);

    bench_config_t configs[] = {
        { "1 MB corpus",    1ull   * 1024 * 1024,   256,    4096 },
        { "10 MB corpus",   10ull  * 1024 * 1024,   2560,   4096 },
        { "100 MB corpus",  100ull * 1024 * 1024,   25600,  4096 },
        { "1 GB corpus",    1024ull* 1024 * 1024,   262144, 4096 },
    };
    int num_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int i = 0; i < num_configs; i++) {
        run_benchmark(&configs[i]);
    }

    printf("======================================================\n");
    printf("  Benchmark complete.                                  \n");
    printf("======================================================\n");

    return 0;
}
