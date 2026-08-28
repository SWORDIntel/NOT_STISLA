/*
 * SSE4.2 vs scalar benchmark for AVX1-only CPUs.
 * Measures the impact of the new SSE4.2 SIMD path on chunked_search
 * and the overall search_batch pipeline.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "keystone.h"

#define N_ARRAY   100000
#define N_QUERIES 100000
#define N_ROUNDS  20

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    /* Build a sorted array of random int64s */
    int64_t* arr = malloc(N_ARRAY * sizeof(int64_t));
    srand(42);
    for (size_t i = 0; i < N_ARRAY; i++) arr[i] = ((int64_t)rand() << 32) | rand();
    /* Sort */
    for (size_t i = 1; i < N_ARRAY; i++) {
        int64_t v = arr[i]; size_t j = i;
        while (j > 0 && arr[j-1] > v) { arr[j] = arr[j-1]; j--; }
        arr[j] = v;
    }

    /* Build query set: 50% hits, 50% misses */
    int64_t* queries = malloc(N_QUERIES * sizeof(int64_t));
    for (size_t i = 0; i < N_QUERIES; i++) {
        if (i % 2 == 0) queries[i] = arr[rand() % N_ARRAY];
        else            queries[i] = ((int64_t)rand() << 32) | rand();
    }

    /* Detect CPU features */
    uint32_t feat = keystone_detect_cpu_features();
    printf("CPU features: 0x%08X\n", feat);
    printf("  AVX:    %s\n", (feat & KEYSTONE_CPU_AVX)    ? "yes" : "no");
    printf("  AVX2:   %s\n", (feat & KEYSTONE_CPU_AVX2)   ? "yes" : "no");
    printf("  AVX512: %s\n", (feat & KEYSTONE_CPU_AVX512) ? "yes" : "no");
    printf("  SSE42:  %s\n", (feat & KEYSTONE_CPU_SSE42)  ? "yes" : "no");
    printf("\n");

    /* --- Benchmark single-key search (keystone_search) --- */
    /* Warmup */
    for (size_t i = 0; i < 1000; i++) keystone_search(arr, N_ARRAY, queries[i % N_QUERIES], NULL, 4);

    double t0 = now_sec();
    size_t found_total = 0;
    for (int r = 0; r < N_ROUNDS; r++) {
        for (size_t i = 0; i < N_QUERIES; i++) {
            if (keystone_search(arr, N_ARRAY, queries[i], NULL, 4) != KEYSTONE_NOT_FOUND)
                found_total++;
        }
    }
    double t1 = now_sec();
    double single_ns = (t1 - t0) / (N_ROUNDS * N_QUERIES) * 1e9;
    printf("Single-key search: %.1f ns/query (%zu hits in %d rounds of %d queries)\n",
           single_ns, found_total, N_ROUNDS, N_QUERIES);

    /* --- Benchmark batch search (keystone_search_batch) --- */
    keystone_batch_item_t* items = malloc(N_QUERIES * sizeof(keystone_batch_item_t));
    for (size_t i = 0; i < N_QUERIES; i++) {
        items[i].key = queries[i];
        items[i].ordinal = i;
        items[i].result = KEYSTONE_NOT_FOUND;
    }

    /* Warmup */
    keystone_search_batch(arr, N_ARRAY, items, 100, NULL, 4);

    t0 = now_sec();
    size_t batch_found = 0;
    for (int r = 0; r < N_ROUNDS; r++) {
        batch_found += keystone_search_batch(arr, N_ARRAY, items, N_QUERIES, NULL, 4);
    }
    t1 = now_sec();
    double batch_ns = (t1 - t0) / (N_ROUNDS * N_QUERIES) * 1e9;
    printf("Batch search:      %.1f ns/query (%zu hits/round)\n",
           batch_ns, batch_found / N_ROUNDS);

    /* --- Benchmark zero-copy batch (keystone_search_keys_batch_auto) --- */
    size_t* results = malloc(N_QUERIES * sizeof(size_t));
    t0 = now_sec();
    size_t zc_found = 0;
    for (int r = 0; r < N_ROUNDS; r++) {
        zc_found += keystone_search_keys_batch_auto(arr, N_ARRAY, queries, N_QUERIES, results, NULL, 4, NULL);
    }
    t1 = now_sec();
    double zc_ns = (t1 - t0) / (N_ROUNDS * N_QUERIES) * 1e9;
    printf("Zero-copy batch:   %.1f ns/query (%zu hits/round)\n",
           zc_ns, zc_found / N_ROUNDS);

    /* --- Benchmark auto batch with OpenMP (keystone_search_batch_auto) --- */
    keystone_parallel_config_t omp_cfg = {0};
    omp_cfg.num_threads = 0;  /* auto-detect */
    omp_cfg.use_thread_pool = 1;
    omp_cfg.batch_chunk = 64;
    t0 = now_sec();
    size_t omp_found = 0;
    for (int r = 0; r < N_ROUNDS; r++) {
        omp_found += keystone_search_batch_auto(arr, N_ARRAY, items, N_QUERIES, NULL, 4, &omp_cfg);
    }
    t1 = now_sec();
    double omp_ns = (t1 - t0) / (N_ROUNDS * N_QUERIES) * 1e9;
    printf("Auto+OpenMP batch: %.1f ns/query (%zu hits/round)\n",
           omp_ns, omp_found / N_ROUNDS);

    /* --- Benchmark small-window linear scan (local_search path) --- */
    /* This exercises the SSE4.2 chunked_search path directly for small windows */
    int64_t small_arr[64];
    for (size_t i = 0; i < 64; i++) small_arr[i] = (int64_t)i * 2;
    size_t small_found = 0;
    t0 = now_sec();
    for (int r = 0; r < 100000; r++) {
        for (int64_t k = 0; k < 128; k++) {
            if (keystone_search(small_arr, 64, k, NULL, 4) != KEYSTONE_NOT_FOUND)
                small_found++;
        }
    }
    t1 = now_sec();
    double small_ns = (t1 - t0) / (100000 * 128) * 1e9;
    printf("Small-window scan:  %.1f ns/query (64-element array, 128 keys)\n", small_ns);

    free(arr); free(queries); free(items); free(results);
    return 0;
}
