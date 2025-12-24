/**
 * DSMIL NOT_Competitor Benchmark Suite
 *
 * Simple performance verification for NOT_Competitor
 */

#include "../include/not_stisla.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

/* Timing utilities */
static inline uint64_t ns_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000ULL + (uint64_t)tv.tv_usec * 1000ULL;
}

/* Binary search for comparison */
static size_t bin_search(const int64_t* arr, size_t n, int64_t key) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < key) {
            lo = mid + 1;
        } else if (arr[mid] > key) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return SIZE_MAX;
}

/* Generate uniform test data */
static void generate_test_data(int64_t* arr, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        arr[i] = (int64_t)i * 2;  // 0, 2, 4, 6, ... (uniform)
    }
}

int main() {
    printf("🎯 DSMIL NOT_Competitor Benchmark Suite\n");
    printf("Version: %s\n", not_stisla_version());
    printf("Build: %s\n", not_stisla_build_info());
    printf("\n");

    const size_t DATA_SIZE = 100000;
    const size_t NUM_QUERIES = 50000;

    /* Generate test data */
    int64_t* data = malloc(DATA_SIZE * sizeof(int64_t));
    int64_t* queries = malloc(NUM_QUERIES * sizeof(int64_t));
    assert(data && queries && "Failed to allocate memory");

    generate_test_data(data, DATA_SIZE);

    /* Generate queries (all exist in data) */
    srand(42);
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        size_t idx = rand() % DATA_SIZE;
        queries[i] = data[idx];
    }

    /* Initialize NOT_Competitor */
    not_stisla_anchor_table_t* table = not_stisla_anchor_table_create();
    assert(table && "Failed to create anchor table");

    /* Warm-up */
    for (size_t i = 0; i < 1000; ++i) {
        not_stisla_search(data, DATA_SIZE, queries[i], table, 8);
    }

    /* Benchmark binary search */
    uint64_t bin_start = ns_now();
    size_t bin_found = 0;
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        if (bin_search(data, DATA_SIZE, queries[i]) != SIZE_MAX) {
            bin_found++;
        }
    }
    uint64_t bin_time = ns_now() - bin_start;

    /* Benchmark NOT_Competitor */
    uint64_t not_stisla_start = ns_now();
    size_t not_stisla_found = 0;
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        if (not_stisla_search(data, DATA_SIZE, queries[i], table, 8) != NOT_Competitor_NOT_FOUND) {
            not_stisla_found++;
        }
    }
    uint64_t not_stisla_time = ns_now() - not_stisla_start;

    /* Results */
    double bin_ns_per_op = (double)bin_time / NUM_QUERIES;
    double not_stisla_ns_per_op = (double)not_stisla_time / NUM_QUERIES;
    double speedup = bin_ns_per_op / not_stisla_ns_per_op;

    printf("📊 Performance Results:\n");
    printf("Binary Search:     %.1f ns/op (%zu found)\n", bin_ns_per_op, bin_found);
    printf("NOT_Competitor:        %.1f ns/op (%zu found)\n", not_stisla_ns_per_op, not_stisla_found);
    printf("Speedup:           %.2fx faster than binary search\n", speedup);

    /* Statistics */
    size_t searches, anchors, memory;
    not_stisla_get_stats(table, &searches, &anchors, &memory);
    printf("\n📈 NOT_Competitor Statistics:\n");
    printf("Searches performed: %zu\n", searches);
    printf("Anchors learned:    %zu\n", anchors);
    printf("Memory usage:       %zu bytes\n", memory);

    printf("\n✅ Benchmark completed successfully!\n");
    printf("NOT_Competitor delivers %.1fx actual speedup\n", speedup);

    /* Cleanup */
    not_stisla_anchor_table_destroy(table);
    free(queries);
    free(data);

    return 0;
}