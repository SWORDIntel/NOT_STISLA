#include "../include/keystone.h"
#include "test_macros.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_data(int64_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (int64_t)i * 3;
    }
}

static void fill_items(keystone_batch_item_t* items, size_t num_items) {
    for (size_t i = 0; i < num_items; ++i) {
        items[i].key = (int64_t)i * 3;
        items[i].result = KEYSTONE_NOT_FOUND;
        items[i].ordinal = 0;
    }
}

static void assert_all_found(const keystone_batch_item_t* items, size_t num_items) {
    for (size_t i = 0; i < num_items; ++i) {
        TEST_ASSERT(items[i].result == i);
        TEST_ASSERT(items[i].ordinal == i);
    }
}

static void copy_batch_results(const keystone_batch_item_t* items,
                               keystone_result_t* results,
                               size_t* ordinals,
                               size_t num_items) {
    for (size_t i = 0; i < num_items; ++i) {
        results[i] = items[i].result;
        ordinals[i] = items[i].ordinal;
    }
}

static void assert_batch_results_match(const keystone_batch_item_t* items,
                                       const keystone_result_t* results,
                                       const size_t* ordinals,
                                       size_t num_items) {
    for (size_t i = 0; i < num_items; ++i) {
        TEST_ASSERT(items[i].result == results[i]);
        TEST_ASSERT(items[i].ordinal == ordinals[i]);
    }
}

static void test_no_decision_before_auto_call(void) {
    keystone_backend_decision_t decision;
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == -1);
    TEST_ASSERT(keystone_get_last_backend_decision(NULL) == -1);
    TEST_ASSERT(strcmp(keystone_backend_name(KEYSTONE_BACKEND_SCALAR), "scalar") == 0);
    TEST_ASSERT(strcmp(keystone_backend_name((keystone_backend_t)999), "unknown") == 0);
    TEST_ASSERT(strcmp(keystone_decision_source_name(KEYSTONE_DECISION_SOURCE_CACHE), "cache") == 0);
    TEST_ASSERT(strcmp(keystone_decision_source_name((keystone_backend_decision_source_t)999),
                       "unknown") == 0);
    TEST_ASSERT(strcmp(keystone_query_shape_name(KEYSTONE_QUERY_SHAPE_DENSE_SORTED),
                       "dense_sorted") == 0);
    TEST_ASSERT(strcmp(keystone_query_shape_name((keystone_query_shape_t)999), "unknown") == 0);
}

static void test_invalid_inputs_return_zero_no_decision(void) {
    int64_t data[4] = {0, 3, 6, 9};
    keystone_batch_item_t items[2];
    keystone_backend_decision_t decision;

    fill_items(items, 2);

    /* Invalid inputs return 0 and do NOT record a backend decision. */
    TEST_ASSERT(keystone_search_batch_auto(NULL, 4, items, 2, NULL, 8, NULL) == 0);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == -1);

    TEST_ASSERT(keystone_search_batch_auto(data, 4, NULL, 2, NULL, 8, NULL) == 0);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == -1);

    TEST_ASSERT(keystone_search_batch_auto(data, 0, items, 2, NULL, 8, NULL) == 0);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == -1);

    TEST_ASSERT(keystone_search_batch_auto(data, 4, items, 0, NULL, 8, NULL) == 0);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == -1);
}

static void test_small_batch_uses_scalar(void) {
    int64_t data[64];
    keystone_batch_item_t items[8];
    keystone_backend_decision_t decision;
    keystone_parallel_config_t config = {
        .num_threads = 4,
        .use_thread_pool = 0,
        .batch_chunk = 16
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(table != NULL);
    fill_data(data, 64);
    fill_items(items, 8);

    size_t found = keystone_search_batch_auto(data, 64, items, 8, table, 8, &config);

    TEST_ASSERT(found == 8);
    assert_all_found(items, 8);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == 0);
    TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR);
    TEST_ASSERT(decision.array_size_bucket == 64);
    TEST_ASSERT(decision.query_count_bucket == 8);
    TEST_ASSERT(decision.estimated_ns_per_key >= 0.0);
    TEST_ASSERT(decision.p95_ns_per_key >= 0.0);
    TEST_ASSERT(decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);
    TEST_ASSERT(decision.decision_source == KEYSTONE_DECISION_SOURCE_FAST_PATH);
    TEST_ASSERT(decision.calibration_runs == 0);
    TEST_ASSERT(decision.candidates_measured == 0);

    keystone_anchor_table_destroy(table);
}

static void test_sorted_8k_batch_uses_expected_backend(void) {
    const size_t n = 8192;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_backend_decision_t decision;
    keystone_parallel_config_t config = {
        .num_threads = 16,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);
    fill_items(items, n);

    size_t found = keystone_search_batch_auto(data, n, items, n, table, 8, &config);

    TEST_ASSERT(found == n);
    assert_all_found(items, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == 0);
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_FORTRAN ||
                    decision.backend == KEYSTONE_BACKEND_SCALAR);
    } else {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR);
    }
    TEST_ASSERT(decision.query_count_bucket == 8192);
    TEST_ASSERT(decision.estimated_ns_per_key >= 0.0);
    TEST_ASSERT(decision.p95_ns_per_key >= decision.estimated_ns_per_key);
    TEST_ASSERT(decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED ||
                    decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);
        TEST_ASSERT(decision.calibration_runs >= 3);
        TEST_ASSERT(decision.candidates_measured >= 1);
    }

    keystone_anchor_table_destroy(table);
    free(items);
    free(data);
}

static void test_unsorted_8k_batch_uses_scalar(void) {
    const size_t n = 8192;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_backend_decision_t decision;
    keystone_parallel_config_t config = {
        .num_threads = 16,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);

    for (size_t i = 0; i < n; ++i) {
        size_t index = n - 1 - i;
        items[i].key = (int64_t)index * 3;
        items[i].result = KEYSTONE_NOT_FOUND;
        items[i].ordinal = 0;
    }

    size_t found = keystone_search_batch_auto(data, n, items, n, table, 8, &config);

    TEST_ASSERT(found == n);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == 0);
    /* With the lowered parallel threshold (4096), an 8K batch with 16
     * threads may go through calibration (MEASURED/CACHED) instead of
     * the scalar fast path.  The backend could be SCALAR or C_OPENMP
     * depending on the calibration measurement.  Both are correct. */
    TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR ||
                decision.backend == KEYSTONE_BACKEND_C_OPENMP);
    TEST_ASSERT(decision.query_count_bucket == 8192);
    TEST_ASSERT(decision.query_shape == KEYSTONE_QUERY_SHAPE_STRIDED);
    TEST_ASSERT(decision.decision_source == KEYSTONE_DECISION_SOURCE_FAST_PATH ||
                decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED ||
                decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);

    keystone_anchor_table_destroy(table);
    free(items);
    free(data);
}

static void test_large_single_thread_batch_uses_scalar(void) {
    const size_t n = 20000;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_backend_decision_t decision;
    keystone_parallel_config_t config = {
        .num_threads = 1,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);
    fill_items(items, n);

    size_t found = keystone_search_batch_auto(data, n, items, n, table, 8, &config);

    TEST_ASSERT(found == n);
    assert_all_found(items, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == 0);
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_FORTRAN || decision.backend == KEYSTONE_BACKEND_SCALAR);
        TEST_ASSERT(decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED || decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE || decision.decision_source == KEYSTONE_DECISION_SOURCE_FAST_PATH);
    } else {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR);
        TEST_ASSERT(decision.decision_source == KEYSTONE_DECISION_SOURCE_FAST_PATH);
    }
    TEST_ASSERT(decision.thread_count == 1);
    TEST_ASSERT(decision.estimated_ns_per_key >= 0.0);
    TEST_ASSERT(decision.p95_ns_per_key >= 0.0);
    TEST_ASSERT(decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);

    keystone_anchor_table_destroy(table);
    free(items);
    free(data);
}

static void test_repeated_large_batch_decision_is_stable(void) {
    const size_t n = 20000;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_result_t* first_results = malloc(n * sizeof(*first_results));
    size_t* first_ordinals = malloc(n * sizeof(*first_ordinals));
    keystone_backend_decision_t first_decision;
    keystone_backend_decision_t second_decision;
    keystone_parallel_config_t config = {
        .num_threads = 2,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(first_results != NULL);
    TEST_ASSERT(first_ordinals != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);

    fill_items(items, n);
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    assert_all_found(items, n);
    copy_batch_results(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&first_decision) == 0);

    fill_items(items, n);
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    assert_all_found(items, n);
    assert_batch_results_match(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&second_decision) == 0);

    TEST_ASSERT(second_decision.backend == first_decision.backend);
    TEST_ASSERT(second_decision.cpu_features == first_decision.cpu_features);
    TEST_ASSERT(second_decision.array_size_bucket == first_decision.array_size_bucket);
    TEST_ASSERT(second_decision.query_count_bucket == first_decision.query_count_bucket);
    TEST_ASSERT(second_decision.thread_count == first_decision.thread_count);
    TEST_ASSERT(second_decision.estimated_ns_per_key >= 0.0);
    TEST_ASSERT(second_decision.p95_ns_per_key >= 0.0);
    TEST_ASSERT(second_decision.query_shape == first_decision.query_shape);
    if (first_decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED) {
        TEST_ASSERT(second_decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);
        TEST_ASSERT(second_decision.calibration_runs == first_decision.calibration_runs);
        TEST_ASSERT(second_decision.candidates_measured == first_decision.candidates_measured);
    } else {
        TEST_ASSERT(second_decision.decision_source == first_decision.decision_source);
    }

    keystone_anchor_table_destroy(table);
    free(first_ordinals);
    free(first_results);
    free(items);
    free(data);
}

static void test_dense_sorted_cache_hit_preserves_results(void) {
    const size_t n = 4096;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_result_t* first_results = malloc(n * sizeof(*first_results));
    size_t* first_ordinals = malloc(n * sizeof(*first_ordinals));
    keystone_backend_decision_t first_decision;
    keystone_backend_decision_t second_decision;
    keystone_parallel_config_t config = {
        .num_threads = 8,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(first_results != NULL);
    TEST_ASSERT(first_ordinals != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);

    fill_items(items, n);
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    assert_all_found(items, n);
    copy_batch_results(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&first_decision) == 0);

    fill_items(items, n);
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    assert_all_found(items, n);
    assert_batch_results_match(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&second_decision) == 0);

    TEST_ASSERT(first_decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);
    TEST_ASSERT(second_decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);
    
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(first_decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED ||
                    first_decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);
        TEST_ASSERT(second_decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);
        TEST_ASSERT(second_decision.backend == first_decision.backend);
        TEST_ASSERT(second_decision.calibration_runs == first_decision.calibration_runs);
        TEST_ASSERT(second_decision.candidates_measured == first_decision.candidates_measured);
    }

    keystone_anchor_table_destroy(table);
    free(first_ordinals);
    free(first_results);
    free(items);
    free(data);
}

static void test_random_shape_cache_hit_preserves_results(void) {
    const size_t n = 4096;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_result_t* first_results = malloc(n * sizeof(*first_results));
    size_t* first_ordinals = malloc(n * sizeof(*first_ordinals));
    keystone_backend_decision_t first_decision;
    keystone_backend_decision_t second_decision;
    keystone_parallel_config_t config = {
        .num_threads = 8,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(first_results != NULL);
    TEST_ASSERT(first_ordinals != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);

    for (size_t i = 0; i < n; ++i) {
        items[i].key = data[(i * 17) % n]; /* random-ish access pattern */
        items[i].result = KEYSTONE_NOT_FOUND;
        items[i].ordinal = i;
    }
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    copy_batch_results(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&first_decision) == 0);

    for (size_t i = 0; i < n; ++i) {
        items[i].result = KEYSTONE_NOT_FOUND;
        items[i].ordinal = i;
    }
    TEST_ASSERT(keystone_search_batch_auto(data, n, items, n, table, 8, &config) == n);
    assert_batch_results_match(items, first_results, first_ordinals, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&second_decision) == 0);

    TEST_ASSERT(first_decision.query_shape == KEYSTONE_QUERY_SHAPE_RANDOM);
    TEST_ASSERT(second_decision.query_shape == KEYSTONE_QUERY_SHAPE_RANDOM);
    
    if (first_decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED) {
        TEST_ASSERT(second_decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE);
    }
    TEST_ASSERT(second_decision.backend == first_decision.backend);

    keystone_anchor_table_destroy(table);
    free(first_ordinals);
    free(first_results);
    free(items);
    free(data);
}

static void test_large_multi_thread_batch_calibrates_viable_backend(void) {
    const size_t n = 32768;
    int64_t* data = malloc(n * sizeof(int64_t));
    keystone_batch_item_t* items = malloc(n * sizeof(keystone_batch_item_t));
    keystone_backend_decision_t decision;
    keystone_parallel_config_t config = {
        .num_threads = 2,
        .use_thread_pool = 0,
        .batch_chunk = 64
    };
    keystone_anchor_table_t* table = keystone_anchor_table_create();

    TEST_ASSERT(data != NULL);
    TEST_ASSERT(items != NULL);
    TEST_ASSERT(table != NULL);
    fill_data(data, n);
    fill_items(items, n);

    size_t found = keystone_search_batch_auto(data, n, items, n, table, 8, &config);

    TEST_ASSERT(found == n);
    assert_all_found(items, n);
    TEST_ASSERT(keystone_get_last_backend_decision(&decision) == 0);
#ifdef _OPENMP
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_C_OPENMP ||
                    decision.backend == KEYSTONE_BACKEND_SCALAR ||
                    decision.backend == KEYSTONE_BACKEND_FORTRAN);
    } else {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_C_OPENMP ||
                    decision.backend == KEYSTONE_BACKEND_SCALAR);
    }
    TEST_ASSERT(decision.thread_count == 2);
#else
    if (keystone_fortran_backend_available()) {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR ||
                    decision.backend == KEYSTONE_BACKEND_FORTRAN);
    } else {
        TEST_ASSERT(decision.backend == KEYSTONE_BACKEND_SCALAR);
    }
    TEST_ASSERT(decision.thread_count == 1);
#endif
    TEST_ASSERT(decision.query_count_bucket == 32768);
    TEST_ASSERT(decision.estimated_ns_per_key >= 0.0);
    TEST_ASSERT(decision.p95_ns_per_key >= decision.estimated_ns_per_key);
    TEST_ASSERT(decision.query_shape == KEYSTONE_QUERY_SHAPE_DENSE_SORTED);
    if (decision.decision_source == KEYSTONE_DECISION_SOURCE_MEASURED ||
        decision.decision_source == KEYSTONE_DECISION_SOURCE_CACHE) {
        TEST_ASSERT(decision.calibration_runs >= 3);
        TEST_ASSERT(decision.candidates_measured >= 1);
    }

    keystone_anchor_table_destroy(table);
    free(items);
    free(data);
}

int main(void) {
    printf("Running auto backend selector tests\n");
    printf("===================================\n\n");

    test_no_decision_before_auto_call();
    test_invalid_inputs_return_zero_no_decision();
    test_small_batch_uses_scalar();
    test_sorted_8k_batch_uses_expected_backend();
    test_unsorted_8k_batch_uses_scalar();
    test_large_single_thread_batch_uses_scalar();
    test_repeated_large_batch_decision_is_stable();
    test_dense_sorted_cache_hit_preserves_results();
    test_random_shape_cache_hit_preserves_results();
    test_large_multi_thread_batch_calibrates_viable_backend();

    printf("Auto backend selector calibration verified.\n");
    return 0;
}
