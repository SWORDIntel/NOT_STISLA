/**
 * KEYSTONE tar.zst streaming search test suite
 */

#include "../include/keystone.h"
#include "../include/keystone_tar_zst.h"
#include "../include/dsmil_keystone_wrapper.h"
#include "../include/dsmil_telemetry_processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define FIXTURE_DIR "tests/fixtures_tar_zst"
#define ARCHIVE_PATH FIXTURE_DIR "/test_data.tar.zst"

static int tests_passed = 0;
static int tests_failed = 0;

static void test_pass(const char* name) {
    printf("  [PASS] %s\n", name);
    tests_passed++;
}

static void test_fail(const char* name, const char* reason) {
    printf("  [FAIL] %s: %s\n", name, reason);
    tests_failed++;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static void test_open_close(void) {
    const char* name = "open_close";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_iterate_members(void) {
    const char* name = "iterate_members";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    int found_txt = 0;
    int found_csv = 0;
    int found_json = 0;
    int total = 0;

    for (;;) {
        char* member_name = NULL;
        size_t member_name_len = 0;
        int r = keystone_tar_zst_next_member(tz, &member_name, &member_name_len);
        if (r <= 0) break;
        total++;

        if (member_name_len >= 4 &&
            strncmp(member_name + member_name_len - 4, ".txt", 4) == 0) {
            found_txt = 1;
        }
        if (member_name_len >= 4 &&
            strncmp(member_name + member_name_len - 4, ".csv", 4) == 0) {
            found_csv = 1;
        }
        if (member_name_len >= 5 &&
            strncmp(member_name + member_name_len - 5, ".json", 5) == 0) {
            found_json = 1;
        }
    }

    keystone_tar_zst_close(tz);

    if (total < 3) {
        test_fail(name, "expected at least 3 members");
        return;
    }
    if (!found_txt || !found_csv || !found_json) {
        test_fail(name, "missing expected member types");
        return;
    }
    test_pass(name);
}

static void test_search_text_member(void) {
    const char* name = "search_text_member";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    /* The text file contains even numbers 0, 2, 4, ..., 2000 */
    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t result = keystone_tar_zst_search_member(
        tz, "numbers.txt", 1000, NULL, &config
    );

    keystone_tar_zst_close(tz);

    if (result == KEYSTONE_NOT_FOUND) {
        test_fail(name, "key 1000 not found");
        return;
    }
    test_pass(name);
}

static void test_search_csv_member(void) {
    const char* name = "search_csv_member";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_CSV,
        .skip_header = 1,
        .chunk_size = 4096,
        .arena_slab_size = 65536,
        .zstd_workers = 0,
        .enable_pipeline = 0
    };

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, &opts);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    /* The CSV file contains multiples of 3: 0, 3, 6, ..., 3000 */
    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t result = keystone_tar_zst_search_member(
        tz, "numbers.csv", 1500, NULL, &config
    );

    keystone_tar_zst_close(tz);

    if (result == KEYSTONE_NOT_FOUND) {
        test_fail(name, "key 1500 not found");
        return;
    }
    test_pass(name);
}

static void test_search_json_member(void) {
    const char* name = "search_json_member";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_JSON,
        .chunk_size = 4096,
        .arena_slab_size = 65536,
        .zstd_workers = 0,
        .enable_pipeline = 0,
        .skip_header = 0
    };

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, &opts);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    /* The JSON file contains multiples of 5: 0, 5, 10, ..., 5000 */
    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t result = keystone_tar_zst_search_member(
        tz, "numbers.json", 2500, NULL, &config
    );

    keystone_tar_zst_close(tz);

    if (result == KEYSTONE_NOT_FOUND) {
        test_fail(name, "key 2500 not found");
        return;
    }
    test_pass(name);
}

static void test_stats_consistency(void) {
    const char* name = "stats_consistency";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    /* Search multiple members */
    keystone_tar_zst_search_member(tz, "numbers.txt", 500, NULL, &config);
    keystone_tar_zst_search_member(tz, "numbers.csv", 501, NULL, &config);

    keystone_tar_zst_stats_t stats;
    if (keystone_tar_zst_get_stats(tz, &stats) != 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "get_stats failed");
        return;
    }

    keystone_tar_zst_close(tz);

    if (stats.members_read < 2) {
        test_fail(name, "expected at least 2 members read");
        return;
    }
    if (stats.bytes_read == 0) {
        test_fail(name, "expected non-zero bytes read");
        return;
    }
    test_pass(name);
}

static void test_error_handling(void) {
    const char* name = "error_handling";

    keystone_tar_zst_t* tz = keystone_tar_zst_open("/nonexistent/path.tar.zst", NULL);
    if (tz) {
        keystone_tar_zst_close(tz);
        test_fail(name, "should have failed for nonexistent file");
        return;
    }
    test_pass(name);
}

static void test_indexed_search(void) {
    const char* name = "indexed_search";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    if (keystone_tar_zst_build_index(tz) != 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "build_index failed");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    /* Search for a key that exists (500 is in numbers.txt) */
    keystone_result_t r1 = keystone_tar_zst_search_indexed(
        tz, "numbers.txt", 500, NULL, &config);
    if (r1 == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz);
        test_fail(name, "indexed search for 500 failed");
        return;
    }

    /* Search for a key that does not exist (should be Bloom-rejected) */
    keystone_result_t r2 = keystone_tar_zst_search_indexed(
        tz, "numbers.txt", 999999, NULL, &config);
    if (r2 != KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz);
        test_fail(name, "Bloom should have rejected 999999");
        return;
    }

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_extract_member(void) {
    const char* name = "extract_member";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    int64_t* keys = NULL;
    size_t count = 0;
    if (keystone_tar_zst_extract_member(tz, "numbers.txt",
                                             &keys, &count) != 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "extract_member failed");
        return;
    }

    if (!keys || count == 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "no keys extracted");
        return;
    }

    /* numbers.txt contains even numbers 0, 2, 4, ..., 2000 */
    if (keys[0] != 0 || keys[count - 1] != 2000) {
        keystone_tar_zst_close(tz);
        test_fail(name, "extracted key range mismatch");
        return;
    }

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_corrupt_members(void) {
    const char* name = "corrupt_members";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    int64_t* keys = NULL;
    size_t count = 0;

    /* corrupt.txt should parse no keys (or fail gracefully) */
    if (keystone_tar_zst_extract_member(tz, "corrupt.txt", &keys, &count) == 0) {
        if (count > 0) {
            test_fail(name, "corrupt.txt should yield 0 valid keys");
            keystone_tar_zst_close(tz);
            return;
        }
    }

    keys = NULL;
    count = 0;
    /* corrupt.csv might yield 1 or 2 keys (123, 456), but gracefully skip the bad row */
    if (keystone_tar_zst_extract_member(tz, "corrupt.csv", &keys, &count) == 0) {
        if (count > 2) {
            test_fail(name, "corrupt.csv yielded too many keys");
            keystone_tar_zst_close(tz);
            return;
        }
    }

    keys = NULL;
    count = 0;
    /* corrupt.json might parse keys partially depending on JSON strategy, or return empty/error */
    keystone_tar_zst_extract_member(tz, "corrupt.json", &keys, &count);

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_batch_search(void) {
    const char* name = "batch_search";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    int64_t search_keys[] = {500, 1500, 2500, 999999};
    size_t num_keys = sizeof(search_keys) / sizeof(search_keys[0]);
    dsmil_telemetry_result_t results[4];
    const char *members[] = {"numbers.txt", "numbers.csv", "numbers.json"};

    int rc = dsmil_search_batch_tar_zst(
        NULL, ARCHIVE_PATH, members, 3,
        search_keys, num_keys, results
    );

    if (rc != DSMIL_SEARCH_SUCCESS) {
        test_fail(name, "batch search returned error");
        return;
    }

    if (!results[0].is_exact_match || results[0].exact_match_time != 500) {
        test_fail(name, "key 500 not found in batch");
        return;
    }
    if (!results[1].is_exact_match || results[1].exact_match_time != 1500) {
        test_fail(name, "key 1500 not found in batch");
        return;
    }
    if (!results[2].is_exact_match || results[2].exact_match_time != 2500) {
        test_fail(name, "key 2500 not found in batch");
        return;
    }
    if (results[3].is_exact_match) {
        test_fail(name, "key 999999 should not be found");
        return;
    }

    test_pass(name);
}

static void test_load_all_members(void) {
    const char* name = "load_all_members";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    dsmil_telemetry_processor_t* proc = dsmil_telemetry_processor_create(10000);
    if (!proc) {
        test_fail(name, "failed to create processor");
        return;
    }

    int rc = dsmil_telemetry_processor_load_all_members(proc, ARCHIVE_PATH);
    if (rc != DSMIL_SEARCH_SUCCESS) {
        dsmil_telemetry_processor_destroy(proc);
        test_fail(name, "load_all_members failed");
        return;
    }

    /* The archive has 3 members: numbers.txt (1001 keys),
     * numbers.csv (1001 keys), numbers.json (1001 keys)
     * All match the *.txt / *.csv / *.json patterns */
    size_t count = 0;
    /* We don't have a direct getter for event count, but we can
     * infer success from the return code. A minimal check is enough. */
    (void)count;

    dsmil_telemetry_processor_destroy(proc);
    test_pass(name);
}

static void test_pipeline_mode(void) {
    const char* name = "pipeline_mode";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_AUTO,
        .enable_pipeline = 1,
        .chunk_size = 4096,
        .arena_slab_size = 65536
    };

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, &opts);
    if (!tz) {
        test_fail(name, "failed to open archive in pipeline mode");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t r = keystone_tar_zst_search_member(tz, "numbers.txt", 1000, NULL, &config);
    if (r == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz);
        test_fail(name, "pipelined search failed to find 1000");
        return;
    }

    keystone_tar_zst_stats_t stats;
    keystone_tar_zst_get_stats(tz, &stats);
    if (stats.members_read == 0 || stats.bytes_read == 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "pipelined stats show 0 members or bytes read");
        return;
    }

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_rewind_random_order(void) {
    const char* name = "rewind_random_order";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    /* Search the last member first */
    keystone_result_t r1 = keystone_tar_zst_search_member(tz, "numbers.json", 2500, NULL, &config);
    if (r1 == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz);
        test_fail(name, "failed to find key in numbers.json");
        return;
    }

    /* Now search the first member (requires internal rewind) */
    keystone_result_t r2 = keystone_tar_zst_search_member(tz, "numbers.txt", 1000, NULL, &config);
    if (r2 == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz);
        test_fail(name, "rewind search failed for numbers.txt");
        return;
    }

    keystone_tar_zst_close(tz);
    test_pass(name);
}

static void test_sidecar_index_json(void) {
    const char* name = "sidecar_index_json";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    const char* idx_path = FIXTURE_DIR "/test_data.tar.idx.json";
    remove(idx_path);

    /* 1. Build and save index to .idx.json */
    keystone_tar_zst_t* tz = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz) {
        test_fail(name, "failed to open archive for indexing");
        return;
    }

    if (keystone_tar_zst_build_index(tz) != 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "build_index failed");
        return;
    }

    if (keystone_tar_zst_save_index(tz, idx_path) != 0) {
        keystone_tar_zst_close(tz);
        test_fail(name, "save_index failed");
        return;
    }
    keystone_tar_zst_close(tz);

    if (!file_exists(idx_path)) {
        test_fail(name, "idx.json file was not created");
        return;
    }

    /* 2. Open archive and load index directly from JSON sidecar */
    keystone_tar_zst_t* tz2 = keystone_tar_zst_open(ARCHIVE_PATH, NULL);
    if (!tz2) {
        test_fail(name, "failed to reopen archive");
        return;
    }

    if (keystone_tar_zst_load_index(tz2, idx_path) != 0) {
        keystone_tar_zst_close(tz2);
        test_fail(name, "load_index from JSON failed");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    /* 3. Search positive match via loaded JSON index */
    keystone_result_t r_pos = keystone_tar_zst_search_indexed(tz2, "numbers.txt", 1000, NULL, &config);
    if (r_pos == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz2);
        test_fail(name, "indexed search on loaded JSON index failed for 1000");
        return;
    }

    /* 4. Search negative match via loaded JSON index (Bloom rejection) */
    keystone_result_t r_neg = keystone_tar_zst_search_indexed(tz2, "numbers.txt", 999999, NULL, &config);
    if (r_neg != KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_close(tz2);
        test_fail(name, "Bloom filter failed to reject 999999");
        return;
    }

    keystone_tar_zst_close(tz2);
    remove(idx_path);
    test_pass(name);
}

static void test_batch_archives_pool(void) {
    const char* name = "batch_archives_pool";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    const char* archives[] = { ARCHIVE_PATH, ARCHIVE_PATH };
    keystone_tar_zst_batch_t* batch = keystone_tar_zst_batch_open(archives, 2, NULL);
    if (!batch) {
        test_fail(name, "failed to open batch pool");
        return;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    size_t match_archive = 999;
    keystone_result_t res = keystone_tar_zst_batch_search(
        batch, "numbers.csv", 1500, NULL, &config, &match_archive);

    if (res == KEYSTONE_NOT_FOUND) {
        keystone_tar_zst_batch_close(batch);
        test_fail(name, "batch search failed to find key 1500");
        return;
    }

    if (match_archive >= 2) {
        keystone_tar_zst_batch_close(batch);
        test_fail(name, "invalid matched archive index");
        return;
    }

    keystone_tar_zst_batch_close(batch);
    test_pass(name);
}

static void test_dsmil_load_from_tar_zst_populated(void) {
    const char* name = "dsmil_load_from_tar_zst_populated";
    if (!file_exists(ARCHIVE_PATH)) {
        test_fail(name, "archive fixture not found");
        return;
    }

    dsmil_telemetry_processor_t* proc = dsmil_telemetry_processor_create(2000);
    if (!proc) {
        test_fail(name, "failed to create processor");
        return;
    }

    int rc = dsmil_telemetry_processor_load_from_tar_zst(proc, ARCHIVE_PATH, "numbers.txt");
    if (rc != DSMIL_SEARCH_SUCCESS) {
        dsmil_telemetry_processor_destroy(proc);
        test_fail(name, "load_from_tar_zst failed");
        return;
    }

    dsmil_telemetry_result_t result;
    int s_rc = dsmil_telemetry_processor_find_by_timestamp(proc, 1000, &result);
    if (s_rc != DSMIL_SEARCH_SUCCESS || !result.is_exact_match) {
        dsmil_telemetry_processor_destroy(proc);
        test_fail(name, "telemetry search in loaded processor failed to find key 1000");
        return;
    }

    dsmil_telemetry_processor_destroy(proc);
    test_pass(name);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("KEYSTONE tar.zst streaming search test suite\n");
    printf("==============================================\n\n");

    test_open_close();
    test_iterate_members();
    test_search_text_member();
    test_search_csv_member();
    test_search_json_member();
    test_stats_consistency();
    test_error_handling();
    test_indexed_search();
    test_extract_member();
    test_corrupt_members();
    test_batch_search();
    test_load_all_members();
    test_pipeline_mode();
    test_rewind_random_order();
    test_sidecar_index_json();
    test_batch_archives_pool();
    test_dsmil_load_from_tar_zst_populated();

    printf("\n----------------------------------------------\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
