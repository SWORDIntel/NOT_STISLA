#include "dsmil_keystone_wrapper.h"
#include "keystone.h"
#include "nst_platform_hints.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * Set error message in context
 */
static void set_error(dsmil_search_context_t *ctx, const char *format, ...) {
    if (!ctx) return;

    va_list args;
    va_start(args, format);
    vsnprintf(ctx->last_error, sizeof(ctx->last_error), format, args);
    va_end(args);
}

/**
 * Clear error message
 */
static void clear_error(dsmil_search_context_t *ctx) {
    if (ctx) {
        ctx->last_error[0] = '\0';
    }
}

/**
 * Check if AVX2 is available via KEYSTONE runtime CPU detection.
 */
static bool check_avx2_support(void) {
    return (keystone_detect_cpu_features() & KEYSTONE_CPU_AVX2) != 0;
}

/* Evaluate local host context for backend dispatch routing decisions. */
__attribute__((unused)) static int _dsmil_evaluate_host_context(_nst_numa_placement_hint_t* hint) {
    if (!hint) return 0;
    _nst_hints_extract_domain_context(hint);
    return _nst_hints_score_environment(hint);
}

/* ============================================================================
 * Core API Implementation
 * ============================================================================ */

dsmil_search_context_t* dsmil_search_create(void) {
    dsmil_search_context_t *ctx = calloc(1, sizeof(dsmil_search_context_t));
    if (!ctx) {
        return NULL;
    }

    // Check AVX2 availability
    ctx->avx2_available = check_avx2_support();

    // Create KEYSTONE anchor table
    ctx->keystone_table = keystone_anchor_table_create();
    if (!ctx->keystone_table) {
        set_error(ctx, "Failed to create KEYSTONE anchor table");
        free(ctx);
        return NULL;
    }

    ctx->initialized = true;
    clear_error(ctx);

#ifdef KEYSTONE_ENABLE_PLATFORM_TUNING
    {
        _nst_numa_placement_hint_t hint = {0};
        _dsmil_evaluate_host_context(&hint);
    }
#endif

    return ctx;
}

void dsmil_search_destroy(dsmil_search_context_t *ctx) {
    if (!ctx) return;

    if (ctx->keystone_table) {
        keystone_anchor_table_destroy(ctx->keystone_table);
        ctx->keystone_table = NULL;
    }

    if (ctx->cached_keys) {
        free(ctx->cached_keys);
        ctx->cached_keys = NULL;
    }

    free(ctx);
}

const char* dsmil_search_get_last_error(const dsmil_search_context_t *ctx) {
    if (!ctx) return "Invalid context";
    return ctx->last_error[0] ? ctx->last_error : "";
}

/* ============================================================================
 * Index Management Implementation
 * ============================================================================ */

dsmil_search_index_t* dsmil_search_index_create_telemetry(const dsmil_telemetry_event_t *events, size_t num_events) {
    if (!events || num_events == 0) return NULL;
    if (num_events > SIZE_MAX / sizeof(int64_t)) return NULL;

    dsmil_search_index_t *index = malloc(sizeof(dsmil_search_index_t));
    if (!index) return NULL;

    index->keys = malloc(num_events * sizeof(int64_t));
    if (!index->keys) {
        free(index);
        return NULL;
    }

    index->num_elements = num_events;
    for (size_t i = 0; i < num_events; i++) {
        index->keys[i] = (int64_t)events[i].timestamp;
        if (i > 0 && index->keys[i] < index->keys[i - 1]) {
            free(index->keys);
            free(index);
            return NULL; /* KEYSTONE requires sorted keys */
        }
    }

    return index;
}

dsmil_search_index_t* dsmil_search_index_create_security(const dsmil_security_event_t *events, size_t num_events) {
    if (!events || num_events == 0) return NULL;
    if (num_events > SIZE_MAX / sizeof(int64_t)) return NULL;

    dsmil_search_index_t *index = malloc(sizeof(dsmil_search_index_t));
    if (!index) return NULL;

    index->keys = malloc(num_events * sizeof(int64_t));
    if (!index->keys) {
        free(index);
        return NULL;
    }

    index->num_elements = num_events;
    for (size_t i = 0; i < num_events; i++) {
        index->keys[i] = (int64_t)events[i].event_id;
        if (i > 0 && index->keys[i] < index->keys[i - 1]) {
            free(index->keys);
            free(index);
            return NULL; /* KEYSTONE requires sorted keys */
        }
    }

    return index;
}

dsmil_search_index_t* dsmil_search_index_create_logs(const dsmil_log_entry_t *logs, size_t num_logs) {
    if (!logs || num_logs == 0) return NULL;
    if (num_logs > SIZE_MAX / sizeof(int64_t)) return NULL;

    dsmil_search_index_t *index = malloc(sizeof(dsmil_search_index_t));
    if (!index) return NULL;

    index->keys = malloc(num_logs * sizeof(int64_t));
    if (!index->keys) {
        free(index);
        return NULL;
    }

    index->num_elements = num_logs;
    for (size_t i = 0; i < num_logs; i++) {
        index->keys[i] = (int64_t)logs[i].log_id;
        if (i > 0 && index->keys[i] < index->keys[i - 1]) {
            free(index->keys);
            free(index);
            return NULL; /* KEYSTONE requires sorted keys */
        }
    }

    return index;
}

void dsmil_search_index_destroy(dsmil_search_index_t *index) {
    if (!index) return;
    if (index->keys) free(index->keys);
    free(index);
}

/* ============================================================================
 * Telemetry Search Implementation
 * ============================================================================ */

int dsmil_search_telemetry_events(
    dsmil_search_context_t *ctx,
    const dsmil_telemetry_event_t *events,
    size_t num_events,
    dsmil_timestamp_t target_time,
    dsmil_telemetry_result_t *result
) {
    if (!ctx || !events || !result || num_events == 0) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Optimize key extraction: check if we can reuse cached keys
    int64_t *timestamps;
    int64_t first_key = (int64_t)events[0].timestamp;
    int64_t last_key = (int64_t)events[num_events - 1].timestamp;
    if (ctx->last_data_ptr == (const void *)events && ctx->cached_count == num_events &&
        ctx->cached_first_key == first_key && ctx->cached_last_key == last_key && ctx->cached_keys) {
        timestamps = ctx->cached_keys;
        ctx->cache_hits++;
    } else {
        // Need to re-extract or re-allocate
        if (num_events > ctx->cached_capacity) {
            if (num_events > SIZE_MAX / sizeof(int64_t)) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            int64_t *new_keys = realloc(ctx->cached_keys, num_events * sizeof(int64_t));
            if (!new_keys) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            ctx->cached_keys = new_keys;
            ctx->cached_capacity = num_events;
        }

        for (size_t i = 0; i < num_events; i++) {
            ctx->cached_keys[i] = (int64_t)events[i].timestamp;
        }
        ctx->cached_count = num_events;
        ctx->last_data_ptr = (const void *)events;
        ctx->cached_first_key = first_key;
        ctx->cached_last_key = last_key;
        timestamps = ctx->cached_keys;
    }

    // Perform search using telemetry-tuned tolerance
    keystone_result_t search_result = keystone_search_telemetry(
        timestamps, num_events, (int64_t)target_time, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->exact_match_time = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->event = &events[found_index];
    result->index = found_index;
    result->exact_match_time = events[found_index].timestamp;
    result->is_exact_match = (events[found_index].timestamp == target_time);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_telemetry_events_indexed(
    dsmil_search_context_t *ctx,
    const dsmil_search_index_t *index,
    const dsmil_telemetry_event_t *events,
    dsmil_timestamp_t target_time,
    dsmil_telemetry_result_t *result
) {
    if (!ctx || !index || !events || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Perform search using pre-extracted keys
    keystone_result_t search_result = keystone_search_telemetry(
        index->keys, index->num_elements, (int64_t)target_time, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;
    ctx->cache_hits++; // Consider indexed search as a cache hit in terms of avoided extraction

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->exact_match_time = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->event = &events[found_index];
    result->index = found_index;
    result->exact_match_time = events[found_index].timestamp;
    result->is_exact_match = (events[found_index].timestamp == target_time);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_telemetry_by_device_time_range(
    dsmil_search_context_t *ctx,
    const dsmil_telemetry_event_t *events,
    size_t num_events,
    uint32_t device_id,
    dsmil_timestamp_t start_time,
    dsmil_timestamp_t end_time,
    dsmil_telemetry_result_t *results,
    size_t max_results,
    size_t *num_found
) {
    if (!ctx || !events || !results || !num_found) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    size_t found = 0;

    // Simple linear scan for device/time range filtering
    // In a production implementation, this could be optimized further
    for (size_t i = 0; i < num_events && found < max_results; i++) {
        const dsmil_telemetry_event_t *event = &events[i];

        if (event->device_id == device_id &&
            event->timestamp >= start_time &&
            event->timestamp <= end_time) {

            results[found].event = event;
            results[found].index = i;
            results[found].exact_match_time = event->timestamp;
            results[found].is_exact_match = true; // Range match
            found++;
        }
    }

    *num_found = found;
    return DSMIL_SEARCH_SUCCESS;
}

/* ============================================================================
 * Security Search Implementation
 * ============================================================================ */

int dsmil_search_security_events(
    dsmil_search_context_t *ctx,
    const dsmil_security_event_t *events,
    size_t num_events,
    dsmil_security_id_t target_id,
    dsmil_security_result_t *result
) {
    if (!ctx || !events || !result || num_events == 0) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Optimize key extraction
    int64_t *event_ids;
    int64_t first_key = (int64_t)events[0].event_id;
    int64_t last_key = (int64_t)events[num_events - 1].event_id;
    if (ctx->last_data_ptr == (const void *)events && ctx->cached_count == num_events &&
        ctx->cached_first_key == first_key && ctx->cached_last_key == last_key && ctx->cached_keys) {
        event_ids = ctx->cached_keys;
        ctx->cache_hits++;
    } else {
        if (num_events > ctx->cached_capacity) {
            if (num_events > SIZE_MAX / sizeof(int64_t)) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            int64_t *new_keys = realloc(ctx->cached_keys, num_events * sizeof(int64_t));
            if (!new_keys) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            ctx->cached_keys = new_keys;
            ctx->cached_capacity = num_events;
        }

        for (size_t i = 0; i < num_events; i++) {
            ctx->cached_keys[i] = (int64_t)events[i].event_id;
        }
        ctx->cached_count = num_events;
        ctx->last_data_ptr = (const void *)events;
        ctx->cached_first_key = first_key;
        ctx->cached_last_key = last_key;
        event_ids = ctx->cached_keys;
    }

    // Perform search using ID-tuned tolerance
    keystone_result_t search_result = keystone_search_ids(
        event_ids, num_events, (int64_t)target_id, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->event = &events[found_index];
    result->index = found_index;
    result->matched_id = events[found_index].event_id;
    result->is_exact_match = (events[found_index].event_id == target_id);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_security_events_indexed(
    dsmil_search_context_t *ctx,
    const dsmil_search_index_t *index,
    const dsmil_security_event_t *events,
    dsmil_security_id_t target_id,
    dsmil_security_result_t *result
) {
    if (!ctx || !index || !events || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Perform search using pre-extracted keys
    keystone_result_t search_result = keystone_search_ids(
        index->keys, index->num_elements, (int64_t)target_id, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;
    ctx->cache_hits++;

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->event = &events[found_index];
    result->index = found_index;
    result->matched_id = events[found_index].event_id;
    result->is_exact_match = (events[found_index].event_id == target_id);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_security_by_severity_time_range(
    dsmil_search_context_t *ctx,
    const dsmil_security_event_t *events,
    size_t num_events,
    uint32_t min_severity,
    uint32_t max_severity,
    dsmil_timestamp_t start_time,
    dsmil_timestamp_t end_time,
    dsmil_security_result_t *results,
    size_t max_results,
    size_t *num_found
) {
    if (!ctx || !events || !results || !num_found) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    size_t found = 0;

    // Linear scan for severity/time range filtering
    for (size_t i = 0; i < num_events && found < max_results; i++) {
        const dsmil_security_event_t *event = &events[i];

        if (event->severity >= min_severity &&
            event->severity <= max_severity &&
            event->timestamp >= start_time &&
            event->timestamp <= end_time) {

            results[found].event = event;
            results[found].index = i;
            results[found].matched_id = event->event_id;
            results[found].is_exact_match = true; // Range match
            found++;
        }
    }

    *num_found = found;
    return DSMIL_SEARCH_SUCCESS;
}

/* ============================================================================
 * Log Search Implementation
 * ============================================================================ */

int dsmil_search_log_entries(
    dsmil_search_context_t *ctx,
    const dsmil_log_entry_t *logs,
    size_t num_logs,
    dsmil_log_id_t target_id,
    dsmil_log_result_t *result
) {
    if (!ctx || !logs || !result || num_logs == 0) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Optimize key extraction
    int64_t *log_ids;
    int64_t first_key = (int64_t)logs[0].log_id;
    int64_t last_key = (int64_t)logs[num_logs - 1].log_id;
    if (ctx->last_data_ptr == (const void *)logs && ctx->cached_count == num_logs &&
        ctx->cached_first_key == first_key && ctx->cached_last_key == last_key && ctx->cached_keys) {
        log_ids = ctx->cached_keys;
        ctx->cache_hits++;
    } else {
        if (num_logs > ctx->cached_capacity) {
            if (num_logs > SIZE_MAX / sizeof(int64_t)) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            int64_t *new_keys = realloc(ctx->cached_keys, num_logs * sizeof(int64_t));
            if (!new_keys) {
                set_error(ctx, "Memory allocation failed");
                return DSMIL_SEARCH_ERROR_MEMORY;
            }
            ctx->cached_keys = new_keys;
            ctx->cached_capacity = num_logs;
        }

        for (size_t i = 0; i < num_logs; i++) {
            ctx->cached_keys[i] = (int64_t)logs[i].log_id;
        }
        ctx->cached_count = num_logs;
        ctx->last_data_ptr = (const void *)logs;
        ctx->cached_first_key = first_key;
        ctx->cached_last_key = last_key;
        log_ids = ctx->cached_keys;
    }

    // Perform search using ID-tuned tolerance
    keystone_result_t search_result = keystone_search_ids(
        log_ids, num_logs, (int64_t)target_id, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->entry = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->entry = &logs[found_index];
    result->index = found_index;
    result->matched_id = logs[found_index].log_id;
    result->is_exact_match = (logs[found_index].log_id == target_id);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_log_entries_indexed(
    dsmil_search_context_t *ctx,
    const dsmil_search_index_t *index,
    const dsmil_log_entry_t *logs,
    dsmil_log_id_t target_id,
    dsmil_log_result_t *result
) {
    if (!ctx || !index || !logs || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    // Perform search using pre-extracted keys
    keystone_result_t search_result = keystone_search_ids(
        index->keys, index->num_elements, (int64_t)target_id, ctx->keystone_table
    );

    // Update statistics
    ctx->search_operations++;
    ctx->cache_hits++;

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->entry = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    // Fill result structure
    size_t found_index = (size_t)search_result;
    result->entry = &logs[found_index];
    result->index = found_index;
    result->matched_id = logs[found_index].log_id;
    result->is_exact_match = (logs[found_index].log_id == target_id);

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_logs_by_facility_time_range(
    dsmil_search_context_t *ctx,
    const dsmil_log_entry_t *logs,
    size_t num_logs,
    uint32_t facility,
    dsmil_timestamp_t start_time,
    dsmil_timestamp_t end_time,
    dsmil_log_result_t *results,
    size_t max_results,
    size_t *num_found
) {
    if (!ctx || !logs || !results || !num_found) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    size_t found = 0;

    // Linear scan for facility/time range filtering
    for (size_t i = 0; i < num_logs && found < max_results; i++) {
        const dsmil_log_entry_t *log_entry = &logs[i];

        if (log_entry->facility == facility &&
            log_entry->timestamp >= start_time &&
            log_entry->timestamp <= end_time) {

            results[found].entry = log_entry;
            results[found].index = i;
            results[found].matched_id = log_entry->log_id;
            results[found].is_exact_match = true; // Range match
            found++;
        }
    }

    *num_found = found;
    return DSMIL_SEARCH_SUCCESS;
}

/* ============================================================================
 * tar.zst Streaming Search Implementation
 * ============================================================================ */

#ifdef KEYSTONE_ENABLE_TAR_ZST

int dsmil_search_telemetry_events_from_tar_zst(
    dsmil_search_context_t *ctx,
    const char *archive_path,
    const char *member_name,
    dsmil_timestamp_t target_time,
    dsmil_telemetry_result_t *result
) {
    if (!ctx || !archive_path || !member_name || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_AUTO,
        .chunk_size = 256 * 1024,
        .arena_slab_size = 1u << 20,
        .zstd_workers = 0,
        .enable_pipeline = 0,
        .skip_header = 0
    };

    keystone_tar_zst_t *tz = keystone_tar_zst_open(archive_path, &opts);
    if (!tz) {
        set_error(ctx, "Failed to open archive: %s", archive_path);
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_TELEMETRY);

    keystone_result_t search_result = keystone_tar_zst_search_member(
        tz, member_name, (int64_t)target_time, ctx->keystone_table, &config
    );

    keystone_tar_zst_close(tz);

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->exact_match_time = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    /* For tar.zst we don't have the original event array; return index only */
    result->event = NULL;
    result->index = (size_t)search_result;
    result->exact_match_time = target_time;
    result->is_exact_match = true;

    ctx->search_operations++;
    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_security_events_from_tar_zst(
    dsmil_search_context_t *ctx,
    const char *archive_path,
    const char *member_name,
    dsmil_security_id_t target_id,
    dsmil_security_result_t *result
) {
    if (!ctx || !archive_path || !member_name || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_AUTO,
        .chunk_size = 256 * 1024,
        .arena_slab_size = 1u << 20,
        .zstd_workers = 0,
        .enable_pipeline = 0,
        .skip_header = 0
    };

    keystone_tar_zst_t *tz = keystone_tar_zst_open(archive_path, &opts);
    if (!tz) {
        set_error(ctx, "Failed to open archive: %s", archive_path);
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t search_result = keystone_tar_zst_search_member(
        tz, member_name, (int64_t)target_id, ctx->keystone_table, &config
    );

    keystone_tar_zst_close(tz);

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->event = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    result->event = NULL;
    result->index = (size_t)search_result;
    result->matched_id = target_id;
    result->is_exact_match = true;

    ctx->search_operations++;
    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_log_entries_from_tar_zst(
    dsmil_search_context_t *ctx,
    const char *archive_path,
    const char *member_name,
    dsmil_log_id_t target_id,
    dsmil_log_result_t *result
) {
    if (!ctx || !archive_path || !member_name || !result) {
        if (ctx) set_error(ctx, "Invalid parameters");
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (!ctx->initialized) {
        set_error(ctx, "Search context not initialized");
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    clear_error(ctx);

    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_AUTO,
        .chunk_size = 256 * 1024,
        .arena_slab_size = 1u << 20,
        .zstd_workers = 0,
        .enable_pipeline = 0,
        .skip_header = 0
    };

    keystone_tar_zst_t *tz = keystone_tar_zst_open(archive_path, &opts);
    if (!tz) {
        set_error(ctx, "Failed to open archive: %s", archive_path);
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    keystone_config_t config;
    keystone_config_init(&config, KEYSTONE_WORKLOAD_IDS);

    keystone_result_t search_result = keystone_tar_zst_search_member(
        tz, member_name, (int64_t)target_id, ctx->keystone_table, &config
    );

    keystone_tar_zst_close(tz);

    if (search_result == KEYSTONE_NOT_FOUND) {
        result->entry = NULL;
        result->index = (size_t)-1;
        result->matched_id = 0;
        result->is_exact_match = false;
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    result->entry = NULL;
    result->index = (size_t)search_result;
    result->matched_id = target_id;
    result->is_exact_match = true;

    ctx->search_operations++;
    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_batch_tar_zst(
    dsmil_search_context_t *ctx,
    const char *archive_path,
    const char **member_names,
    size_t num_members,
    int64_t *keys,
    size_t num_keys,
    dsmil_telemetry_result_t *results
) {
    (void)ctx; /* stateless: do not mutate context */

    if (!archive_path || !member_names || num_members == 0 ||
        !keys || num_keys == 0 || !results) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    keystone_tar_zst_t *tz = keystone_tar_zst_open(archive_path, NULL);
    if (!tz) {
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    /* Track which keys have been found */
    int *found = calloc(num_keys, sizeof(int));
    if (!found) {
        keystone_tar_zst_close(tz);
        return DSMIL_SEARCH_ERROR_MEMORY;
    }

    size_t found_total = 0;

    for (size_t m = 0; m < num_members && found_total < num_keys; m++) {
        int64_t *member_keys = NULL;
        size_t count = 0;

        /* Extract member directly using rewind-capable API */
        if (keystone_tar_zst_extract_member(tz, member_names[m],
                                                &member_keys, &count) != 0) {
            continue;
        }
        if (count == 0 || !member_keys) continue;

        /* Binary search each unfound key in member_keys */
        for (size_t k = 0; k < num_keys && found_total < num_keys; k++) {
            if (found[k]) continue;

            size_t lo = 0, hi = count;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (member_keys[mid] < keys[k]) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if (lo < count && member_keys[lo] == keys[k]) {
                results[k].event = NULL;
                results[k].index = lo;
                results[k].exact_match_time = keys[k];
                results[k].is_exact_match = true;
                found[k] = 1;
                found_total++;
            }
        }
    }

    /* Mark unfound keys */
    for (size_t k = 0; k < num_keys; k++) {
        if (!found[k]) {
            results[k].event = NULL;
            results[k].index = (size_t)-1;
            results[k].exact_match_time = 0;
            results[k].is_exact_match = false;
        }
    }

    free(found);
    keystone_tar_zst_close(tz);

    return (found_total > 0) ? DSMIL_SEARCH_SUCCESS : DSMIL_SEARCH_ERROR_NOT_FOUND;
}

#endif /* KEYSTONE_ENABLE_TAR_ZST */

/* ============================================================================
 * Utility Functions Implementation
 * ============================================================================ */

int dsmil_search_get_stats(
    const dsmil_search_context_t *ctx,
    uint32_t *total_searches,
    double *cache_hit_rate,
    uint32_t *memory_usage,
    double *avg_search_time_ns
) {
    if (!ctx || !total_searches || !cache_hit_rate || !memory_usage || !avg_search_time_ns) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    /* KEYSTONE counters are uint64_t; API returns uint32_t. Saturate to avoid silent wrap. */
    *total_searches = ctx->search_operations > UINT32_MAX ? UINT32_MAX : (uint32_t)ctx->search_operations;
    *cache_hit_rate = ctx->search_operations > 0 ?
        (double)ctx->cache_hits / ctx->search_operations : 0.0;
    *memory_usage = ctx->memory_usage > UINT32_MAX ? UINT32_MAX : (uint32_t)ctx->memory_usage;
    *avg_search_time_ns = 0.0; // Would need timing instrumentation to calculate

    return DSMIL_SEARCH_SUCCESS;
}

int dsmil_search_reset_stats(dsmil_search_context_t *ctx) {
    if (!ctx) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    ctx->search_operations = 0;
    ctx->cache_hits = 0;
    ctx->memory_usage = 0;
    ctx->last_data_ptr = NULL;
    ctx->cached_first_key = 0;
    ctx->cached_last_key = 0;
    ctx->cached_count = 0;

    return DSMIL_SEARCH_SUCCESS;
}

bool dsmil_search_avx2_enabled(const dsmil_search_context_t *ctx) {
    if (!ctx) return false;
    return ctx->avx2_available && ctx->initialized;
}

const char* dsmil_search_get_version(void) {
    return keystone_version();
}
