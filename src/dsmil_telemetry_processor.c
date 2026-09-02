/**
 * DSMIL Telemetry Processor with KEYSTONE Acceleration
 * NATO RESTRICTED
 *
 * High-performance telemetry data processing using KEYSTONE for ultra-fast
 * timestamp-based event lookups and time-series analysis.
 */

#include "dsmil_keystone_wrapper.h"
#include "keystone.h"
#include "nst_platform_hints.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ============================================================================
 * DSMIL Telemetry Processing Context
 * ============================================================================ */

typedef struct dsmil_telemetry_processor {
    dsmil_search_context_t *search_ctx;
    size_t max_events;
    size_t event_count;
    dsmil_telemetry_event_t *events;
    int64_t *timestamps;  // For KEYSTONE search
    bool timestamps_sorted;
    bool initialized;
} dsmil_telemetry_processor_t;

static size_t telemetry_lower_bound_timestamp(
    const int64_t *timestamps,
    size_t count,
    dsmil_timestamp_t target
) {
    size_t low = 0;
    size_t high = count;

    while (low < high) {
        size_t mid = low + (high - low) / 2;

        if ((dsmil_timestamp_t)timestamps[mid] < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}

static size_t telemetry_upper_bound_timestamp(
    const int64_t *timestamps,
    size_t count,
    dsmil_timestamp_t target
) {
    size_t low = 0;
    size_t high = count;

    while (low < high) {
        size_t mid = low + (high - low) / 2;

        if ((dsmil_timestamp_t)timestamps[mid] <= target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}

static void telemetry_fill_range_result(
    dsmil_telemetry_result_t *result,
    const dsmil_telemetry_event_t *event,
    size_t index
) {
    result->event = event;
    result->index = index;
    result->exact_match_time = event->timestamp;
    result->is_exact_match = false; /* Caller may override for exact boundary hits */
}

static void telemetry_fill_not_found(dsmil_telemetry_result_t *result) {
    result->event = NULL;
    result->index = (size_t)-1;
    result->exact_match_time = 0;
    result->is_exact_match = false;
}

/* ============================================================================
 * Telemetry Processor API
 * ============================================================================ */

/**
 * Create a new telemetry processor with KEYSTONE acceleration
 */
dsmil_telemetry_processor_t* dsmil_telemetry_processor_create(size_t max_events) {
    dsmil_telemetry_processor_t *processor = calloc(1, sizeof(dsmil_telemetry_processor_t));
    if (!processor) {
        return NULL;
    }

    processor->search_ctx = dsmil_search_create();
    if (!processor->search_ctx) {
        free(processor);
        return NULL;
    }

    processor->max_events = max_events;
    processor->events = calloc(max_events, sizeof(dsmil_telemetry_event_t));
    processor->timestamps = calloc(max_events, sizeof(int64_t));
    processor->timestamps_sorted = true;

    if (!processor->events || !processor->timestamps) {
        dsmil_search_destroy(processor->search_ctx);
        free(processor->events);
        free(processor->timestamps);
        free(processor);
        return NULL;
    }

    processor->initialized = true;
    return processor;
}

/**
 * Destroy telemetry processor
 */
void dsmil_telemetry_processor_destroy(dsmil_telemetry_processor_t *processor) {
    if (!processor) return;

    if (processor->search_ctx) {
        dsmil_search_destroy(processor->search_ctx);
    }

    free(processor->events);
    free(processor->timestamps);
    free(processor);
}

/**
 * Add telemetry event to processor
 */
int dsmil_telemetry_processor_add_event(dsmil_telemetry_processor_t *processor,
                                      const dsmil_telemetry_event_t *event) {
    if (!processor || !event || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (processor->event_count >= processor->max_events) {
        return DSMIL_SEARCH_ERROR_MEMORY;
    }

    if (processor->event_count > 0 &&
        processor->events[processor->event_count - 1].timestamp > event->timestamp) {
        processor->timestamps_sorted = false;
    }

    // Copy event
    processor->events[processor->event_count] = *event;
    processor->timestamps[processor->event_count] = (int64_t)event->timestamp;
    processor->event_count++;

    return DSMIL_SEARCH_SUCCESS;
}

/**
 * Find telemetry event by timestamp (KEYSTONE accelerated)
 */
int dsmil_telemetry_processor_find_by_timestamp(
    dsmil_telemetry_processor_t *processor,
    dsmil_timestamp_t target_time,
    dsmil_telemetry_result_t *result
) {
    if (!processor || !result || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    if (processor->event_count == 0) {
        telemetry_fill_not_found(result);
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    if (processor->timestamps_sorted) {
        dsmil_search_index_t index = {
            .keys = processor->timestamps,
            .num_elements = processor->event_count
        };

        return dsmil_search_telemetry_events_indexed(
            processor->search_ctx,
            &index,
            processor->events,
            target_time,
            result
        );
    }

    // KEYSTONE expects sorted keys; preserve correctness for unsorted append streams.
    for (size_t i = 0; i < processor->event_count; i++) {
        const dsmil_telemetry_event_t *event = &processor->events[i];
        if (event->timestamp == target_time) {
            result->event = event;
            result->index = i;
            result->exact_match_time = event->timestamp;
            result->is_exact_match = true;
            return DSMIL_SEARCH_SUCCESS;
        }
    }

    telemetry_fill_not_found(result);
    return DSMIL_SEARCH_ERROR_NOT_FOUND;
}

/**
 * Find telemetry events in time range
 */
int dsmil_telemetry_processor_find_in_time_range(
    dsmil_telemetry_processor_t *processor,
    dsmil_timestamp_t start_time,
    dsmil_timestamp_t end_time,
    dsmil_telemetry_result_t *results,
    size_t max_results,
    size_t *num_found
) {
    if (!processor || !results || !num_found || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    size_t found = 0;

    if (start_time > end_time || max_results == 0 || processor->event_count == 0) {
        *num_found = 0;
        return DSMIL_SEARCH_SUCCESS;
    }

    if (processor->timestamps_sorted) {
        size_t begin = telemetry_lower_bound_timestamp(processor->timestamps,
                                                       processor->event_count,
                                                       start_time);
        size_t end = telemetry_upper_bound_timestamp(processor->timestamps,
                                                     processor->event_count,
                                                     end_time);

        for (size_t i = begin; i < end && found < max_results; i++) {
            telemetry_fill_range_result(&results[found], &processor->events[i], i);
            if (processor->events[i].timestamp == start_time || processor->events[i].timestamp == end_time) {
                results[found].is_exact_match = true;
            }
            found++;
        }

        *num_found = found;
        return DSMIL_SEARCH_SUCCESS;
    }

    // Preserve existing behavior for conservatively handled unsorted input.
    for (size_t i = 0; i < processor->event_count && found < max_results; i++) {
        const dsmil_telemetry_event_t *event = &processor->events[i];

        if (event->timestamp >= start_time && event->timestamp <= end_time) {
            telemetry_fill_range_result(&results[found], event, i);
            found++;
        }
    }

    *num_found = found;
    return DSMIL_SEARCH_SUCCESS;
}

/**
 * Find events by device ID within time range
 */
int dsmil_telemetry_processor_find_by_device_time_range(
    dsmil_telemetry_processor_t *processor,
    uint32_t device_id,
    dsmil_timestamp_t start_time,
    dsmil_timestamp_t end_time,
    dsmil_telemetry_result_t *results,
    size_t max_results,
    size_t *num_found
) {
    if (!processor || !results || !num_found || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    size_t found = 0;

    if (start_time > end_time || max_results == 0 || processor->event_count == 0) {
        *num_found = 0;
        return DSMIL_SEARCH_SUCCESS;
    }

    if (processor->timestamps_sorted) {
        size_t begin = telemetry_lower_bound_timestamp(processor->timestamps,
                                                       processor->event_count,
                                                       start_time);
        size_t end = telemetry_upper_bound_timestamp(processor->timestamps,
                                                     processor->event_count,
                                                     end_time);

        for (size_t i = begin; i < end && found < max_results; i++) {
            const dsmil_telemetry_event_t *event = &processor->events[i];

            if (event->device_id == device_id) {
                telemetry_fill_range_result(&results[found], event, i);
                if (event->timestamp == start_time || event->timestamp == end_time) {
                    results[found].is_exact_match = true;
                }
                found++;
            }
        }

        *num_found = found;
        return DSMIL_SEARCH_SUCCESS;
    }

    // Preserve existing behavior for conservatively handled unsorted input.
    for (size_t i = 0; i < processor->event_count && found < max_results; i++) {
        const dsmil_telemetry_event_t *event = &processor->events[i];

        if (event->device_id == device_id &&
            event->timestamp >= start_time &&
            event->timestamp <= end_time) {

            telemetry_fill_range_result(&results[found], event, i);
            found++;
        }
    }

    *num_found = found;
    return DSMIL_SEARCH_SUCCESS;
}

/**
 * Get telemetry processing statistics
 */
int dsmil_telemetry_processor_get_stats(
    const dsmil_telemetry_processor_t *processor,
    uint32_t *total_events,
    uint32_t *search_operations,
    double *avg_search_time_ns,
    uint32_t *memory_usage
) {
    if (!processor || !total_events || !search_operations ||
        !avg_search_time_ns || !memory_usage) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    *total_events = (uint32_t)processor->event_count;
    *search_operations = 0;
    *avg_search_time_ns = 0.0;
    *memory_usage = 0;

    // Get search statistics from KEYSTONE context
    double cache_hit_rate;
    int rc = dsmil_search_get_stats(
        processor->search_ctx,
        search_operations,
        &cache_hit_rate,
        memory_usage,
        avg_search_time_ns
    );
    return rc;
}

/**
 * Clear all telemetry events
 */
int dsmil_telemetry_processor_clear(dsmil_telemetry_processor_t *processor) {
    if (!processor || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    processor->event_count = 0;
    processor->timestamps_sorted = true;

    // Reset search statistics
    return dsmil_search_reset_stats(processor->search_ctx);
}

/* ============================================================================
 * tar.zst Streaming Loaders
 * ============================================================================ */

#ifdef KEYSTONE_ENABLE_TAR_ZST

int dsmil_telemetry_processor_load_from_tar_zst(
    dsmil_telemetry_processor_t *processor,
    const char *archive_path,
    const char *member_name
) {
    if (!processor || !archive_path || !member_name || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

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
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    int64_t *keys = NULL;
    size_t count = 0;
    if (keystone_tar_zst_extract_member(tz, member_name, &keys, &count) != 0) {
        keystone_tar_zst_close(tz);
        return DSMIL_SEARCH_ERROR_NOT_FOUND;
    }

    int result = DSMIL_SEARCH_SUCCESS;
    if (keys && count > 0) {
        for (size_t i = 0; i < count; i++) {
            dsmil_telemetry_event_t event = {
                .timestamp = (dsmil_timestamp_t)keys[i],
                .event_type = 0,
                .device_id = 0,
                .layer_id = 0,
                .event_data = NULL,
                .data_size = 0
            };
            int rc = dsmil_telemetry_processor_add_event(processor, &event);
            if (rc != DSMIL_SEARCH_SUCCESS) {
                result = rc;
                break;
            }
        }
    }

    keystone_tar_zst_close(tz);
    return result;
}

/* Simple glob matcher supporting * (any chars) and ? (one char) */
static int glob_match(const char *pattern, const char *text) {
    const char *star = NULL;
    const char *backup = NULL;
    while (*text) {
        if (*pattern == *text || *pattern == '?') {
            pattern++;
            text++;
        } else if (*pattern == '*') {
            star = pattern++;
            backup = text;
        } else if (star) {
            pattern = star + 1;
            text = ++backup;
        } else {
            return 0;
        }
    }
    while (*pattern == '*') pattern++;
    return *pattern == '\0';
}

static int member_is_telemetry(const char *name) {
    static const char *patterns[] = {
        "*.telemetry", "*.bin", "*.csv", "*.json", "*.txt"
    };
    size_t num = sizeof(patterns) / sizeof(patterns[0]);
    for (size_t i = 0; i < num; i++) {
        if (glob_match(patterns[i], name)) return 1;
    }
    return 0;
}

int dsmil_telemetry_processor_load_all_members(
    dsmil_telemetry_processor_t *processor,
    const char *archive_path
) {
    if (!processor || !archive_path || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

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
        return DSMIL_SEARCH_ERROR_INIT_FAILED;
    }

    int result = DSMIL_SEARCH_SUCCESS;

    for (;;) {
        char *name = NULL;
        size_t name_len = 0;
        int r = keystone_tar_zst_next_member(tz, &name, &name_len);
        if (r <= 0) break;
        (void)name_len;

        if (!member_is_telemetry(name)) continue;

        int64_t *keys = NULL;
        size_t count = 0;
        if (keystone_tar_zst_extract_member(tz, name, &keys, &count) != 0) {
            continue;
        }
        if (!keys || count == 0) continue;

        for (size_t i = 0; i < count; i++) {
            dsmil_telemetry_event_t event = {
                .timestamp = (dsmil_timestamp_t)keys[i],
                .event_type = 0,
                .device_id = 0,
                .layer_id = 0,
                .event_data = NULL,
                .data_size = 0
            };
            int rc = dsmil_telemetry_processor_add_event(processor, &event);
            if (rc != DSMIL_SEARCH_SUCCESS) {
                result = rc;
                goto done;
            }
        }
    }

done:
    keystone_tar_zst_close(tz);
    return result;
}

#endif /* KEYSTONE_ENABLE_TAR_ZST */

/* ============================================================================
 * High-Level Telemetry Analysis Functions
 * ============================================================================ */

/**
 * Analyze telemetry patterns using KEYSTONE acceleration
 */
int dsmil_telemetry_processor_analyze_patterns(
    dsmil_telemetry_processor_t *processor,
    dsmil_timestamp_t analysis_window_start,
    dsmil_timestamp_t analysis_window_end,
    char *analysis_report,
    size_t max_report_length
) {
    if (!processor || !analysis_report || !processor->initialized) {
        return DSMIL_SEARCH_ERROR_INVALID_PARAM;
    }

    // Find events in analysis window
    dsmil_telemetry_result_t *events_in_window = calloc(processor->event_count,
                                                      sizeof(dsmil_telemetry_result_t));
    if (!events_in_window) {
        return DSMIL_SEARCH_ERROR_MEMORY;
    }

    size_t num_events_found;
    int ret = dsmil_telemetry_processor_find_in_time_range(
        processor,
        analysis_window_start,
        analysis_window_end,
        events_in_window,
        processor->event_count,
        &num_events_found
    );

    if (ret != DSMIL_SEARCH_SUCCESS) {
        free(events_in_window);
        return ret;
    }

    // Generate analysis report with overflow guard
    size_t report_len = 0;
    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "DSMIL Telemetry Analysis Report (KEYSTONE Accelerated)\n");
    }
    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "====================================================\n\n");
    }
    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "Analysis Window: %llu - %llu\n",
                              (unsigned long long)analysis_window_start,
                              (unsigned long long)analysis_window_end);
    }
    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "Events Found: %zu\n", num_events_found);
    }
    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "Search Performance: KEYSTONE accelerated lookup\n\n");
    }

    /* Maximum device IDs tracked in pattern analysis (configurable limit). */
    #define DSMIL_TELEMETRY_MAX_DEVICES 256
    uint32_t device_counts[DSMIL_TELEMETRY_MAX_DEVICES] = {0};
    uint32_t max_device_count = 0;
    uint32_t most_active_device = 0;

    for (size_t i = 0; i < num_events_found; i++) {
        uint32_t device_id = events_in_window[i].event->device_id;
        if (device_id < DSMIL_TELEMETRY_MAX_DEVICES) {
            device_counts[device_id]++;
            if (device_counts[device_id] > max_device_count) {
                max_device_count = device_counts[device_id];
                most_active_device = device_id;
            }
        }
    }

    if (report_len < max_report_length) {
        report_len += snprintf(analysis_report + report_len, max_report_length - report_len,
                              "Most Active Device: %u (%u events)\n",
                              most_active_device, max_device_count);
    }

    free(events_in_window);
    return DSMIL_SEARCH_SUCCESS;
}
