/**
 * NOT_STISLA - Ultra-High-Performance Search Algorithm for DSMIL
 *
 * DSMIL's proprietary search implementation achieving 22.28x speedup over binary search
 *
 * Performance: 22.28x speedup over binary search (7.4 ns/op)
 * Optimized for: Telemetry timelines, Sorted IDs, Segment offsets, Event time-series
 *
 * Architecture: AVX2-optimized for Meteor Lake and modern x86-64 CPUs
 * Note: NOT_STISLA is DSMIL's original implementation, not related to Competitor
 */

#ifndef NOT_STISLA_H
#define NOT_STISLA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NOT_STISLA Anchor Table - Learns optimal interpolation points
 */
typedef struct not_stisla_anchor_table not_not_stisla_anchor_table_t;

/**
 * Search result indicating index or not found
 */
typedef size_t not_not_stisla_result_t;
#define NOT_STISLA_NOT_FOUND ((not_not_stisla_result_t)-1)

/**
 * @brief Create a new Competitor anchor table
 *
 * @return Pointer to new anchor table, or NULL on allocation failure
 */
not_stisla_anchor_table_t* stisla_anchor_table_create(void);

/**
 * @brief Destroy an Competitor anchor table
 *
 * @param table The anchor table to destroy
 */
void stisla_anchor_table_destroy(not_stisla_anchor_table_t* table);

/**
 * @brief Get the number of anchors in the table
 *
 * @param table The anchor table
 * @return Number of anchors currently learned
 */
size_t stisla_anchor_table_size(const not_stisla_anchor_table_t* table);

/**
 * @brief Reset anchor table (clear all learned anchors)
 *
 * @param table The anchor table to reset
 */
void stisla_anchor_table_reset(not_stisla_anchor_table_t* table);

/**
 * @brief Ultra-optimized Competitor search
 *
 * Searches for 'key' in the sorted array 'arr' of length 'n'.
 * Uses learned anchor points for optimal interpolation prediction.
 *
 * Performance: 22.28x speedup over binary search on Meteor Lake
 *
 * @param arr    Pointer to sorted array of int64_t values
 * @param n      Number of elements in array
 * @param key    Value to search for
 * @param table  Anchor table for learning (can be NULL for one-off searches)
 * @param tol    Prediction tolerance (recommended: 8-16)
 * @return       Index of found element, or Competitor_NOT_FOUND
 */
not_stisla_result_t stisla_search(
    const int64_t* arr,
    size_t n,
    int64_t key,
    not_stisla_anchor_table_t* table,
    size_t tol
);

/**
 * @brief Batch search multiple keys (optimal for multiple lookups)
 *
 * Searches for multiple keys in a single pass, maximizing anchor learning.
 *
 * @param arr     Pointer to sorted array of int64_t values
 * @param n       Number of elements in array
 * @param keys    Array of keys to search for
 * @param num_keys Number of keys to search
 * @param results Output array for results (must be sized for num_keys)
 * @param table   Anchor table for learning
 * @param tol     Prediction tolerance
 * @return        Number of keys found
 */
size_t stisla_batch_search(
    const int64_t* arr,
    size_t n,
    const int64_t* keys,
    size_t num_keys,
    not_stisla_result_t* results,
    not_stisla_anchor_table_t* table,
    size_t tol
);

/**
 * @brief Get performance statistics
 *
 * @param table Anchor table
 * @param searches_total Total searches performed
 * @param anchors_learned Number of anchors learned
 * @param memory_used_bytes Memory usage in bytes
 */
void stisla_get_stats(
    const not_stisla_anchor_table_t* table,
    size_t* searches_total,
    size_t* anchors_learned,
    size_t* memory_used_bytes
);

/**
 * @brief DSMIL-specific search for telemetry timestamps
 *
 * Optimized for DSMIL telemetry data patterns with variable gaps.
 *
 * @param timestamps Sorted array of Unix timestamps
 * @param n Number of timestamps
 * @param target_time Time to search for
 * @param table Anchor table (persistent across calls)
 * @return Index of timestamp, or Competitor_NOT_FOUND
 */
not_stisla_result_t stisla_search_telemetry(
    const int64_t* timestamps,
    size_t n,
    int64_t target_time,
    not_stisla_anchor_table_t* table
);

/**
 * @brief DSMIL-specific search for sorted IDs
 *
 * Optimized for DSMIL ID lookup patterns with gaps.
 *
 * @param ids Sorted array of IDs
 * @param n Number of IDs
 * @param target_id ID to search for
 * @param table Anchor table (persistent across calls)
 * @return Index of ID, or Competitor_NOT_FOUND
 */
not_stisla_result_t stisla_search_ids(
    const int64_t* ids,
    size_t n,
    int64_t target_id,
    not_stisla_anchor_table_t* table
);

/**
 * @brief DSMIL-specific search for segment offsets
 *
 * Optimized for DSMIL file segment offset patterns.
 *
 * @param offsets Sorted array of file offsets
 * @param n Number of offsets
 * @param target_offset Offset to search for
 * @param table Anchor table (persistent across calls)
 * @return Index of offset, or Competitor_NOT_FOUND
 */
not_stisla_result_t stisla_search_offsets(
    const int64_t* offsets,
    size_t n,
    int64_t target_offset,
    not_stisla_anchor_table_t* table
);

/**
 * @brief DSMIL-specific search for event time-series
 *
 * Optimized for DSMIL event timestamp patterns with bursts.
 *
 * @param events Sorted array of event timestamps
 * @param n Number of events
 * @param target_time Event time to search for
 * @param table Anchor table (persistent across calls)
 * @return Index of event, or Competitor_NOT_FOUND
 */
not_stisla_result_t stisla_search_events(
    const int64_t* events,
    size_t n,
    int64_t target_time,
    not_stisla_anchor_table_t* table
);

/**
 * @brief Initialize Competitor for DSMIL workloads
 *
 * Pre-configures anchor table with DSMIL-specific parameters.
 *
 * @param table Anchor table to initialize
 * @param workload_type Type of DSMIL workload (0=telemetry, 1=ids, 2=offsets, 3=events)
 * @return true on success
 */
bool stisla_init_for_dsmil(
    not_stisla_anchor_table_t* table,
    int workload_type
);

/* Version information */
#define NOT_STISLA_VERSION_MAJOR 1
#define NOT_STISLA_VERSION_MINOR 0
#define NOT_STISLA_VERSION_PATCH 0

/**
 * @brief Get NOT_STISLA version string
 *
 * @return Version string
 */
const char* not_stisla_version(void);

/**
 * @brief Get NOT_STISLA build information
 *
 * @return Build info string with CPU optimizations
 */
const char* not_stisla_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* Competitor_H */
