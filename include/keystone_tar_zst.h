/**
 * KEYSTONE - tar.zst streaming search interface
 *
 * Provides live streaming decompression and search over .tar.zst archives
 * without fully materialising the archive in memory.
 */

#ifndef KEYSTONE_TAR_ZST_H
#define KEYSTONE_TAR_ZST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations from keystone.h */
typedef size_t keystone_result_t;
typedef struct keystone_anchor_table keystone_anchor_table_t;
typedef struct keystone_config keystone_config_t;
typedef struct keystone_batch_item keystone_batch_item_t;
typedef struct keystone_parallel_config keystone_parallel_config_t;
#define KEYSTONE_NOT_FOUND ((keystone_result_t)-1)

/* ============================================================================
 * Options & Types
 * ============================================================================ */

typedef enum keystone_tar_zst_format {
    KEYSTONE_TAR_ZST_FORMAT_AUTO = 0,
    KEYSTONE_TAR_ZST_FORMAT_CSV,
    KEYSTONE_TAR_ZST_FORMAT_JSON,
    KEYSTONE_TAR_ZST_FORMAT_TEXT
} keystone_tar_zst_format_t;

typedef struct keystone_tar_zst_options {
    keystone_tar_zst_format_t format;
    int zstd_workers;       /* >1 enables multi-threaded zstd decompression */
    int enable_pipeline;    /* non-zero enables producer/consumer pipelining */
    int auto_load_index;    /* non-zero auto-checks and loads <path>.idx.json if present */
    int retain_keys;        /* non-zero keeps full sorted key copies in memory on index build */
    size_t chunk_size;      /* I/O chunk size (default 256 KiB) */
    size_t arena_slab_size; /* Arena slab size (default 1 MiB) */
    int skip_header;        /* For CSV: skip first line as header */
} keystone_tar_zst_options_t;

typedef struct keystone_tar_zst keystone_tar_zst_t;

/* ============================================================================
 * Core Archive API
 * ============================================================================ */

/**
 * @brief Open a .tar.zst archive for streaming search.
 *
 * @param path Path to the archive file.
 * @param opts Options; may be NULL for defaults.
 * @return Handle or NULL on error.
 */
keystone_tar_zst_t* keystone_tar_zst_open(const char* path,
                                              const keystone_tar_zst_options_t* opts);

/**
 * @brief Close archive and release all resources.
 */
void keystone_tar_zst_close(keystone_tar_zst_t* tz);

/**
 * @brief Rewind the archive reader back to the beginning of the archive.
 *
 * @param tz Archive handle.
 * @return 0 on success, <0 on error.
 */
int keystone_tar_zst_rewind(keystone_tar_zst_t* tz);

/**
 * @brief Advance to the next archive member.
 *
 * @param tz Archive handle.
 * @param out_name Receives pointer to member name (valid until next call).
 * @param out_name_len Receives length of member name.
 * @return 1 if a member was found, 0 at end-of-archive, <0 on error.
 */
int keystone_tar_zst_next_member(keystone_tar_zst_t* tz,
                                   char** out_name,
                                   size_t* out_name_len);

/**
 * @brief Search a named member by streaming, parsing, then KEYSTONE search.
 *
 * Streams the named member, parses into an int64_t buffer via the arena,
 * then calls keystone_search_enhanced. The buffer is reset after search.
 *
 * @param tz Archive handle.
 * @param member_name Name of the tar entry to search.
 * @param key Key to search for.
 * @param table Anchor table (may be NULL).
 * @param config Search config (may be NULL for defaults).
 * @return Search result or KEYSTONE_NOT_FOUND.
 */
keystone_result_t keystone_tar_zst_search_member(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config
);

/**
 * @brief Batch search a named member.
 *
 * Streams the member once, parses into int64_t buffer, then runs
 * keystone_search_batch_auto. The buffer is reset after search.
 */
size_t keystone_tar_zst_search_member_batch(
    keystone_tar_zst_t* tz,
    const char* member_name,
    keystone_batch_item_t* items,
    size_t num_items,
    keystone_anchor_table_t* table,
    size_t tol,
    const keystone_parallel_config_t* config
);

/**
 * @brief Extract parsed int64_t keys from a named member.
 *
 * Streams and parses the member.  The returned keys pointer is owned by
 * the arena and remains valid until the next parse operation or close.
 *
 * @param tz Archive handle.
 * @param member_name Name of the tar entry to extract.
 * @param out_keys Receives pointer to the parsed key array.
 * @param out_count Receives number of keys.
 * @return 0 on success, <0 on error.
 */
int keystone_tar_zst_extract_member(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t** out_keys,
    size_t* out_count
);

/* ============================================================================
 * Offset Index API
 * ============================================================================ */

/**
 * @brief Build an in-memory member offset index by scanning the archive once.
 *
 * After this call the handle is rewound; subsequent search-by-name calls
 * can jump directly to indexed members.
 *
 * @return 0 on success, <0 on error.
 */
int keystone_tar_zst_build_index(keystone_tar_zst_t* tz);

/**
 * @brief Search using the pre-built offset index (faster for repeated access).
 */
keystone_result_t keystone_tar_zst_search_indexed(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config
);

/**
 * @brief Save the current index to a persistent sidecar index file (e.g. .idx.json).
 *
 * @param tz Archive handle with built index.
 * @param path Destination path; if NULL, defaults to "<archive_path>.idx.json".
 * @return 0 on success, <0 on error.
 */
int keystone_tar_zst_save_index(keystone_tar_zst_t* tz, const char* path);

/**
 * @brief Load an index from a persistent sidecar index file without scanning the archive.
 *
 * @param tz Archive handle.
 * @param path Source path; if NULL, defaults to "<archive_path>.idx.json".
 * @return 0 on success, <0 on error.
 */
int keystone_tar_zst_load_index(keystone_tar_zst_t* tz, const char* path);

/* ============================================================================
 * Multi-Archive Batch API
 * ============================================================================ */

typedef struct keystone_tar_zst_batch keystone_tar_zst_batch_t;

/**
 * @brief Open multiple .tar.zst archives as a unified batch pool.
 *
 * Archives in the pool can be queried in parallel. Companion .idx.json
 * files are automatically discovered and loaded when present.
 *
 * @param archive_paths Array of archive file paths.
 * @param num_archives Number of archives in the array.
 * @param opts Options for each archive handle (may be NULL).
 * @return Batch pool handle or NULL on error.
 */
keystone_tar_zst_batch_t* keystone_tar_zst_batch_open(
    const char** archive_paths,
    size_t num_archives,
    const keystone_tar_zst_options_t* opts
);

/**
 * @brief Close batch pool and release all underlying archive handles.
 */
void keystone_tar_zst_batch_close(keystone_tar_zst_batch_t* batch);

/**
 * @brief Search across all archives in the batch pool in parallel.
 *
 * Uses pre-loaded or built indices to reject non-matching archives in O(1) time.
 *
 * @param batch The batch pool handle.
 * @param member_name Name of the target archive member.
 * @param key Key to search for.
 * @param table Anchor table (optional).
 * @param config Search config (optional).
 * @param out_archive_idx Optional pointer to receive index of the archive containing match.
 * @return Search result index in the matching member, or KEYSTONE_NOT_FOUND.
 */
keystone_result_t keystone_tar_zst_batch_search(
    keystone_tar_zst_batch_t* batch,
    const char* member_name,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config,
    size_t* out_archive_idx
);

/* ============================================================================
 * Error & Stats Helpers
 * ============================================================================ */

/**
 * @brief Get a human-readable error string for the last operation.
 */
const char* keystone_tar_zst_error_string(keystone_tar_zst_t* tz);

/**
 * @brief Get archive-level statistics.
 */
typedef struct keystone_tar_zst_stats {
    uint64_t members_read;
    uint64_t bytes_read;
    uint64_t decompress_time_ns;
    uint64_t parse_time_ns;
} keystone_tar_zst_stats_t;

int keystone_tar_zst_get_stats(keystone_tar_zst_t* tz,
                                 keystone_tar_zst_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* KEYSTONE_TAR_ZST_H */
