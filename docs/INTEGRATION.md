# Integrating KEYSTONE

KEYSTONE is designed as a standalone, ultra-low latency interpolation search engine for sorted 64-bit integer arrays. It acts as a powerful accelerator for indexing, telemetry, and time-series data structures.

**Important Note:** KEYSTONE is *not* a full database. It does not manage disk persistence, transactional consistency (ACID), or query parsing (SQL). It is a "database-adjacent" component—a highly specialized data structure you embed within your own storage or analytical engines to accelerate point lookups and batch searches.

## 1. Basic Single-Key Lookup

For a simple query on a sorted array, KEYSTONE operates similarly to `bsearch()` but learns the data distribution to achieve near `O(1)` performance on subsequent lookups.

```c
#include "keystone.h"

int main() {
    // 1. Create the anchor table (the "index" or "model" of the data)
    keystone_anchor_table_t* table = keystone_anchor_table_create();
    
    // 2. The data array MUST be sorted.
    int64_t data[] = { 10, 20, 30, 40, 50, ... };
    size_t data_size = sizeof(data) / sizeof(data[0]);
    
    // 3. Search for a key
    int64_t target = 40;
    keystone_result_t result = keystone_search_enhanced(
        data, data_size, target, table, NULL
    );
    
    if (result != KEYSTONE_NOT_FOUND) {
        printf("Found key at index: %zu\n", result);
    }
    
    // 4. Cleanup
    keystone_anchor_table_destroy(table);
    return 0;
}
```

## 2. Batch Lookup with the Auto-Backend Router

When processing hundreds or thousands of queries at once, KEYSTONE's `dsmil_keystone_wrapper` can automatically evaluate the data shape and dispatch the batch to the optimal hardware backend (Scalar, AVX2, AVX-512, OpenMP, etc.).

```c
#include "keystone.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    int64_t data[] = { /* sorted int64_t payload */ };
    size_t data_size = ...;
    
    keystone_anchor_table_t* table = keystone_anchor_table_create();
    
    // 1. Prepare batch items
    size_t num_queries = 1000;
    keystone_batch_item_t* items = malloc(num_queries * sizeof(keystone_batch_item_t));
    for (size_t i = 0; i < num_queries; i++) {
        items[i].key = /* target key */;
        items[i].ordinal = i;
    }
    
    // 2. Perform the batch search (Auto-router selects SIMD, scalar, or threaded paths)
    size_t found = keystone_search_batch_auto(
        data, data_size, items, num_queries, table, 0, NULL
    );
    
    // 3. Inspect the Decision Provenance
    keystone_backend_decision_t decision;
    if (keystone_get_last_backend_decision(&decision) == 0) {
        printf("Backend chosen: %s\n", keystone_backend_name(decision.backend));
        printf("Decision source: %s\n", keystone_decision_source_name(decision.decision_source));
        printf("Detected query shape: %s\n", keystone_query_shape_name(decision.query_shape));
        printf("Estimated ns/key: %.2f\n", decision.estimated_ns_per_key);
        printf("Calibration runs performed: %zu\n", decision.calibration_runs);
    }
    
    // 4. Cleanup
    free(items);
    keystone_anchor_table_destroy(table);
    return 0;
}
```

## 3. Compressed Archive Ingestion (.tar.zst)

KEYSTONE provides built-in utilities to stream directly from `.tar.zst` archives without inflating the entire file into memory or extracting to disk.

### 3.1 Streaming Search & Member Extraction

```c
#include "keystone_tar_zst.h"
#include <stdio.h>

int search_archive_example() {
    keystone_tar_zst_options_t opts = {
        .format = KEYSTONE_TAR_ZST_FORMAT_AUTO,
        .enable_pipeline = 1,        // Dual-threaded producer/consumer ring buffer
        .chunk_size = 256 * 1024,
        .arena_slab_size = 1024 * 1024
    };

    keystone_tar_zst_t* tz = keystone_tar_zst_open("logs_2026.tar.zst", &opts);
    if (!tz) return -1;

    // Search for key 1500 directly within a member (auto-rewinds if member precedes cursor)
    keystone_result_t idx = keystone_tar_zst_search_member(
        tz, "telemetry.csv", 1500, NULL, NULL
    );

    if (idx != KEYSTONE_NOT_FOUND) {
        printf("Found key at sorted offset %zu in telemetry.csv\n", idx);
    }

    keystone_tar_zst_close(tz);
    return 0;
}
```

### 3.2 Persistent Sidecar Indexing (`.idx.json`)

To avoid rescanning archives on every run, generate an accompanying `.idx.json` index:

```c
// Generate and persist sidecar index
keystone_tar_zst_t* tz = keystone_tar_zst_open("logs_2026.tar.zst", NULL);
keystone_tar_zst_build_index(tz);
keystone_tar_zst_save_index(tz, "logs_2026.tar.zst.idx.json");
keystone_tar_zst_close(tz);

// Later / in another process: instant sub-millisecond open via sidecar
keystone_tar_zst_options_t opts = { .auto_load_index = 1 };
keystone_tar_zst_t* fast_tz = keystone_tar_zst_open("logs_2026.tar.zst", &opts);

// Non-matching keys are rejected in O(1) time via Bloom filters
keystone_result_t res = keystone_tar_zst_search_indexed(
    fast_tz, "telemetry.csv", 1500, NULL, NULL
);
keystone_tar_zst_close(fast_tz);
```

### 3.3 Multi-Archive Batch Pools

Query pools of partitioned batch archives in parallel:

```c
const char* archives[] = {
    "batch_00000.tar.zst",
    "batch_00001.tar.zst",
    "batch_00002.tar.zst"
};

// Opens pool and automatically loads companion .idx.json files if present
keystone_tar_zst_batch_t* batch = keystone_tar_zst_batch_open(archives, 3, NULL);

size_t matching_archive = 0;
keystone_result_t result = keystone_tar_zst_batch_search(
    batch, "events.csv", 987654, NULL, NULL, &matching_archive
);

if (result != KEYSTONE_NOT_FOUND) {
    printf("Found key in archive %s at offset %zu\n",
           archives[matching_archive], result);
}

keystone_tar_zst_batch_close(batch);
```

### 3.4 Telemetry Processor Integration

```c
#include "dsmil_telemetry_processor.h"

int load_telemetry_example() {
    dsmil_telemetry_processor_t* processor = dsmil_telemetry_processor_create(10000);
    
    // Decompresses member, parses int64 timestamps, and populates processor
    int rc = dsmil_telemetry_processor_load_from_tar_zst(
        processor, "historical_data.tar.zst", "sensors_2026.csv"
    );
    
    if (rc == DSMIL_SEARCH_SUCCESS) {
        dsmil_telemetry_result_t res;
        dsmil_telemetry_processor_find_by_timestamp(processor, 1718000000, &res);
        if (res.is_exact_match) {
            printf("Found telemetry event at index %zu\n", res.index);
        }
    }
    
    dsmil_telemetry_processor_destroy(processor);
    return 0;
}
```
