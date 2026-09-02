# DSMIL Telemetry Processor

The DSMIL Telemetry Processor acts as a bridge between KEYSTONE's core `int64_t` batch search capabilities and high-level structured data from archives like `.tar.zst` containing CSV, JSON, or text logs.

## Input and Output Contracts

### Inputs
1. **Archive Members**: The processor consumes `.tar.zst` streams using either targeted member ingestion (`dsmil_telemetry_processor_load_from_tar_zst`) or bulk ingestion across all recognized member patterns (`dsmil_telemetry_processor_load_all_members`). Decompression is bounded, chunked, and supports multi-threaded ring buffering.
2. **Key Extraction**:
   - For **CSV**: Parses rows, ignoring header lines if configured. Treats the first valid numeric column (or a specific designated column) as the key. Malformed lines or lines containing non-numeric data are safely ignored.
   - For **JSON**: Looks for arrays of numbers, or specific `"timestamp"`/`"id"` keys, attempting to extract 64-bit integers.
   - For **TXT**: Reads one 64-bit integer per line.
3. **Sorting & Verification**: Extracted keys are sorted in-place, and event timestamps are indexed within the telemetry processor's internal state. When companion `.idx.json` sidecars exist, candidate files and members are pre-filtered via min/max bounds and Bloom filters before decompression begins.

### Outputs
1. **Telemetry Results**: 
   - A `dsmil_telemetry_result_t` is returned for each queried key.
   - `is_exact_match`: Boolean indicating whether an exact match was found.
   - `exact_match_time`: The matched key itself, confirming the hit.
2. **Missing Keys (Failures)**:
   - If a key is missing, `is_exact_match` is set to `0`. No exception is thrown; it is a valid state for lookups to miss.
   - Missing lookups are aggressively rejected by Bloom filters at the index level (if indexing is enabled) before hitting the binary search logic.

## Operational Guidance

### Untrusted Input Handling
When ingesting archives from external or untrusted sources:
- **Corrupt Members**: Files with malformed headers, invalid JSON, or textual garbage in CSV columns are gracefully skipped. Keys that cannot be parsed as `int64_t` are simply discarded. The system uses a strict memory-safe arena parsing engine.
- **Decompression Bombs**: The processor employs a strict cap (`KEYSTONE_TAR_ZST_MAX_DECOMPRESS_BYTES`) on the maximum uncompressed size of any single archive member. If a member exceeds this limit, parsing is aborted with an error, preventing out-of-memory attacks.

### Audit Trails
All access to archive data is logged in the `keystone_tar_zst_stats_t` telemetry object, tracking:
- Number of members decompressed.
- Bytes successfully parsed.
- Aggregate parsing and decompression time.
These stats should be forwarded to your application's audit logging infrastructure to detect tampering or suspiciously small/large ingest patterns.

### Separation of Data
- **Benchmark Data**: Benchmark environments using `test_data.tar.zst` or large synthetic telemetry dumps must execute in separate namespaces or directories from production data.
- **Production Isolation**: Under no circumstances should benchmarking tools (like `compare_search_auto` or `dsmil_benchmark`) be pointed at live production `.tar.zst` archives. The benchmark runner modifies cache behaviors and intentionally alters thread counts to probe performance bounds, which can negatively impact co-resident production workloads.
