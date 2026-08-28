# Changelog

All notable changes to the KEYSTONE search engine are documented in this file.

## [1.2.0] - 2026-08-28

### Security Fixes (P1/P2)
- **P1: tar.zst parser OOB read** — Replaced unbounded `strtoll()` with a bounded streaming integer parser that respects buffer length. Eliminates out-of-bounds read vulnerability in `.tar.zst` integer parsing.
- **P1: CUDA cache race** — Replaced spinlock-released-before-use pattern with a reader-lease protocol (refcount + generation). Eviction now waits for `readers == 0` before `cudaFree`. Added `keystone_search_batch_cuda_versioned()` with `dataset_version` for in-place host array mutation detection. Added `keystone_cuda_cache_invalidate()`.
- **P2: Archive index min/max** — `first_key`/`last_key` now recomputed from the sorted array after parsing, not trusted from stream order (wrong for unsorted source data).
- **P2: Auto-backend data races** — `g_backend_cache` and `g_last_backend_decision` protected by mutexes; `valid=1` published last after all fields written. Eliminates torn reads on concurrent access.
- **P2: Signed overflow in query-shape classifier** — All key deltas and `max-min` range computed in `__int128`, eliminating UB near `INT64_MIN`/`INT64_MAX`.
- **P2: FNV hash collision verification** — Hash indexer now retains original string bytes and verifies them on every positive hit, eliminating false matches from 64-bit hash collisions.
- **QIHSE ingestion principal** — Bridge carries an authenticated `ingestion_principal` via `keystone_qihse_bridge_set_principal()`. New `keystone_qihse_bridge_dispatch_credential_authenticated()` uses `qihse_kv_set_user()` and refuses writes without a principal, per QIHSE AGENTS.md invariant #1.

### Performance
- **Archive index keys retention** — Sorted keys retained in `tar_zst_index_entry`, eliminating repeat decompression/parsing for positive lookups. Fallback re-streams if keys not retained.
- **LSD radix sort** — Replaced `qsort` with 8-pass LSD radix sort (O(n), sequential memory access) for 64-bit hash keys, carrying offsets/strings/lens.
- **Zero-copy NumPy batch API** — New `keystone_search_keys_batch_auto()` takes raw `int64_t*` keys and `size_t*` results directly from NumPy buffers. Python `search_batch_keys()` skips per-key `_CBatchItem` marshalling. **15.6x faster** on 1M queries (0.151s vs 2.351s).

### Correctness
- **Anchor LRU tracking** — Endpoint anchors now initialize `use_count`/`last_used`; usage-update block no longer guards on `active_table != table`, so caller table anchors get LRU timestamps refreshed.

### Sandy Bridge / AVX1-only CPU Support
- **SSE4.2 SIMD path** — Added branchless 128-bit SIMD path (`_mm_cmpeq_epi64` / PCMPEQQ) to `keystone_chunked_search`, 2x unrolled for Sandy Bridge's dual 128-bit execution ports. Previously AVX1-only CPUs fell through to a scalar loop that couldn't auto-vectorize.
- **Double-precision interpolation** — Replaced `__int128` division (80-100+ cycle libgcc `__divti3` call) with double-precision fast path (~20-40 cycles). `__int128` fallback only for overflow edge cases. **2x faster single-key search** on Sandy Bridge.
- **Software prefetch enabled for SSE4.2** — Prefetch was `#ifdef`'d out on AVX1-only CPUs. Added SSE4.2 branch with Sandy Bridge-tuned distances (32/64 elements vs 64/128).
- **Branchless scalar fallback** — Removed early returns that blocked GCC auto-vectorization.
- **Wider SIMD scan window** — `keystone_local_search` uses 64-element window on SSE4.2+ (was fixed at 32).
- **OpenMP auto-enabled** — Makefile auto-detects compiler OpenMP support and enables `-fopenmp` by default. **2x faster batch search** on 8-core machines.
- **Lowered parallel threshold** — Auto-backend uses OpenMP for batches >= 4096 items (was 16384). Configurable via `KEYSTONE_AUTO_PARALLEL_MIN_ITEMS`.

### Benchmark Results (Sandy Bridge Xeon E5-2407, 2.2GHz, 8-core)
| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Single-key search | 317 ns | 157 ns | 2.0x |
| Batch (serial) | 400 ns | 330 ns | 1.2x |
| Batch (auto+OpenMP) | N/A | 165 ns | 2.4x |
| Small-window scan | 125 ns | 82 ns | 1.5x |

## [1.1.0] - Upcoming

### API Changes
- **Auto-Backend Router**: Added `keystone_search_batch_auto` for intelligent, shape-aware batch query routing across Scalar, SIMD, and parallel boundaries.
- **Decision Provenance**: Exposed `keystone_backend_decision_t` and `keystone_get_last_backend_decision()` enabling full inspection of the router's hardware choices, calibration runs, and detected array shapes.
- **Archive Streamer API**: Introduced `keystone_tar_zst_t` and related methods for direct search over compressed archives (`.tar.zst`).
- **Telemetry Processing**: Introduced `dsmil_telemetry_processor.h` combining archive parsing with KEYSTONE's search.
- **Stable Metadata Documentation**: Promoted `keystone_backend_decision_t` fields to stable Doxygen comments.

### Benchmark Changes
- **CSV & JSON Reporting**: Integrated `-csv` and `-json` outputs in `dsmil_benchmark` and `performance_proof` for easier dashboard consumption.
- **Matrix Runner**: Created `run_perf_matrix.sh` with `perf stat` hardware counter tracking (cache misses, branch misses).
- **Archive Benchmarking**: Large archive extraction (`test_data.tar.zst`) now leverages `dsmil_search_batch_tar_zst` in batch workloads for extremely high-throughput telemetry simulation.
- **Host Capability Logs**: The benchmark suite now prints explicit capabilities (AVX2, AVX-512, AMX, Graviton4) prior to executing search workloads.

### Platform-Specific Behavior Changes
- **AVX-512 Isolation**: Separated the experimental `__AVX512F__` search logic out of the main compiler path into `keystone_avx512.c`, compiling securely with `-mavx512f -mavx512dq`.
- **AMX Constraints**: Bounded AMX usage strictly to "feature detection only," awaiting tile-stride integer search correctness validation.
- **Graviton4 Profiling**: Explicitly introduced Neoverse V2 caching topology logic locking anchors directly to the 2MB private L2 cache.
- **Build Mode Explicit Requirements**: Created robust dependency checks allowing fallback scalar implementations (`KEYSTONE_FORCE_SCALAR`) or fully parallel setups (`KEYSTONE_ENABLE_OPENMP`, `KEYSTONE_ENABLE_FORTRAN`).
