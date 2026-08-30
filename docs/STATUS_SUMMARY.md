# KEYSTONE Status Summary

Condensed from DYNAMIC_HOT_PATH_PLAN, FORTRAN_BACKEND_PLAN, IMPROVEMENT_PLAN, OPTIMIZATION_SUMMARY, and PERFORMANCE_ROADMAP.

## Done

### Core Search & SIMD
- Scalar C reference path with correctness cross-checks.
- Runtime CPU feature detection (AVX2, AVX-512, AMX, NEON, SVE).
- AVX2 local scan path for small search windows when built for x86 AVX2.
- AVX-512 local scan path is build-gated and should be treated as target-silicon experimental until measured on real AVX-512 hardware.
- Software prefetching (L1 `_MM_HINT_T0`, L2 `_MM_HINT_T1`) for medium/large arrays.

### Memory
- `keystone_optimize_array_memory()` — transparent huge-page hints via `madvise(MADV_HUGEPAGE)`.
- Huge-page gating by size threshold (>1MB); non-fatal on failure.
- Memory ramp runner (`scripts/run_memory_ramp.sh`) with bounded RAM cap (`MemAvailable` fraction).
- Zero major page faults in pilot up to 128M rows.

### Batch & Auto-Selection
- `keystone_search_batch_auto()` — first-use local calibration across viable scalar/C batch, OpenMP, and Fortran candidates.
- Calibration cache keyed by CPU feature mask, array-size bucket, query-count bucket, thread count.
- Static fallback remains for invalid/uncalibrated fallback cases, but normal uncached decisions use measured median/p95 timing.
- p95 field exposed in public `keystone_backend_decision_t` from calibration samples or cached measured values.
- Decision provenance is exposed as fast path, measured, cache, or static fallback.
- Query shape is exposed as general or dense sorted.
- Cache-hit correctness tests compare result and ordinal arrays across measured and cached selector runs.
- `scripts/compare_search.c` CSV output includes auto backend, decision source, query shape, calibration run count, and candidate count.
- Benchmark matrix runner (`scripts/run_perf_matrix.sh`) with configurable size/query/hit-rate/gap/stride sweeps.

### Fortran Backend
- `fortran/keystone_batch.f90` exports `keystone_batch_search_i64`.
- C adapter `keystone_search_batch_fortran` with `int64_t` ABI.
- Optional native build: enabled explicitly with `KEYSTONE_ENABLE_FORTRAN=1`, or auto-enabled by the Makefile when `gfortran` is present unless `KEYSTONE_ENABLE_FORTRAN=0`.
- Narrow auto-route exception for dense sorted query shapes within the configured query-count window.

### Testing & Benchmarks
- Correctness tests for scalar, batch, parallel API behavior, optional Fortran paths, and archive handling.
- Deterministic hit/miss profiles (100/75/50/25/0%).
- Gap profiles: dense, sparse, jittered.
- Stride profiles: sequential, strided, random.
- Thread scaling sweeps (1–32 threads).

## Still To Do

The items below are the current engineering backlog. The root README intentionally keeps this detail out of the executive overview; see [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) for the surrounding architecture.

### Calibration & Cache
- Add more workload profile fields (hit-rate, gap, stride) to calibration cache keys and decision output.
- Add fallback-policy verification for cache and static fallback paths.
- Add richer host/build metadata to calibration CSV output.

### Fortran
- Expand benchmark coverage beyond dense all-hit workloads.
- Decide whether Fortran stays explicit-only or becomes a broader selectable backend.
- Require 10%+ win in multiple real workload buckets before expanding the narrow auto-route exception.

### RAM-Capacity & Scaling
- Add bounded RAM-capacity timing: 5%/10%/25%/40%/60% of `MemAvailable` with page-fault and RSS reporting.
- Thread-count sweep for C OpenMP on 1M and 4M query batches.
- Add apples-to-apples scalar batch backend for fair out-of-range miss routing.
- True OS cold-cache testing (behind explicit opt-in because dropping caches affects the whole host).

### Architecture Candidates (deferred)
- **AVX-512**: Split into isolated object (`src/keystone_avx512.c`), compile only with AVX-512 flags, validate on real AVX-512 hardware.
- **AMX**: CPU feature detection exists, but no AMX search backend should be claimed until there is a real tiled integer search design.
- **GPU/NPU**: Native-first also includes resident accelerators, but no GPU or NPU backend should be claimed until transfer costs, correctness, fallback behavior, and dispatch provenance are implemented and measured.

### Profiling & Measurement
- Tune prefetch distance per workload instead of global fixed distance.
- Measure cache/TLB behavior with `perf stat` before accepting changes.
- Add `effective_read_giB/s` and `first_touch_ms` to standard benchmark reporting.
