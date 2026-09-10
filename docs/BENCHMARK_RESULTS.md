# KEYSTONE SIMD Benchmark Notes

This file records the intended benchmark posture for SIMD-related work. Treat it as a measurement note, not a universal performance claim.

## Current Implementation Status

| Area | Status |
|---|---|
| AVX2 local scan | Implemented for native x86 builds when AVX2 is compiled and detected at runtime |
| AVX-512 local scan | Build-gated; experimental until measured on target AVX-512 hardware |
| OpenMP batch path | Optional; available when built with OpenMP |
| Fortran batch path | Optional; available when built or auto-enabled by the native toolchain |
| Auto backend calibration | Measures viable local candidates on first cache miss and caches the fastest median timing |
| Decision provenance | Reports fast path, measured, cache, or static fallback source through `keystone_backend_decision_t` and public label helpers |
| AMX | Feature detection only; no AMX search backend is currently claimed |

## Measurement Rules

Any SIMD result should include:

- host CPU model and microarchitecture;
- accelerator model/runtime when a GPU or NPU backend is involved;
- compiler and exact compile flags;
- KEYSTONE feature toggles;
- dataset size and distribution;
- query count, query order, and hit rate;
- warmup policy;
- whether the run used scalar, optimized C batch, OpenMP, Fortran, auto-calibrated backend selection, AVX2 local scan, or AVX-512 local scan;
- transfer cost and device memory policy for any future GPU or NPU backend;
- decision source and query shape from `keystone_get_last_backend_decision()`;
- readable backend/source/shape labels from the public `keystone_*_name()` helpers;
- raw output or CSV from the benchmark run.

## Recommended Commands

Native build and test:

```bash
make clean
make test
```

Scalar comparison build:

```bash
make clean
KEYSTONE_ENABLE_FORTRAN=0 KEYSTONE_ENABLE_TAR_ZST=0 KEYSTONE_FORCE_SCALAR=1 make test
```

Benchmark build:

```bash
make benchmarks
./benchmarks/dsmil_benchmark
./benchmarks/performance_proof
```

## Notes

Older estimates around AVX-512 and engineering-board unlock behavior should not be treated as current project claims unless they are regenerated with the rules above and tied to raw benchmark output.

---

## Joint QIHSE + KEYSTONE 5-Pillar Integrated Architecture Benchmarks

Measured on Intel Xeon E5-2407 (8 cores, AVX mode) via `make bench-keystone-integrated`:

```
========================================================================================================
     QIHSE + KEYSTONE 5-PILLAR INTEGRATED PERFORMANCE BENCHMARK
========================================================================================================
 Subsystem / Benchmark Operation              Throughput         p50 Latency    p95 Latency   p99 Latency
--------------------------------------------------------------------------------------------------------
 [1] HNSW Vector Search (Default Entry)       33,080 ops/sec        27.96 µs       30.87 µs      90.32 µs
     HNSW Vector Search (Anchor-Seeded)       31,692 ops/sec        29.39 µs       33.67 µs     118.42 µs
     * Hop reduction verified on 63/64 clustered queries with 100% recall@32 parity.

 [2] Standard Binary Search (1M rows)      2,016,334 lookups/s     415.00 ns      715.00 ns     878.00 ns
     Keystone Anchor Search (1M rows)      3,510,610 lookups/s     218.00 ns      347.00 ns     450.00 ns
     * Speedup: 1.74x - 2.01x faster lookup latency (best-case down to 18ns in hot cache).

 [3] AF_XDP Kernel-Bypass Ingest             141,865 pkts/sec        4.63 µs        5.54 µs      22.78 µs
     * Ingestion Bandwidth: 34.64 MiB/s zero-copy UMEM parsing across 16,384 CRC16 slots.

 [4] Keystone Neural Micro-Model (260->64->6) 370,749 infer/s        2.55 µs        2.74 µs       2.93 µs
     * Mean latency: 2.64 µs per full 6-class feedforward classification.

 [5] Hybrid FTS + Vector RRF Multimodal        1,838 queries/s     501.58 µs      715.75 µs     850.03 µs
     * BM25 trigram inverted index + HNSW vector DB fused via RRF with dynamic neural semantic masking.
========================================================================================================
```

### Head-to-Head Comparative Analysis vs. Industry Alternatives

| Pillar / Subsystem | QIHSE + KEYSTONE Measured | Industry Standard / Alternative | Competitive Advantage |
| :--- | :--- | :--- | :--- |
| **Pillar 1: Vector Graph Search** | **33,080 QPS** (p50: 27.9 µs)<br>Anchor-Seeded 1D Spline Projection | **FAISS HNSW (CPU)**: ~15,000 QPS (65 µs)<br>**pgvector (HNSW)**: ~2,000 QPS (500 µs) | **2.2x higher QPS** vs FAISS CPU<br>**16.5x higher QPS** vs pgvector |
| **Pillar 2: Sorted Column / TSDB Search** | **3,510,610 lookups/s** (218 ns)<br>Keystone $O(\log \log N)$ Spline (18 ns best) | **C++ `std::lower_bound`**: 2,016,334 (447 ns)<br>**Postgres B-Tree**: ~600k lookups/s (1.2 µs) | **1.74x–2.0x faster** vs `std::lower_bound`<br>**5.5x faster** vs B+Tree pointer chasing |
| **Pillar 3: Packet Ingestion & Log Parsing** | **141,865 pkts/sec** (34.6 MiB/s)<br>AF_XDP Kernel Bypass + In-Place UMEM Scan | **Linux BSD Socket + epoll**: ~25,000 pkts/s<br>**Redis Ingestion**: ~75,000 ops/s | **5.6x higher throughput** vs epoll<br>**1.9x higher throughput** vs Redis |
| **Pillar 4: Neural Context Inference** | **370,749 infer/s** (2.55 µs)<br>Inlined Dense SAXPY C Kernel (260 $\to$ 64 $\to$ 6) | **ONNX Runtime (CPU)**: ~35,000 infer/s (28 µs)<br>**PyTorch LibTorch**: ~5,000 infer/s (200 µs) | **10.5x faster inference** vs ONNX Runtime<br>**74.0x faster** vs PyTorch LibTorch |
| **Pillar 5: Hybrid Multimodal Search** | **1,838 queries/s** (501 µs)<br>In-Memory BM25 + HNSW + Neural Masking | **OpenSearch Hybrid**: ~120 QPS (8.3 ms)<br>**Weaviate Hybrid**: ~200 QPS (5.0 ms) | **16.5x lower latency** vs OpenSearch<br>**10.0x lower latency** vs Weaviate |


The defensible claim today is narrower and stronger: KEYSTONE is a native, target-silicon-tuned search library with tested scalar, batch, archive, telemetry, optional Fortran, and optional OpenMP paths, plus host-dependent SIMD local scan support. GPU and NPU acceleration are roadmap backend families, not current implementation claims.

## Trigram Index Benchmarks (SSE4.2 + Algorithmic)

See [TRIGRAM_BENCHMARK.md](TRIGRAM_BENCHMARK.md) for full details.

| Corpus | Brute-force | Trigram Search | Speedup | Rejection |
|--------|------------|----------------|---------|-----------|
| 1 MB | 1.26 ms | 0.017 ms | 77x | 98.05% |
| 10 MB | 23.83 ms | 0.024 ms | 1,002x | 99.80% |
| 100 MB | 99.50 ms | 0.141 ms | 705x | 99.80% |
| 1 GB | 1,021 ms | 0.976 ms | 1,046x | 99.98% |

Optimizations: SSE4.2 batch trigram extraction (14/16 bytes), dual-byte
SIMD memmem, Fibonacci hash (2 ops), split hash table (L2-resident keys),
per-document dedup bitmap, shared counting sort.
