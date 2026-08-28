<p align="center">
  <img src="docs/Logo.png" alt="KEYSTONE logo" width="720">
</p>

### KEYSTONE helps large databases find the right record fast — using adaptive search, resident-hardware acceleration, and repeatable indexing logic.

[![C](https://img.shields.io/badge/C-11-blue.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Fortran](https://img.shields.io/badge/Fortran-90%2B-purple.svg)](https://en.wikipedia.org/wiki/Fortran)
[![Python](https://img.shields.io/badge/Python-3-yellow.svg)](https://www.python.org/)
[![SIMD](https://img.shields.io/badge/SIMD-SSE4.2%20%7C%20AVX2%20%7C%20AVX--512-black.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![Parallel](https://img.shields.io/badge/Parallel-OpenMP-green.svg)](https://www.openmp.org/)
[![Archives](https://img.shields.io/badge/Ingestion-tar.zst-orange.svg)](https://facebook.github.io/zstd/)
[![Platform](https://img.shields.io/badge/Platform-Linux-success.svg)](https://www.kernel.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-red.svg)](LICENSE)

**KEYSTONE** is a high-performance intelligence ingestion and database indexing engine. It is designed to operate as a standalone fast-search layer, or as the native pre-processing pipeline for the [QIHSE](https://github.com/SWORDIntel/QIHSE) database ecosystem.

It combines zero-allocation dirty log tokenization, anchor-guided interpolation search, SIMD-assisted local scans, and a native C neural network (micro-model) to pull structured semantic hits from chaotic post-compromise data.

</div>

---

## Current Status

KEYSTONE is a working native C library and benchmark suite, not just a design note.

| Area | Status |
|---|---|
| Core sorted `int64_t` lookup | Implemented and tested |
| Anchor-guided interpolation search | Implemented and tested |
| Batch lookup | Implemented and tested |
| Runtime backend calibration | Implemented and tested |
| Unstructured / Dirty Log Tokenizer | Implemented (zero-allocation email:pass extraction) |
| Heterogeneous Hash Indexer | Implemented (FNV-1a column projection) |
| Native Context Micro-Model | Implemented (6-class DNN with confidence gating) |
| OpenMP batch path | Auto-enabled by default when compiler supports it |
| Fortran batch backend | Optional; enabled when requested |
| `.tar.zst` archive search | Optional; enabled when `libarchive` and `libzstd` are available |
| SSE4.2 small-window scan | Implemented for native x86 builds with SSE4.2+ (AVX1-only CPUs) |
| AVX2 small-window scan | Implemented for native x86 builds with AVX2 |
| AVX-512 path | Build-gated and hardware-dependent |

The default workflow is intentionally native: build on the machine, container image, or silicon family where the code will run, then measure there.

---

## Quick Start

```bash
make clean
make test
```

`make test` builds and runs the available test binaries for the current host. Optional components are included when their local dependencies are present.

For a scalar-only comparison build:

```bash
make clean
KEYSTONE_ENABLE_FORTRAN=0 KEYSTONE_ENABLE_TAR_ZST=0 KEYSTONE_FORCE_SCALAR=1 make test
```

For build-only verification:

```bash
make tests
```

For benchmarks:

```bash
make benchmarks
./benchmarks/dsmil_benchmark
./benchmarks/performance_proof
```

---

## Mission

Modern databases rarely fail because storage is unavailable. They fail because lookup paths drift, records fragment, indexes become inconsistent, and high-volume datasets become expensive to search reliably.

KEYSTONE addresses that layer directly.

It is designed for systems where record identity, index stability, and lookup speed matter at the same time. Instead of treating indexing as a side effect of storage, KEYSTONE treats precise indexed retrieval as the core primitive.

---

## What KEYSTONE Is

KEYSTONE is a functional indexing and lookup engine for sorted `int64_t` datasets. It sits close to the data layer, supporting fast search, batch lookup, telemetry processing, archive-aware ingestion, and performance validation.

It is not a decorative wrapper around a database. It is a low-level search component designed for practical integration into larger systems that need predictable record access.

| Capability | Purpose |
|---|---|
| **Anchor-guided interpolation search** | Reduces search work by using learned anchor points across sorted data. |
| **Unstructured data ingestion** | A zero-allocation C tokenizer aggressively hunts for `email:pass`, URLs, and identifiers in raw, dirty stealer logs and memory dumps. |
| **Hash indexing** | Projects strings into 64-bit integer hashes via FNV-1a, feeding them directly into KEYSTONE's fast scalar search. |
| **Native micro-model** | Extracts 256 bytes of context around a hit and classifies the target (e.g., FINANCIAL, CORPORATE) using a compiled 6-class neural network. |
| **Batch lookup** | Handles high-volume query workloads efficiently. |
| **Runtime backend calibration** | Measures viable local backends on first use, caches the fastest decision, and reports the selected path. |
| **SIMD-assisted search windows** | Uses compiled AVX2 and build-gated AVX-512 scan paths where available. |
| **OpenMP parallelism** | Scales batch workloads across CPU threads when built with OpenMP. |
| **Fortran batch backend** | Provides an optional high-throughput backend for numerical batch processing. |
| **Benchmark suite** | Measures latency, throughput, and backend behavior across dataset profiles. |

---

## Why It Matters

High-speed lookup is easy to claim and difficult to prove. Real systems need more than a fast happy path.

KEYSTONE is built around the problems that appear when datasets become large, query volume increases, and records must remain stable across repeated processing runs.

It is useful where the cost of bad indexing is operationally significant:

- stale or unstable lookup positions;
- slow batch search over large sorted datasets;
- duplicated lookup logic across services;
- poor visibility into backend performance;
- inconsistent ingestion from compressed archives;
- weak testability around search behavior;
- performance claims without reproducible benchmarks.

KEYSTONE provides a focused answer: a compact, benchmarkable, database-adjacent indexing engine with multiple execution backends and a clear performance model.

---

## Architecture

```mermaid
flowchart TB
    subgraph Intelligence["Post-Compromise Intelligence Pipeline"]
        DIRTY["Dirty Log / Memory Dump"] --> PARSE["Custom C Tokenizer"]
        PARSE -->|Extracts email:pass| HASH["Hash Indexer (FNV-1a)"]
        HASH --> AT["Anchor Table"]
        AT --> R["Result Offset"]
        R --> BRIDGE["Model Context Bridge"]
        BRIDGE -->|256-byte window| MODEL["Native Micro-Model"]
        MODEL --> CLASS["6-Class Semantic Triage"]
    end

    subgraph Search["Numeric Search Pipeline"]
        Q["Query Key / Query Batch"] --> Auto{"Runtime Backend Calibrator"}
        Auto -->|Single / Small| S["Scalar Anchor Search"]
        Auto -->|Sorted Batch| CB["Optimized C Batch"]
        Auto -->|Large Batch + OpenMP| MP["C OpenMP Batch"]
        Auto -->|Dense Batch| FT["Optional Fortran Batch"]
        S --> SIMD["SIMD Scan (AVX2 / AVX-512)"]
        S --> AT
        CB --> AT
        SIMD --> AT
        MP --> AT
        FT --> AT
    end
```

---

## Backend Selection Model

```mermaid
flowchart TD
    Start["Incoming Lookup Workload"] --> Mode{"Single or Batch?"}
    Mode -->|Single| Scalar["Scalar Interpolation Search"]
    Mode -->|Batch| Size{"Dataset and Batch Size"}

    Size -->|Small / Low Overhead Preferred| Scalar
    Size -->|Sorted Batch| CB["Optimized C Merge-Walk Batch"]
    Size -->|Large / Repeated Queries| CPU{"Build + Workload Capabilities"}

    CPU -->|OpenMP Built + Enough Queries| OMP["C OpenMP Batch Execution"]
    CPU -->|Fortran Built + Dense Sorted Shape| FORTRAN["Fortran Batch Execution"]
    CPU -->|Otherwise| CB

    OMP --> Cal["Measured Local Calibration"]
    FORTRAN --> Cal
    CB --> Cal
    Scalar --> Anchor["Anchor Table / Adaptive Learning"]
    Cal --> Anchor
    Anchor --> Result["Stable Result Index"]
```

---

## Data Flow

```mermaid
flowchart LR
    Source["Source Dataset"] --> Normalize["Sorted int64_t Keyspace"]
    Normalize --> Anchor["Anchor-Guided Search Layer"]
    Anchor --> Backend["Selected Execution Backend"]
    Backend --> Index["Result Index"]
    Index --> Consumer["Database / Telemetry / Analysis Consumer"]

    Archive["Compressed Archive"] --> Member["Member Offset Index"]
    Member --> Anchor

    Bench["Benchmark Harness"] --> Backend
    Tests["Validation Suite"] --> Anchor
```

---

## System Profile

| Layer | Function |
|---|---|
| **Core search engine** | Performs anchor-guided interpolation search over sorted integer data. |
| **Dirty Log Tokenizer** | Aggressively rips identifiers (`email:pass`) from unstructured dumps. |
| **Hash Indexer** | Projects heterogeneous strings into 64-bit space for rapid search. |
| **Context Bridge** | Slices perfectly shaped context windows from raw data bypassing full decompression. |
| **Micro-Model Inference** | A compiled DNN executing focal-loss trained classification on context data. |
| **Adaptive backend layer** | Routes workloads across scalar, optimized C, OpenMP, and Fortran paths. |
| **Anchor table** | Maintains learned search anchors to improve repeated lookup behavior. |
| **Archive interface** | Enables `.tar.zst` workflows without treating archive handling as an afterthought. |

---

## Feature Matrix

| Feature | Scalar / Anchor C | Optimized C Batch | SIMD Local Scan | OpenMP Batch | Fortran Batch | `.tar.zst` |
|---|---:|---:|---:|---:|---:|---:|
| Single search | Yes | No | Yes, inside local windows | No | No | No |
| Batch search | Yes | Yes | Indirect | Optional | Optional | No |
| Auto backend calibration | Yes | Yes | Build/runtime detected | Optional measured candidate | Optional measured candidate | No |
| Decision provenance | Fast path / measured / cached / fallback | Measured or cached | Build/runtime detected | Measured or cached | Measured or cached | No |
| Anchor learning | Yes | No for merge-walk batch | Yes through scalar path | Per-thread clone path | No | No |
| Runtime tuning | Yes | Yes | Build/runtime gated | Build gated | Build gated | No |
| Archive ingestion | No | No | No | No | No | Yes |
| Member offset indexing | No | No | No | No | No | Yes |
| Benchmark validation | Yes | Yes | Partial / host-specific | Yes when built | Yes when built | Yes |
| Linux support | Yes | Yes | Host-dependent | Compiler/runtime-dependent | Toolchain-dependent | Dependency-dependent |

---

## Practical Use Cases

### Database Index Acceleration

KEYSTONE is suitable for systems that need fast lookup across large sorted integer keyspaces, especially where keys map into record offsets, entity identifiers, telemetry IDs, event IDs, or compressed member indexes.

### Canonical Record Retrieval

When records are normalized into stable integer identifiers, KEYSTONE can act as the lookup layer that keeps retrieval fast and repeatable.

### Telemetry and Event Search

Telemetry systems often generate large ordered datasets that must be searched repeatedly. KEYSTONE is designed for that access pattern: high-volume lookup, repeatable index resolution, and measurable backend performance.

### Archive-Aware Data Processing

Compressed archive workflows are common in telemetry, exports, backups, and evidence packages. KEYSTONE includes `.tar.zst` ingestion support so archive processing can remain close to the indexed lookup model.

### Benchmark-Driven Optimization

The project includes performance tooling intended to compare backend behavior across dense, large, and jittered dataset profiles. This makes it suitable as both functional software and a performance engineering showcase.

### Systems Programming Demonstration

KEYSTONE demonstrates practical low-level engineering across C11, optional Fortran, Python-based benchmark visualization, SIMD-aware execution, optional OpenMP parallelism, compressed archive ingestion, and structured validation.

### QIHSE Unified Wire Protocol Bridge

KEYSTONE can act as a native ingestion head-node for [QIHSE](https://github.com/SWORDIntel/QIHSE). When compiled with the bridge enabled, KEYSTONE parses dirty archives, categorizes the extracted intelligence using the micro-model, and streams the structured hits directly into QIHSE's Key-Value or Vector databases via UWP memory layout.

To build KEYSTONE as a QIHSE ingestor:

```bash
make clean
KEYSTONE_ENABLE_QIHSE_BRIDGE=1 QIHSE_ROOT=/path/to/QIHSE make
```

---

## Performance Orientation

KEYSTONE is designed around a simple premise: search performance should be measurable, backend-aware, and reproducible.

The benchmark suite is intended to produce concrete latency and speedup comparisons across workload profiles, including dense datasets, larger million-scale datasets, jittered distributions, and backend-specific behavior.

| Performance Concern | KEYSTONE Response |
|---|---|
| **Small lookup overhead** | Scalar interpolation remains available where vector or parallel overhead would be wasteful. |
| **Large batch volume** | Optimized C batch, OpenMP, and optional Fortran paths provide acceleration options for heavier workloads. |
| **Resident hardware variability** | Runtime feature detection plus first-use timing calibration supports backend selection based on the local CPU today and leaves room for measured GPU/NPU backends later. |
| **Repeated lookup behavior** | Anchor learning helps reduce repeated search cost across sorted keyspaces. |
| **Benchmark credibility** | Dedicated benchmark outputs support reviewable performance claims. |

### Benchmark Posture

Performance numbers are meaningful only with the host silicon, compiler flags, feature toggles, dataset shape, query hit rate, warmup policy, and transfer costs attached. Treat checked-in benchmark reports as examples of measurement style, not universal guarantees.

For serious comparison, run the benchmark matrix on the target machine and keep the generated CSV/output with the exact build command.

---

## Repository Structure

```mermaid
flowchart LR
    Root["KEYSTONE Root"] --> SRC["src/"]
    Root --> INC["include/"]
    Root --> TEST["tests/"]
    Root --> BENCH["benchmarks/"]
    Root --> DOCS["docs/"]
    Root --> FORT["fortran/"]
    Root --> SCRIPTS["scripts/"]

    SRC --> Core["Core Search Engine"]
    SRC --> Wrapper["Integration Wrapper"]
    SRC --> Telemetry["Telemetry Processor"]
    SRC --> Archive["tar.zst Interface"]

    INC --> PublicAPI["Public Headers"]
    INC --> TelemetryAPI["Telemetry Headers"]
    INC --> WrapperAPI["Wrapper Headers"]

    TEST --> NativeTest["Native Core Tests"]
    TEST --> BackendTest["Backend Selection Tests"]
    TEST --> ArchiveTest["Archive Tests"]
    TEST --> PerfTest["Performance Tests"]

    BENCH --> Harness["Benchmark Harness"]
    BENCH --> Proof["Performance Proofing"]
    BENCH --> Charts["Visualization Tools"]

    FORT --> FBackend["Fortran Batch Backend"]
```

---



---

## Intended Users

KEYSTONE is intended for technical users who care about lookup correctness, runtime behavior, and database-adjacent performance.

| User Type | Why It Fits |
|---|---|
| **Database engineers** | Useful for fast lookup layers and stable integer keyspaces. |
| **Systems programmers** | Demonstrates C11, SIMD, OpenMP, Fortran integration, and archive handling. |
| **Security researchers** | Useful for telemetry, indicator stores, evidence indexes, and high-volume lookup datasets. |
| **Data engineers** | Supports repeatable indexing before downstream processing or enrichment. |
| **Performance engineers** | Provides benchmarkable backend behavior across workload shapes. |
| **Research teams** | Works as a compact foundation for indexed retrieval experiments. |

---

## Quality Model

| Goal | Standard |
|---|---|
| **Correctness** | Lookup behavior should remain predictable across supported backends. |
| **Repeatability** | The same sorted dataset and key should resolve consistently. |
| **Performance visibility** | Backend performance should be measurable rather than assumed. |
| **Backend flexibility** | Scalar, SIMD, parallel, and Fortran paths should serve different workload profiles. |
| **Archive practicality** | Compressed ingestion should integrate with the lookup model rather than exist as a detached helper. |
| **Operational usefulness** | The system should be practical for real database, telemetry, and analysis workflows. |

---

## Requirements

| Area | Requirement |
|---|---|
| **Operating system** | Linux |
| **Primary compiler** | GCC or Clang with C11 support |
| **Optional numerical backend** | Fortran 90+ capable compiler |
| **Optional archive support** | `libarchive` and `libzstd` |
| **Optional build detection** | `pkg-config` |
| **Benchmark visualization** | Python 3 with numerical and plotting support |
| **Parallel acceleration** | OpenMP-capable compiler/runtime (auto-enabled by default) |
| **Vector acceleration** | SSE4.2, AVX2, or AVX-512 capable CPU where available |
| **Future accelerator backends** | GPU or NPU runtime/toolchain only after explicit backend implementation and measurement |

---

## Native Tuning Model

KEYSTONE is intentionally built as a native, silicon-tuned component. The
default Makefile uses `-O3 -march=native` and enables resident CPU paths such as
SSE4.2, AVX2, optional AVX-512, OpenMP (auto-enabled), Fortran, and `.tar.zst`
support when the local toolchain and libraries allow it. CPU execution is the
current implemented surface; GPU and NPU execution are future backend families
that must earn their place through explicit data-movement-aware benchmarks.

That means the preferred deployment model is to build KEYSTONE on the machine,
container image, or target silicon family where it will run. It is not trying to
produce one lowest-common-denominator binary for every host. For reproducible
comparisons, pin the build flags and optional feature switches explicitly:

```bash
make clean
KEYSTONE_ENABLE_FORTRAN=0 KEYSTONE_ENABLE_TAR_ZST=0 KEYSTONE_FORCE_SCALAR=1 make test
```

For the normal native path:

```bash
make test
```

To build without executing tests:

```bash
make tests
```

This native posture is deliberate. KEYSTONE favors accurate local dispatch and reproducible local measurement over a single portable binary that hides the silicon-specific behavior.

---

## Security and Data Handling

KEYSTONE is designed for datasets that may be operationally sensitive. Treat inputs, benchmark outputs, generated indexes, and archive contents according to the sensitivity of the source material.

Recommended handling posture:

- do not commit private datasets or generated operational indexes;
- keep production database credentials outside the repository;
- validate archive contents before processing untrusted inputs;
- separate benchmark datasets from real operational data;
- preserve audit trails where indexed records support legal, investigative, or high-trust decisions;
- treat indexing errors as data integrity failures, not cosmetic defects.

---

## What KEYSTONE Is Not

KEYSTONE is not a general spreadsheet sorter, dashboard framework, ORM, database replacement, or broad ETL platform.

It is a focused indexing and lookup component for sorted integer datasets. Its strength is precision, backend-aware performance, and suitability for integration into larger database and telemetry systems.

---



---

## Python SDK Quickstart

KEYSTONE provides a high-performance Python SDK (`pip install -e python`):

```python
import keystone
import numpy as np

# 1. Single & Auto-Calibrated Batch Interpolation Search
data = np.arange(0, 1000000, 7, dtype=np.int64)
indices = keystone.KeystoneSearch.search_batch(data, [70, 7000, 140000])

# Inspect backend provenance (scalar, OpenMP, AVX2, Fortran)
decision = keystone.KeystoneSearch.get_last_decision()
print(f"Active Backend: {decision.backend} (p95: {decision.p95_ns_per_key:.1f} ns/key)")

# 2. DSMIL Telemetry Processor with Keystone Acceleration
with keystone.TelemetryProcessor(max_events=100000) as tp:
    tp.add_event(keystone.TelemetryEvent(timestamp=1600000000, event_type=1, device_id=42, layer_id=2))
    ev = tp.find_by_timestamp(1600000000)

# 3. Cluster Slot Router (16,384 CRC16 Slots)
slot = keystone.ClusterRouter.get_slot("device:alpha:42")
print(f"Assigned Cluster Slot: {slot}")

# 4. Neural Micro-Model Classification (260->64->6 Feedforward)
clf = keystone.NeuralClassifier()
cls, name, conf = clf.classify("auth_failure admin@pentagon.af.mil topsecret_token=998822")
print(f"Classification: {name} ({conf*100:.1f}%)")
```

---

## Joint QIHSE + KEYSTONE 5-Pillar Architecture Benchmarks

Measured on host hardware (Intel Xeon E5-2407, AVX execution mode):

| Pillar / Subsystem | QIHSE + KEYSTONE Measured | Industry Standard / Alternative | Competitive Advantage |
| :--- | :--- | :--- | :--- |
| **[1] Vector Graph Search** | **33,080 QPS** (p50: 27.9 µs)<br>Anchor-Seeded 1D Spline Projection | **FAISS HNSW (CPU)**: ~15,000 QPS (65 µs)<br>**pgvector (HNSW)**: ~2,000 QPS (500 µs) | **2.2x higher QPS** vs FAISS CPU<br>**16.5x higher QPS** vs pgvector |
| **[2] Sorted Column / TSDB Search** | **3,510,610 lookups/s** (218 ns)<br>Keystone $O(\log \log N)$ Spline (18 ns best) | **C++ `std::lower_bound`**: 2,016,334 (447 ns)<br>**Postgres B-Tree**: ~600k lookups/s (1.2 µs) | **1.74x–2.0x faster** vs `std::lower_bound`<br>**5.5x faster** vs B+Tree pointer chasing |
| **[3] Packet Ingest / Log Scan** | **141,865 pkts/sec** (34.6 MiB/s)<br>AF_XDP Kernel Bypass + In-Place UMEM Scan | **Linux BSD Socket + epoll**: ~25,000 pkts/s<br>**Redis Ingestion**: ~75,000 ops/s | **5.6x higher throughput** vs epoll<br>**1.9x higher throughput** vs Redis |
| **[4] Neural Context Inference** | **370,749 infer/s** (2.55 µs)<br>Inlined Dense SAXPY C Kernel (260 $\to$ 64 $\to$ 6) | **ONNX Runtime (CPU)**: ~35,000 infer/s (28 µs)<br>**PyTorch LibTorch**: ~5,000 infer/s (200 µs) | **10.5x faster inference** vs ONNX Runtime<br>**74.0x faster** vs PyTorch LibTorch |
| **[5] Hybrid Multimodal Search** | **1,838 queries/s** (501 µs)<br>In-Memory BM25 + HNSW + Neural Masking | **OpenSearch Hybrid**: ~120 QPS (8.3 ms)<br>**Weaviate Hybrid**: ~200 QPS (5.0 ms) | **16.5x lower latency** vs OpenSearch<br>**10.0x lower latency** vs Weaviate |

> 📊 **Full Benchmark Details:** See [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md).

---

## License

**AGPL-3.0-or-later. This is strong copyleft. See [LICENSE](LICENSE) before use, modification, redistribution, hosting, or derivative work.**

Network use, redistribution, modification, and derivative use carry obligations. Do not treat this repository as permissive code.

If your intended use is proprietary, closed-source, commercial, hosted, or otherwise incompatible with AGPL compliance, obtain written permission or a separate license from the repository owner first.

KEYSTONE does not include covert license telemetry or phone-home enforcement. Compliance is enforced through the AGPL license terms, copyright ownership, visible notices, and separate commercial licensing where appropriate.

Unlicensed use outside the terms of the repository license is not authorized. Respect the license, attribute properly, and do not repackage the work as your own.

For commercial or proprietary licensing discussions, contact the repository owner.

---

<div align="center">

**KEYSTONE**  
Precision indexing. Adaptive lookup. Database discipline.

</div>
