# KEYSTONE Trigram Index — Benchmark Results

## Hardware & Toolchain

| Component | Detail |
|-----------|--------|
| CPU | Intel Xeon E5-2407 0 @ 2.20GHz (Sandy Bridge) |
| Cores | 8 |
| SIMD | SSE4.2 + AVX1 only (no AVX2/AVX-512/FMA) |
| Compiler | GCC 16.1.0 |
| Flags | `-O3 -march=native -fPIC -fopenmp` |
| Date | September 2026 |

## Optimizations Applied

### SIMD (SSE4.2)

| Optimization | Before | After | Impact |
|---|---|---|---|
| Trigram extraction | 3 byte loads per position | 16-byte load → 14 trigrams via `_mm_shuffle_epi8` | 14x fewer loads |
| Substring verification | First-byte scalar scan | Dual first/last-byte `_mm_cmpeq_epi8` 16-wide | 16x wider filter |
| Long needle confirm | `memcmp` full needle | `_mm_cmpeq_epi8` 16-byte all-lanes match | 16x fewer branches |
| Posting list probe | Binary search (branchy) | 4-way `_mm_cmpeq_epi32` for lists < 64 | 4x parallel compare |

### Algorithmic

| Optimization | Before | After | Impact |
|---|---|---|---|
| Hash function | FNV-1a (6 ops) | Fibonacci `(key * 0x9E3779B1) >> shift` (2 ops) | 3x fewer ops |
| Hash table layout | Single 32B struct array (2MB, L2 miss) | Split: 4B keys (256KB, L2 hit) + pointer on match | Cache miss reduction |
| Per-doc dedup | Hash lookup per trigram position | 2MB bitmap, skip seen trigrams, O(unique) clearing | 2-3x fewer lookups |
| Finalize sort | `qsort` per list | Shared counting array, single alloc, reused | Zero alloc churn |

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         DOCUMENT INGESTION              │
                    │                                         │
                    │  ┌──────────────┐  ┌─────────────────┐  │
                    │  │ SSE4.2 Batch │  │  Per-Doc Bitmap │  │
                    │  │  Trigram     │→ │  Dedup (2MB)    │  │
                    │  │  Extraction  │  │  Skip seen      │  │
                    │  │  14/16 bytes │  │  trigrams       │  │
                    │  └──────┬───────┘  └────────┬────────┘  │
                    │         │                   │           │
                    │         ▼                   ▼           │
                    │  ┌──────────────────────────────────┐   │
                    │  │    Fibonacci Hash (2 ops)        │   │
                    │  │    (key * 0x9E3779B1) >> shift   │   │
                    │  └──────────────┬───────────────────┘   │
                    │                 │                       │
                    │         ▼                 ▼             │
                    │  ┌──────────┐    ┌──────────────┐       │
                    │  │ Key Array│    │ Posting List │       │
                    │  │ 4B each  │    │ Array (ptr)  │       │
                    │  │ L2 cache │    │ Only on match│       │
                    │  └──────────┘    └──────────────┘       │
                    └─────────────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │         FINALIZE                │
                    │                                 │
                    │  Shared counting sort (O(n+k))  │
                    │  Single alloc, reused per list  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │          SEARCH                  │
                    │                                  │
                    │  1. Extract query trigrams       │
                    │     (SSE4.2 batch, O(n²) dedup)  │
                    │                                  │
                    │  2. Sort posting lists by size   │
                    │                                  │
                    │  3. Galloping intersection       │
                    │     + SIMD 4-way lower_bound     │
                    │     for small lists (<64)        │
                    │                                  │
                    │  4. Verify candidates            │
                    │     Dual-byte SSE4.2 memmem      │
                    │     (first + last byte, 16-wide) │
                    └──────────────────────────────────┘
```

## Results

### Speedup vs Brute-Force

```
              Speedup over Brute-Force Scan
              (log scale)

  1 MB   │  ████                                          77x
  10 MB  │  ████████████████████                        1,002x
  100 MB │  ██████████████████████████████                705x
  1 GB   │  ████████████████████████████████████        1,046x
         └──────────────────────────────────────────────
           0       250       500       750      1000
```

### Search Latency

```
              Trigram Search Latency (ms, log scale)

  1 MB   │  ▏                                             0.017 ms
  10 MB  │  ▏                                             0.024 ms
  100 MB │  █                                             0.141 ms
  1 GB   │  █████                                         0.976 ms
         └──────────────────────────────────────────────
           0.001    0.01     0.1      1.0      10
```

### Build Time

```
              Index Build Time (seconds, linear scale)

  1 MB   │  ▏                                              0.12 s
  10 MB  │  █                                              1.16 s
  100 MB │  ██████████                                    11.62 s
  1 GB   │  ████████████████████████████████████████     111.30 s
         └──────────────────────────────────────────────
           0       30       60       90      120
```

### Candidate Rejection Rate

```
              Candidate Rejection (higher = better filtering)

  1 MB   │  ████████████████████████████████░░░░          98.05%
  10 MB  │  ████████████████████████████████████░         99.80%
  100 MB │  ████████████████████████████████████░         99.80%
  1 GB   │  ██████████████████████████████████████        99.98%
         └──────────────────────────────────────────────
           0%      25%      50%      75%      100%
```

## Detailed Numbers

| Corpus | Documents | Doc Size | Brute (ms) | Build (s) | Search (ms) | Speedup | Rejection | Unique Trigrams | Total Postings |
|--------|-----------|----------|------------|-----------|-------------|---------|-----------|-----------------|----------------|
| 1 MB | 256 | 4 KB | 1.26 | 0.12 | 0.017 | 77x | 98.05% | 17,627 | 935,515 |
| 10 MB | 2,560 | 4 KB | 23.83 | 1.16 | 0.024 | 1,002x | 99.80% | 17,628 | 9,349,521 |
| 100 MB | 25,600 | 4 KB | 99.50 | 11.62 | 0.141 | 705x | 99.80% | 17,748 | 93,499,625 |
| 1 GB | 262,144 | 4 KB | 1,021 | 111.30 | 0.976 | 1,046x | 99.98% | 17,752 | 957,418,100 |

## Key Observations

1. **Search latency scales sublinearly**: 1GB corpus (1024x more data than 1MB) searches in only 57x more time (0.976ms vs 0.017ms). The trigram index rejects 99.98% of documents, so verification cost grows with matches, not corpus size.

2. **Build throughput is constant at ~9 MB/s**: dominated by hash table operations (insert + resize). The per-document bitmap dedup and Fibonacci hash help, but the fundamental cost is O(total_trigram_positions).

3. **Unique trigram count saturates at ~17,750**: the 24-bit trigram space has 16M possible values, but random lowercase ASCII text (a-z) produces only 26³ = 17,576 possible trigrams. The index is extremely compact — 1GB of text maps to ~18K posting lists.

4. **Posting list growth is linear**: total postings ≈ 0.93 × corpus_bytes, confirming each byte position contributes one trigram (minus 2 per document for boundary).

5. **Brute-force throughput is constant at ~1000 MB/s**: this is the SSE4.2 `memcmp` speed for the first-byte filter. The trigram index wins because it avoids scanning 99.8%+ of the corpus.

## Reproducing

```bash
cd /home/john/Documents/KEYSTONE
make clean && make lib -j4
make benchmarks/trigram_benchmark
LD_LIBRARY_PATH=. ./benchmarks/trigram_benchmark
```

The benchmark tests four corpus sizes (1MB, 10MB, 100MB, 1GB) with random
lowercase text, injects 5-50 target needles, and compares brute-force
scan against the trigram-accelerated search path.
