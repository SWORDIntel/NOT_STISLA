# KEYSTONE Build Modes

The KEYSTONE search engine is designed to be easily consumed as a library without hiding its native build assumptions. It can be compiled in various configurations depending on your hardware, dependency constraints, and operational requirements.

## 1. Native Default Build (Recommended)
This is the standard build mode for KEYSTONE, leveraging autodetected CPU features like AVX2 and AVX-512 where appropriate, while keeping dependencies minimal.

```bash
make
```
**Features:**
- Compiles with `-march=native -O3` by default.
- AVX-512 experimental features are isolated and compiled if supported.
- `libarchive` and `libzstd` are autodetected via `pkg-config`. If present, the `.tar.zst` extraction paths are enabled automatically.
- **OpenMP is auto-enabled** if the compiler supports it (GCC always does). Set `KEYSTONE_ENABLE_OPENMP=0` to disable.
- **SSE4.2 SIMD path** is compiled when `-msse4.1` is active (via `-march=native` on SSE4.2+ CPUs). This provides 128-bit integer SIMD for AVX1-only CPUs (Sandy Bridge, Ivy Bridge) that lack AVX2's 256-bit integer ops. Runtime dispatch via `keystone_detect_cpu_features()` selects the best available path.

## 2. Dependency-Minimal (Scalar-Only) Build
If you are deploying KEYSTONE to embedded systems, legacy hardware without SIMD, or environments strictly forbidding vectorization, you can force a purely scalar (C fallback) build.

```bash
make KEYSTONE_FORCE_SCALAR=1
```
**Features:**
- Disables AVX2/AVX-512 explicitly via `-mno-avx2`.
- Avoids all SIMD includes.
- Still utilizes KEYSTONE's core anchor table and interpolation algorithms, guaranteeing sub-logarithmic latency even on scalar paths.

## 3. Archive-Enabled Build (Explicit)
To guarantee that telemetry processing and `.tar.zst` streaming features are compiled (and error out if dependencies are missing), explicitly request it:

```bash
make KEYSTONE_ENABLE_TAR_ZST=1
```
**Features:**
- Requires `libarchive`, `libzstd`, and POSIX threads (`-lpthread`).
- Enables `keystone_tar_zst.c` and `dsmil_telemetry_processor.c`.
- Activates persistent sidecar indexing (`.idx.json`), dual-threaded producer/consumer ring buffering (`enable_pipeline`), and multi-archive parallel batch pools.

## 4. OpenMP Build
OpenMP is now **auto-enabled by default** when the compiler supports it. You no longer need to explicitly request it.

```bash
make    # OpenMP auto-detected and enabled
```

To explicitly enable or disable:
```bash
make KEYSTONE_ENABLE_OPENMP=1   # force enable
make KEYSTONE_ENABLE_OPENMP=0   # force disable
```
**Features:**
- Adds `-fopenmp` to the compiler and linker flags.
- The `auto_backend` router evaluates multi-threaded batch dispatch for batches >= 4096 items (configurable via `KEYSTONE_AUTO_PARALLEL_MIN_ITEMS` at compile time).
- On 8-core Sandy Bridge Xeon: **2x faster batch search** (165 ns/query vs 330 ns/query serial).

## 5. Fortran Scientific Build
For workloads deeply integrated with scientific computing or requiring strict legacy Fortran batch processing pipelines:

```bash
make KEYSTONE_ENABLE_FORTRAN=1
```
**Features:**
- Requires `gfortran`.
- Compiles the Fortran interoperability layer (`libkeystone_batch.so`).
- Instructs the auto-backend router to measure the Fortran execution path during calibration.

## 6. Benchmark-Only Build
To compile just the benchmark suite (without the test suite):

```bash
make benchmarks
```
**Features:**
- Compiles `dsmil_benchmark` and `performance_proof`.
- Useful for validating target-silicon performance before deploying to production.

## 7. Packaging: Static vs Shared Library
KEYSTONE currently compiles as a collection of native object files (`.o`), allowing maximum inlining and Link-Time Optimization (LTO) when statically linked directly into your parent application.

**For a Static Library (`libkeystone.a`):**
```bash
ar rcs libkeystone.a src/*.o
```

**For a Shared Library (`libkeystone.so`):**
To build KEYSTONE as a dynamically linked object, you must append `-fPIC` to the `CFLAGS` during `make`, and link the resulting objects:
```bash
make CFLAGS="-O3 -march=native -fPIC"
gcc -shared -o libkeystone.so src/*.o
```
*(Note: Shared library boundaries may prevent cross-module SIMD inlining. Static linkage is heavily recommended for maximum search throughput.)*
