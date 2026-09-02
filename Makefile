# KEYSTONE Makefile
# Standard make / make test workflow

CC      := gcc
CFLAGS  := -O3 -march=native -fPIC -Wall -Wextra -Werror=implicit-function-declaration -I./include -DKEYSTONE_ENABLE_PLATFORM_TUNING
LDFLAGS := -lm

# Optional OpenMP (default: auto-enabled if the compiler supports it,
# since multi-core CPUs benefit from parallel batch search.  Set
# KEYSTONE_ENABLE_OPENMP=0 to disable.)
ifeq ($(KEYSTONE_ENABLE_OPENMP),1)
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
else ifneq ($(KEYSTONE_ENABLE_OPENMP),0)
    ifeq ($(shell echo | $(CC) -fopenmp -dM -E - 2>/dev/null | grep -q '_OPENMP' && echo yes),yes)
        CFLAGS  += -fopenmp
        LDFLAGS += -fopenmp
    endif
endif

# Optional tar.zst streaming support (default: enabled if libarchive + libzstd are available)
ifeq ($(KEYSTONE_ENABLE_TAR_ZST),1)
    TAR_ZST_CFLAGS := -DKEYSTONE_ENABLE_TAR_ZST
    TAR_ZST_LDFLAGS := -larchive -lzstd -lpthread
    CFLAGS  += $(TAR_ZST_CFLAGS)
    LDFLAGS += $(TAR_ZST_LDFLAGS)
else ifneq ($(KEYSTONE_ENABLE_TAR_ZST),0)
    ifeq ($(shell pkg-config --exists libarchive libzstd 2>/dev/null && echo yes),yes)
        KEYSTONE_ENABLE_TAR_ZST := 1
        TAR_ZST_CFLAGS := -DKEYSTONE_ENABLE_TAR_ZST
        TAR_ZST_LDFLAGS := $(shell pkg-config --libs libarchive libzstd) -lpthread
        CFLAGS  += $(TAR_ZST_CFLAGS)
        LDFLAGS += $(TAR_ZST_LDFLAGS)
    endif
endif

# Optional Fortran backend (default: enabled if gfortran is available)
ifeq ($(KEYSTONE_ENABLE_FORTRAN),1)
    ifeq ($(shell command -v gfortran >/dev/null 2>&1 && echo yes),yes)
        FORTRAN_CFLAGS := -DKEYSTONE_ENABLE_FORTRAN
        FORTRAN_LDFLAGS := -L./fortran -lkeystone_batch -Wl,-rpath,'$$ORIGIN/fortran'
        CFLAGS  += $(FORTRAN_CFLAGS)
        LDFLAGS += $(FORTRAN_LDFLAGS)
    else
        $(error KEYSTONE_ENABLE_FORTRAN=1 requires gfortran)
    endif
else ifneq ($(KEYSTONE_ENABLE_FORTRAN),0)
    ifeq ($(shell command -v gfortran >/dev/null 2>&1 && echo yes),yes)
        FORTRAN_CFLAGS := -DKEYSTONE_ENABLE_FORTRAN
        FORTRAN_LDFLAGS := -L./fortran -lkeystone_batch -Wl,-rpath,'$$ORIGIN/fortran'
        CFLAGS  += $(FORTRAN_CFLAGS)
        LDFLAGS += $(FORTRAN_LDFLAGS)
    endif
endif

# Optional CUDA backend
ifeq ($(KEYSTONE_ENABLE_CUDA),1)
    ifeq ($(shell command -v nvcc >/dev/null 2>&1 && echo yes),yes)
        CUDA_CFLAGS := -DKEYSTONE_ENABLE_CUDA
        CUDA_LDFLAGS := -L./cuda -lkeystone_cuda -Wl,-rpath,'$$ORIGIN/cuda'
        CFLAGS  += $(CUDA_CFLAGS)
        LDFLAGS += $(CUDA_LDFLAGS)
    else
        $(error KEYSTONE_ENABLE_CUDA=1 requires nvcc)
    endif
else ifneq ($(KEYSTONE_ENABLE_CUDA),0)
    ifeq ($(shell command -v nvcc >/dev/null 2>&1 && echo yes),yes)
        CUDA_CFLAGS := -DKEYSTONE_ENABLE_CUDA
        CUDA_LDFLAGS := -L./cuda -lkeystone_cuda -Wl,-rpath,'$$ORIGIN/cuda'
        CFLAGS  += $(CUDA_CFLAGS)
        LDFLAGS += $(CUDA_LDFLAGS)
    endif
endif

# Optional QIHSE Unified Wire Protocol Bridge
ifeq ($(KEYSTONE_ENABLE_QIHSE_BRIDGE),1)
    ifndef QIHSE_ROOT
        $(error KEYSTONE_ENABLE_QIHSE_BRIDGE=1 requires QIHSE_ROOT to be set (e.g. QIHSE_ROOT=/path/to/QIHSE))
    endif
    QIHSE_CFLAGS := -DKEYSTONE_ENABLE_QIHSE_BRIDGE -I"$(QIHSE_ROOT)/include"
    QIHSE_LDFLAGS := -L"$(QIHSE_ROOT)" -lqihse -Wl,-rpath,"$(QIHSE_ROOT)"
    CFLAGS += $(QIHSE_CFLAGS)
    LDFLAGS += $(QIHSE_LDFLAGS)
    SRC += src/qihse_keystone_bridge.c
else
    # Standalone stub compilation
    SRC += src/qihse_keystone_bridge.c
endif

# CPU ISA feature flags
HOST_CPU_FLAGS ?= $(shell awk -F: '/^flags/{sub(/^ /, "", $$2); print $$2; exit}' /proc/cpuinfo 2>/dev/null)
cpu_has = $(if $(filter $(1),$(HOST_CPU_FLAGS)),1,0)

KEYSTONE_ENABLE_AVX2   ?= $(call cpu_has,avx2)
KEYSTONE_ENABLE_AVX512 ?= $(if $(and $(filter avx512f,$(HOST_CPU_FLAGS)),$(filter avx512dq,$(HOST_CPU_FLAGS))),1,0)

# SIMD detection
ifeq ($(KEYSTONE_FORCE_SCALAR),1)
    CFLAGS += -mno-avx2 -mno-avx512f
else
    ifeq ($(KEYSTONE_ENABLE_AVX2),1)
        CFLAGS += -mavx2
    endif
    ifeq ($(KEYSTONE_ENABLE_AVX512),1)
        CFLAGS += -mavx512f -mavx512dq
    endif
endif

SRC     := src/keystone.c src/dsmil_keystone_wrapper.c src/dsmil_telemetry_processor.c \
           src/nst_prefetch_profile.c src/nst_platform_hints.c src/nst_memory_topology.c \
           src/nst_vector_config.c src/nst_batch_scheduler.c src/nst_cache_line_align.c \
           src/nst_branch_predict.c src/nst_dram_locality.c src/keystone_avx512.c
OBJS    := $(SRC:.c=.o)

TEST_SRC := tests/test_core_native.c tests/test_auto_backend.c \
            tests/test_fortran_backend.c tests/test_telemetry_processor_perf.c \
            tests/dsmil_integration_test.c tests/test_performance_fix.c
TEST_BIN := bin/test_enhanced bin/test_auto_backend bin/test_fortran_backend \
            bin/test_telemetry_processor_perf bin/test_performance_fix \
            bin/test_core_native

ifeq ($(KEYSTONE_ENABLE_TAR_ZST),1)
SRC     += src/keystone_tar_zst.c
TEST_SRC += tests/test_tar_zst.c
TEST_BIN += bin/test_tar_zst
endif

SRC     += src/dsmil_hash_indexer.c src/dsmil_dirty_parser.c src/dsmil_model_bridge.c src/dsmil_micro_model.c

OBJS    := $(SRC:.c=.o)

BENCH_SRC := benchmarks/dsmil_benchmark.c benchmarks/performance_proof.c
BENCH_BIN := benchmarks/dsmil_benchmark benchmarks/performance_proof

.PHONY: all lib tests test check run-tests benchmarks clean

all: lib tests benchmarks

lib: libkeystone.so

libkeystone.so: $(OBJS)
	$(CC) -shared -fPIC -o $@ $^ $(LDFLAGS)

tests: $(TEST_BIN)

test: check

check: tests
	@set -e; \
	for test_bin in $(TEST_BIN); do \
		echo "==> $$test_bin"; \
		./$$test_bin; \
	done

run-tests: check

benchmarks: $(BENCH_BIN)

bin:
	mkdir -p bin

# Pattern rules
# Explicit rule for AVX-512 object to isolate experimental code
src/keystone_avx512.o: src/keystone_avx512.c
	$(if $(filter 1,$(KEYSTONE_ENABLE_AVX512)),$(CC) $(CFLAGS) -mavx512f -mavx512dq -c $< -o $@,$(CC) $(CFLAGS) -mno-avx512f -c $< -o $@)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -c $< -o $@

benchmarks/%.o: benchmarks/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Test binaries
bin/test_enhanced: $(OBJS) tests/dsmil_integration_test.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_core_native: $(OBJS) tests/test_core_native.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_fortran_backend: $(OBJS) tests/test_fortran_backend.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_auto_backend: $(OBJS) tests/test_auto_backend.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_telemetry_processor_perf: $(OBJS) tests/test_telemetry_processor_perf.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_performance_fix: $(OBJS) tests/test_performance_fix.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

bin/test_tar_zst: $(OBJS) tests/test_tar_zst.o | bin
	$(CC) -o $@ $^ $(LDFLAGS)

# Benchmark binaries
benchmarks/dsmil_benchmark: $(OBJS) benchmarks/dsmil_benchmark.o benchmarks/benchmark_writer.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmarks/performance_proof: $(OBJS) benchmarks/performance_proof.o benchmarks/benchmark_writer.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Fortran backend (optional)
fortran/libkeystone_batch.so: fortran/keystone_batch.f90
	mkdir -p fortran
	gfortran -O3 -shared -fPIC -fopenmp -Jfortran $< -o $@

FORTRAN_ENABLED := no
ifeq ($(KEYSTONE_ENABLE_FORTRAN),1)
FORTRAN_ENABLED := yes
else ifneq ($(KEYSTONE_ENABLE_FORTRAN),0)
ifeq ($(shell command -v gfortran >/dev/null 2>&1 && echo yes),yes)
FORTRAN_ENABLED := yes
endif
endif

ifeq ($(FORTRAN_ENABLED),yes)
all: fortran/libkeystone_batch.so
libkeystone.so $(TEST_BIN) $(BENCH_BIN): fortran/libkeystone_batch.so | bin
endif

# CUDA backend (optional)
CUDA_HOME ?= $(firstword $(wildcard /usr/local/cuda /usr/local/cuda-13.3 /opt/cuda))
CUDA_INCLUDE ?= $(firstword $(wildcard $(CUDA_HOME)/targets/x86_64-linux/include $(CUDA_HOME)/include))
cuda/libkeystone_cuda.so: cuda/keystone_cuda.cu
	mkdir -p cuda
	nvcc -O3 --std=c++17 -U_GNU_SOURCE -I$(CUDA_INCLUDE) --compiler-options '-fPIC' -shared $< -o $@

CUDA_ENABLED := no
ifeq ($(KEYSTONE_ENABLE_CUDA),1)
CUDA_ENABLED := yes
else ifneq ($(KEYSTONE_ENABLE_CUDA),0)
ifeq ($(shell command -v nvcc >/dev/null 2>&1 && echo yes),yes)
CUDA_ENABLED := yes
endif
endif

ifeq ($(CUDA_ENABLED),yes)
all: cuda/libkeystone_cuda.so
libkeystone.so $(TEST_BIN) $(BENCH_BIN): cuda/libkeystone_cuda.so | bin
endif

clean:
	rm -f $(OBJS) tests/*.o benchmarks/*.o libkeystone.so
	rm -rf bin
	rm -f scripts/compare_search_auto
	rm -f fortran/libkeystone_batch.so fortran/*.mod
	rm -f cuda/libkeystone_cuda.so
