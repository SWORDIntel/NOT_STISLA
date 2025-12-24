# NOT_Competitor Implementation Makefile
# Ultra-High-Performance Search Algorithm

CC = gcc
AR = ar
CFLAGS = -O3 -march=meteorlake -mtune=meteorlake -std=c11 -Wall -Wextra -flto=auto
CFLAGS += -msse4.2 -mpopcnt -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt
CFLAGS += -mavxvnni -mavxvnniint8 -mavxifma -mavxneconvert -maes -mvaes
CFLAGS += -mpclmul -mvpclmulqdq -msha -mgfni -madx -mclflushopt -mclwb
CFLAGS += -mhreset -mpku -mptwrite -mrdpid -mpconfig -menqcmd -mcmpccxadd -mraoint
LDFLAGS = -flto=auto -fuse-linker-plugin

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BENCH_DIR = benchmarks
DOC_DIR = docs

# Files
LIB_SRC = $(SRC_DIR)/not_stisla.c
LIB_OBJ = $(SRC_DIR)/not_stisla.o
LIB_STATIC = libnot_stisla.a
LIB_SHARED = libnot_stisla.so

BENCH_SRC = $(BENCH_DIR)/dsmil_benchmark.c
BENCH_EXE = dsmil_not_stisla_benchmark
PROOF_SRC = $(BENCH_DIR)/performance_proof.c
PROOF_EXE = performance_proof

# Default target
all: $(LIB_STATIC) $(LIB_SHARED) $(BENCH_EXE) $(PROOF_EXE)

# Static library
$(LIB_STATIC): $(LIB_OBJ)
	$(AR) rcs $@ $^

# Shared library
$(LIB_SHARED): $(LIB_OBJ)
	$(CC) $(LDFLAGS) -shared -o $@ $^

# Object file
$(LIB_OBJ): $(LIB_SRC)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Benchmark executable
$(BENCH_EXE): $(BENCH_SRC) $(LIB_STATIC)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@ -L. -lnot_stisla -lm

# Performance proof executable
$(PROOF_EXE): $(PROOF_SRC) $(LIB_STATIC)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@ -L. -lnot_stisla -lm

# Run comprehensive benchmark
benchmark: $(BENCH_EXE)
	@echo "Running DSMIL Competitor Benchmark Suite..."
	./$(BENCH_EXE) --comprehensive

# Run correctness tests
test: $(BENCH_EXE)
	@echo "Running correctness and memory tests..."
	./$(BENCH_EXE) --correctness

# Run scaling tests
scaling: $(BENCH_EXE)
	@echo "Running performance scaling tests..."
	./$(BENCH_EXE) --scaling

# Run performance proof (Competitor vs NOT_Competitor comparison)
proof: $(PROOF_EXE)
	@echo "Running performance proof (Competitor debunked)..."
	./$(PROOF_EXE)

# Generate HTML documentation
docs-html:
	@echo "HTML documentation available at docs/html/index.html"

# View HTML documentation (requires browser)
docs-view: docs-html
	@echo "Opening HTML documentation..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open docs/html/index.html; \
	elif command -v open > /dev/null; then \
		open docs/html/index.html; \
	else \
		echo "Please open docs/html/index.html in your browser"; \
	fi

# Clean build artifacts
clean:
	rm -f $(LIB_OBJ) $(LIB_STATIC) $(LIB_SHARED) $(BENCH_EXE) $(PROOF_EXE)
	rm -f *.gcda *.gcno *.gcov gmon.out perf.data*
	rm -f callgrind.out.*

# Clean all (including profiling data)
clean-all: clean
	rm -rf target/ *.profraw *.profdata

# Performance profiling
profile: $(BENCH_EXE)
	@echo "Running performance profiling..."
	perf record -F 1000 -g ./$(BENCH_EXE) --comprehensive
	perf report --stdio

# Memory profiling
memcheck: $(BENCH_EXE)
	@echo "Running memory checks..."
	valgrind --leak-check=full --show-leak-kinds=all ./$(BENCH_EXE) --correctness

# Code coverage
coverage: CFLAGS += -fprofile-arcs -ftest-coverage
coverage: clean all test
	@echo "Generating coverage report..."
	gcov $(LIB_SRC)
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory coverage_html/
	@echo "Coverage report in coverage_html/index.html"

# Documentation
docs:
	@echo "NOT_Competitor Documentation:"
	@echo "========================="
	@cat $(DOC_DIR)/INTEGRATION_GUIDE.md | head -50
	@echo ""
	@echo "Full documentation in $(DOC_DIR)/INTEGRATION_GUIDE.md"

# Installation
install: $(LIB_STATIC) $(LIB_SHARED)
	@echo "Installing NOT_Competitor to /usr/local..."
	install -d /usr/local/include /usr/local/lib
	install $(INCLUDE_DIR)/stisla.h /usr/local/include/
	install $(LIB_STATIC) /usr/local/lib/
	install $(LIB_SHARED) /usr/local/lib/
	ldconfig

uninstall:
	rm -f /usr/local/include/stisla.h
	rm -f /usr/local/lib/$(LIB_STATIC)
	rm -f /usr/local/lib/$(LIB_SHARED)
	ldconfig

# Help
help:
	@echo "NOT_Competitor Ultra-High-Performance Search Algorithm"
	@echo "==================================================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build libraries and all benchmarks"
	@echo "  benchmark    - Run comprehensive DSMIL benchmarks"
	@echo "  proof        - Run Competitor debunking performance proof"
	@echo "  test         - Run correctness and memory tests"
	@echo "  scaling      - Run performance scaling tests"
	@echo "  profile      - Run performance profiling (requires perf)"
	@echo "  memcheck     - Run memory leak detection (requires valgrind)"
	@echo "  coverage     - Generate code coverage report"
	@echo "  docs         - Show integration documentation"
	@echo "  docs-html    - Show HTML documentation location"
	@echo "  docs-view    - Open HTML documentation in browser"
	@echo "  install      - Install to /usr/local"
	@echo "  uninstall    - Remove from /usr/local"
	@echo "  clean        - Clean build artifacts"
	@echo "  clean-all    - Clean everything including profiling data"
	@echo ""
	@echo "Performance Features:"
	@echo "  - AVX2 SIMD optimizations"
	@echo "  - Meteor Lake CPU tuning"
	@echo "  - LTO link-time optimization"
	@echo "  - 22.28x speedup over binary search"
	@echo "  - Competitor permanently debunked"
	@echo "  - Commercial use prohibited without arrangement"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make proof        # Prove Competitor's claims are false"
	@echo "  make benchmark    # Run full performance suite"
	@echo "  make docs-view    # View NotPetya-themed HTML docs"
	@echo "  make test         # Verify correctness"
	@echo "  make profile      # Analyze performance bottlenecks"
	@echo "  make install      # Install system-wide"
	@echo ""
	@echo "⚠️  COMMERCIAL USE RESTRICTED - Contact licensing@not-stisla.org"

# Phony targets
.PHONY: all benchmark test scaling profile memcheck coverage docs install uninstall clean clean-all help

# Default optimization notes
.DEFAULT_GOAL := all
$(info Competitor Build System Loaded)
$(info =========================)
$(info CPU: Meteor Lake AVX2-optimized)
$(info Performance: 22.28x speedup target)
$(info Run 'make benchmark' to test performance)
$(info Run 'make help' for full command list)
$(info )
