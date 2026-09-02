#!/usr/bin/env bash
#
# build_vector_engine.sh - Build the KEYSTONE Vector Engine
#
# Auto-detects available SIMD and compiles only what's supported.
# The scalar kernel is ALWAYS compiled — it's the fallback floor.
# CUDA is built as a separate .so if nvcc is available.
# VPU support is compiled in if KEYSTONE_HAVE_VPU=1 is set.
#
# Usage:
#   ./build_vector_engine.sh                    # auto-detect everything
#   KEYSTONE_NO_CUDA=1 ./build_vector_engine.sh # skip CUDA
#   KEYSTONE_HAVE_VPU=1 ./build_vector_engine.sh # enable VPU kernel
#   KEYSTONE_ENABLE_AVX512=1 ./build_vector_engine.sh # force AVX-512
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CC=${CC:-cc}
CFLAGS_BASE="${CFLAGS_BASE:--O2 -fPIC -std=c11 -Wall -Wextra}"
OBJS=""
HAVE_CUDA=0
HAVE_VPU=0

# Check if VPU support is requested
if [ "${KEYSTONE_HAVE_VPU:-0}" = "1" ]; then
    HAVE_VPU=1
fi

# Helper: test if compiler accepts a flag
try_compile_flag() {
    printf 'int main(void){return 0;}\n' > /tmp/ks_vec_test.c
    $CC $CFLAGS_BASE "$1" -c /tmp/ks_vec_test.c -o /tmp/ks_vec_test.o 2>/dev/null
}

echo "=== KEYSTONE Vector Engine Build ==="
echo "Compiler: $CC"
echo "Base flags: $CFLAGS_BASE"

# ---- Level 8: SCALAR — ALWAYS COMPILED, NO CONDITIONS ----
echo "[1/8] Compiling scalar kernel (always)..."
$CC $CFLAGS_BASE -c kernels_scalar.c -o kernels_scalar.o
OBJS="$OBJS kernels_scalar.o"

# ---- Level 5: SSE4.2 ----
if try_compile_flag -msse4.2; then
    echo "[2/8] Compiling SSE4.2 kernel..."
    $CC $CFLAGS_BASE -msse4.2 -c kernels_sse42.c -o kernels_sse42.o
    OBJS="$OBJS kernels_sse42.o"
else
    echo "[2/8] SSE4.2 not available — skipping"
fi

# ---- Level 4: AVX (float-only, no AVX2 needed) ----
if try_compile_flag -mavx; then
    echo "[3/8] Compiling AVX kernel..."
    $CC $CFLAGS_BASE -mavx -c kernels_avx.c -o kernels_avx.o
    OBJS="$OBJS kernels_avx.o"
else
    echo "[3/8] AVX not available — skipping"
fi

# ---- Level 3: AVX2 + FMA ----
if try_compile_flag -mavx2; then
    echo "[4/8] Compiling AVX2 kernel..."
    $CC $CFLAGS_BASE -mavx2 -mfma -c kernels_avx2.c -o kernels_avx2.o
    OBJS="$OBJS kernels_avx2.o"
else
    echo "[4/8] AVX2 not available — skipping"
fi

# ---- Level 2: AVX-512 ----
if [ "${KEYSTONE_ENABLE_AVX512:-0}" = "1" ] && try_compile_flag -mavx512f; then
    echo "[5/8] Compiling AVX-512 kernel..."
    $CC $CFLAGS_BASE -mavx512f -mavx512dq -c kernels_avx512.c -o kernels_avx512.o
    OBJS="$OBJS kernels_avx512.o"
else
    echo "[5/8] AVX-512 not available or disabled — skipping"
fi

# ---- Level 6: NEON (ARM only) ----
if [ "$(uname -m)" = "aarch64" ] || try_compile_flag -mfpu=neon; then
    echo "[6/8] Compiling NEON kernel..."
    $CC $CFLAGS_BASE -c kernels_neon.c -o kernels_neon.o 2>/dev/null && OBJS="$OBJS kernels_neon.o" || echo "  NEON compile failed — skipping"
else
    echo "[6/8] NEON not available (not ARM) — skipping"
fi

# ---- Level 7: VPU (Myriad X, guarded) ----
if [ "$HAVE_VPU" = "1" ]; then
    echo "[7/8] Compiling VPU kernel (KEYSTONE_HAVE_VPU=1)..."
    $CC $CFLAGS_BASE -DKEYSTONE_HAVE_VPU=1 -c kernels_vpu.c -o kernels_vpu.o
    VPU_FLAG="-DKEYSTONE_HAVE_VPU=1"
    OBJS="$OBJS kernels_vpu.o"
else
    echo "[7/8] VPU support not requested (set KEYSTONE_HAVE_VPU=1 to enable)"
    VPU_FLAG=""
fi

# ---- Core files (compiled with NO ISA flags) ----
echo "[8/8] Compiling core engine files..."
$CC $CFLAGS_BASE $VPU_FLAG -c lsh.c -o lsh.o
$CC $CFLAGS_BASE $VPU_FLAG -c keystone_engine.c -o keystone_engine.o
$CC $CFLAGS_BASE $VPU_FLAG -c persist.c -o persist.o
OBJS="$OBJS lsh.o keystone_engine.o persist.o"

# ---- OpenMP (optional) ----
OMP_LIBS=""
OMP_FLAGS=""
if try_compile_flag -fopenmp; then
    OMP_LIBS="-fopenmp"
    OMP_FLAGS="-fopenmp"
    echo "  OpenMP: enabled (-fopenmp)"
elif pkg-config --exists omp 2>/dev/null; then
    OMP_LIBS="$(pkg-config --libs omp)"
    OMP_FLAGS="$(pkg-config --cflags omp)"
    echo "  OpenMP: enabled (pkg-config)"
else
    echo "  OpenMP: not available — serial mode"
fi

# Recompile engine with OpenMP if available
if [ -n "${OMP_FLAGS:-}" ]; then
    $CC $CFLAGS_BASE $VPU_FLAG $OMP_FLAGS -c keystone_engine.c -o keystone_engine.o
fi

# ---- Link core library ----
echo "Linking libkeystone_vector.so..."
$CC $CFLAGS_BASE -shared -o libkeystone_vector.so $OBJS -lm $OMP_LIBS -ldl

# ---- Level 1: CUDA (optional separate .so) ----
if [ "${KEYSTONE_NO_CUDA:-0}" != "1" ] && command -v nvcc >/dev/null 2>&1; then
    echo "Building CUDA module (libkeystone_cuda.so)..."
    if nvcc -O3 -shared -Xcompiler -fPIC keystone_cuda.cu -o libkeystone_cuda.so 2>/dev/null; then
        HAVE_CUDA=1
        echo "  CUDA module built successfully"
    else
        echo "  CUDA build failed — building CPU-only (OK)"
    fi
else
    echo "CUDA: skipped (no nvcc or KEYSTONE_NO_CUDA=1)"
fi

echo ""
echo "=== Build Complete ==="
echo "Library: $SCRIPT_DIR/libkeystone_vector.so"
echo "Backends compiled: scalar(always)$(echo "$OBJS" | grep -o 'avx512\|avx2\|avx\|sse42\|neon\|vpu' | sort -u | tr '\n' ' ')cuda=${HAVE_CUDA}"
echo ""
echo "Runtime dispatch will select the best available backend."
