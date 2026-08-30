#!/usr/bin/env python3
"""
KEYSTONE Vector Engine — Python ctypes bindings.

Hardware-agnostic vector store with graceful fallback:
  CUDA -> AVX-512 -> AVX2 -> AVX -> SSE4.2 -> NEON -> VPU -> scalar

VPU support is compiled in only if KEYSTONE_HAVE_VPU was set at build time.
If compiled in but the VPU socket is not present at runtime, the engine
silently falls back to CPU SIMD.
"""
import ctypes
import numpy as np
from pathlib import Path

# Error codes
KEYSTONE_OK = 0
KEYSTONE_ERR_NULL = -1
KEYSTONE_ERR_PARAM = -2
KEYSTONE_ERR_DIM = -3
KEYSTONE_ERR_OOM = -4
KEYSTONE_ERR_CAPACITY = -5
KEYSTONE_ERR_IO = -6
KEYSTONE_ERR_INTERNAL = -7
KEYSTONE_ERR_BACKEND = -8

# Backend IDs
SCALAR = 0
VPU = 1
NEON = 2
SSE42 = 3
AVX = 4
AVX2 = 5
AVX512 = 6
CUDA = 7

# Metrics
L2 = 0
DOT = 1
COSINE = 2


class _Config(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_uint32),
        ("capacity", ctypes.c_uint32),
        ("metric", ctypes.c_int),
        ("lsh_num_tables", ctypes.c_uint32),
        ("lsh_hash_bits", ctypes.c_uint32),
        ("lsh_probes", ctypes.c_uint32),
        ("rerank_k", ctypes.c_uint32),
        ("omp_threads", ctypes.c_int),
        ("force_backend", ctypes.c_int),
        ("vpu_socket_path", ctypes.c_char * 108),
        ("vpu_enabled", ctypes.c_int),
    ]


class _Result(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("distance", ctypes.c_float)]


def _load_lib():
    paths = [
        Path(__file__).parent / "libkeystone_vector.so",
        Path(__file__).parent.parent / "vector_engine" / "libkeystone_vector.so",
        Path(__file__).parent.parent / "KEYSTONE" / "vector_engine" / "libkeystone_vector.so",
        Path("/usr/local/lib/libkeystone_vector.so"),
        Path("libkeystone_vector.so"),
    ]
    for p in paths:
        if p.exists():
            # RTLD_GLOBAL so dlsym(RTLD_DEFAULT, ...) can find kernel symbols
            return ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
    raise RuntimeError(f"libkeystone_vector.so not found in {[str(p) for p in paths]}")


_LIB = _load_lib()

# Set up signatures
_LIB.keystone_vec_detect_features.restype = ctypes.c_uint32
_LIB.keystone_vec_detect_features.argtypes = []

_LIB.keystone_vec_backend_name.restype = ctypes.c_char_p
_LIB.keystone_vec_backend_name.argtypes = [ctypes.c_int]

_LIB.keystone_vec_engine_create.restype = ctypes.c_int
_LIB.keystone_vec_engine_create.argtypes = [ctypes.POINTER(_Config),
                                             ctypes.POINTER(ctypes.c_void_p)]

_LIB.keystone_vec_engine_destroy.restype = None
_LIB.keystone_vec_engine_destroy.argtypes = [ctypes.c_void_p]

_LIB.keystone_vec_engine_get_backend.restype = ctypes.c_int
_LIB.keystone_vec_engine_get_backend.argtypes = [ctypes.c_void_p]

_LIB.keystone_vec_upsert_batch.restype = ctypes.c_int
_LIB.keystone_vec_upsert_batch.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_uint64),
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_uint32]

_LIB.keystone_vec_search_batch.restype = ctypes.c_int
_LIB.keystone_vec_search_batch.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_uint32, ctypes.c_uint32,
                                            ctypes.POINTER(_Result)]

_LIB.keystone_vec_search.restype = ctypes.c_int
_LIB.keystone_vec_search.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_uint32,
                                      ctypes.POINTER(_Result)]

_LIB.keystone_vec_engine_dim.restype = ctypes.c_uint32
_LIB.keystone_vec_engine_dim.argtypes = [ctypes.c_void_p]

_LIB.keystone_vec_engine_count.restype = ctypes.c_uint64
_LIB.keystone_vec_engine_count.argtypes = [ctypes.c_void_p]

_LIB.keystone_vec_save.restype = ctypes.c_int
_LIB.keystone_vec_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_LIB.keystone_vec_load.restype = ctypes.c_int
_LIB.keystone_vec_load.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]


class KeystoneVectorEngine:
    """Hardware-agnostic vector store with graceful fallback.

    Falls back automatically:
      CUDA -> AVX-512 -> AVX2 -> AVX -> SSE4.2 -> NEON -> VPU -> scalar

    VPU is only used if compiled with KEYSTONE_HAVE_VPU=1 AND the VPU
    socket is reachable at runtime. Otherwise silently uses CPU SIMD.
    """

    def __init__(self, dim=384, capacity=1_000_000, metric=COSINE,
                 lsh_num_tables=8, lsh_hash_bits=12, lsh_probes=4,
                 rerank_k=256, force_backend=-1,
                 vpu_enabled=False, vpu_socket_path=""):
        cfg = _Config(
            dim=dim,
            capacity=capacity,
            metric=metric,
            lsh_num_tables=lsh_num_tables,
            lsh_hash_bits=lsh_hash_bits,
            lsh_probes=lsh_probes,
            rerank_k=rerank_k,
            omp_threads=0,
            force_backend=force_backend,
            vpu_socket_path=vpu_socket_path.encode() if vpu_socket_path else b"",
            vpu_enabled=1 if vpu_enabled else 0,
        )
        self._dim = dim
        self._e = ctypes.c_void_p()
        rc = _LIB.keystone_vec_engine_create(ctypes.byref(cfg), ctypes.byref(self._e))
        if rc != KEYSTONE_OK:
            raise RuntimeError(f"keystone_vec_engine_create failed: {rc}")

    @property
    def backend(self) -> str:
        """Current backend name (e.g. 'avx', 'scalar', 'cuda').
        Changes if a backend fails and falls back."""
        b = _LIB.keystone_vec_engine_get_backend(self._e)
        return _LIB.keystone_vec_backend_name(b).decode()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def count(self) -> int:
        return _LIB.keystone_vec_engine_count(self._e)

    def upsert_batch(self, ids, vectors) -> None:
        """Upsert vectors. Source-agnostic: VPU, CPU, GPU, hash_bow — all identical."""
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self._dim:
            raise ValueError(f"expected (n, {self._dim}) float32, got {vecs.shape}")
        rc = _LIB.keystone_vec_upsert_batch(
            self._e,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vecs.shape[0])
        if rc != KEYSTONE_OK:
            raise RuntimeError(f"upsert_batch failed: {rc}")

    def search_batch(self, queries, k=10):
        """Search batch. Returns (ids, distances) arrays of shape (nq, k)."""
        qs = np.ascontiguousarray(queries, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[1] != self._dim:
            raise ValueError(f"expected (nq, {self._dim}) float32, got {qs.shape}")
        nq = qs.shape[0]
        out = (_Result * (nq * k))()
        rc = _LIB.keystone_vec_search_batch(
            self._e,
            qs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nq, k, out)
        if rc != KEYSTONE_OK:
            raise RuntimeError(f"search_batch failed: {rc}")
        arr = np.ctypeslib.as_array(out).reshape(nq, k)
        return arr["id"].copy(), arr["distance"].copy()

    def search(self, query, k=10):
        """Single query search. Returns (ids, distances) of shape (k,)."""
        q = np.ascontiguousarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        ids, dists = self.search_batch(q, k)
        return ids[0], dists[0]

    def save(self, path: str) -> None:
        rc = _LIB.keystone_vec_save(self._e, path.encode())
        if rc != KEYSTONE_OK:
            raise RuntimeError(f"save failed: {rc}")

    @classmethod
    def load(cls, path: str) -> "KeystoneVectorEngine":
        e = ctypes.c_void_p()
        rc = _LIB.keystone_vec_load(ctypes.byref(e), path.encode())
        if rc != KEYSTONE_OK:
            raise RuntimeError(f"load failed: {rc}")
        obj = cls.__new__(cls)
        obj._e = e
        obj._dim = _LIB.keystone_vec_engine_dim(e)
        return obj

    def close(self):
        if self._e:
            _LIB.keystone_vec_engine_destroy(self._e)
            self._e = ctypes.c_void_p()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"KeystoneVectorEngine(dim={self._dim}, count={self.count}, backend='{self.backend}')"


def detect_features() -> dict:
    """Detect available hardware features. Returns dict of feature -> bool."""
    flags = _LIB.keystone_vec_detect_features()
    return {
        "sse42": bool(flags & (1 << 0)),
        "avx": bool(flags & (1 << 1)),
        "avx2": bool(flags & (1 << 2)),
        "avx512f": bool(flags & (1 << 3)),
        "neon": bool(flags & (1 << 4)),
        "vpu": bool(flags & (1 << 5)),
    }


if __name__ == "__main__":
    # Quick self-test
    print("=== KEYSTONE Vector Engine Self-Test ===")
    features = detect_features()
    print(f"Detected features: {features}")

    engine = KeystoneVectorEngine(dim=384, capacity=10000)
    print(f"Engine: {engine}")

    # Generate test vectors
    np.random.seed(42)
    n = 1000
    ids = np.arange(n, dtype=np.uint64)
    vectors = np.random.randn(n, 384).astype(np.float32)
    # Normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    print(f"Upserting {n} vectors...")
    engine.upsert_batch(ids, vectors)
    print(f"Count: {engine.count}")

    # Search for vector 0
    query = vectors[0]
    print("Searching for vector 0...")
    result_ids, result_dists = engine.search(query, k=5)
    print(f"Top-5 results:")
    for i in range(5):
        print(f"  id={result_ids[i]}, dist={result_dists[i]:.6f}")

    # Verify vector 0 is closest to itself
    assert result_ids[0] == 0, f"Expected id=0, got {result_ids[0]}"
    assert result_dists[0] < 0.001, f"Expected dist~0, got {result_dists[0]}"
    print("\nSelf-test PASSED")
