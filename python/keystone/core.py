"""
KEYSTONE Core Python ctypes wrapper for libkeystone.so.
Provides hardware-accelerated interpolation search, batch search, auto-calibration, and anchor tables.
"""

import ctypes
import os
import numpy as np
from enum import IntEnum
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

# Locate libkeystone.so
_LIB_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "libkeystone.so"),
    "/usr/local/lib/libkeystone.so",
    "/usr/lib/libkeystone.so",
]

_lib = None
for p in _LIB_PATHS:
    if os.path.exists(p):
        _lib = ctypes.CDLL(p)
        break

if _lib is None:
    raise ImportError("libkeystone.so not found. Build with: make in KEYSTONE repo")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class KeystoneBackend(IntEnum):
    AUTO = 0
    SCALAR = 1
    C_BATCH = 2
    C_OPENMP = 3
    C_AVX2 = 4
    C_AVX512 = 5
    C_AMX = 6
    FORTRAN = 7


class WorkloadType(IntEnum):
    TELEMETRY = 0
    IDS = 1
    OFFSETS = 2
    EVENTS = 3


# ---------------------------------------------------------------------------
# C Structures
# ---------------------------------------------------------------------------
class _CAnchorTable(ctypes.Structure):
    pass

_CAnchorTable_p = ctypes.POINTER(_CAnchorTable)


class _CBatchItem(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int64),
        ("result", ctypes.c_size_t),
        ("ordinal", ctypes.c_size_t),
    ]


class _CParallelConfig(ctypes.Structure):
    _fields_ = [
        ("num_threads", ctypes.c_int),
        ("use_thread_pool", ctypes.c_int),
        ("batch_chunk", ctypes.c_size_t),
    ]


class _CBackendDecision(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("cpu_features", ctypes.c_uint32),
        ("array_size_bucket", ctypes.c_size_t),
        ("query_count_bucket", ctypes.c_size_t),
        ("thread_count", ctypes.c_int),
        ("estimated_ns_per_key", ctypes.c_double),
        ("p95_ns_per_key", ctypes.c_double),
        ("query_shape", ctypes.c_int),
        ("decision_source", ctypes.c_int),
        ("calibration_runs", ctypes.c_size_t),
        ("candidates_measured", ctypes.c_size_t),
    ]


@dataclass
class BackendDecision:
    backend: str
    decision_source: str
    query_shape: str
    estimated_ns_per_key: float
    p95_ns_per_key: float
    thread_count: int


# ---------------------------------------------------------------------------
# C Signatures
# ---------------------------------------------------------------------------
_lib.keystone_detect_cpu_features.argtypes = []
_lib.keystone_detect_cpu_features.restype = ctypes.c_uint32

_lib.keystone_anchor_table_create.argtypes = []
_lib.keystone_anchor_table_create.restype = _CAnchorTable_p

_lib.keystone_anchor_table_destroy.argtypes = [_CAnchorTable_p]
_lib.keystone_anchor_table_destroy.restype = None

_lib.keystone_anchor_table_size.argtypes = [_CAnchorTable_p]
_lib.keystone_anchor_table_size.restype = ctypes.c_size_t

_lib.keystone_search.argtypes = [
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.c_int64,
    _CAnchorTable_p,
    ctypes.c_size_t,
]
_lib.keystone_search.restype = ctypes.c_size_t

_lib.keystone_search_batch_auto.argtypes = [
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.POINTER(_CBatchItem),
    ctypes.c_size_t,
    _CAnchorTable_p,
    ctypes.c_size_t,
    ctypes.POINTER(_CParallelConfig),
]
_lib.keystone_search_batch_auto.restype = ctypes.c_size_t

# Zero-copy batch API: takes raw int64 keys + size_t results arrays directly.
# Avoids the per-key Python-level _CBatchItem construction/scatter loop.
_lib.keystone_search_keys_batch_auto.argtypes = [
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    _CAnchorTable_p,
    ctypes.c_size_t,
    ctypes.POINTER(_CParallelConfig),
]
_lib.keystone_search_keys_batch_auto.restype = ctypes.c_size_t

_lib.keystone_get_last_backend_decision.argtypes = [ctypes.POINTER(_CBackendDecision)]
_lib.keystone_get_last_backend_decision.restype = ctypes.c_int

_lib.keystone_backend_name.argtypes = [ctypes.c_int]
_lib.keystone_backend_name.restype = ctypes.c_char_p

_lib.keystone_decision_source_name.argtypes = [ctypes.c_int]
_lib.keystone_decision_source_name.restype = ctypes.c_char_p

_lib.keystone_query_shape_name.argtypes = [ctypes.c_int]
_lib.keystone_query_shape_name.restype = ctypes.c_char_p


# ---------------------------------------------------------------------------
# High-Level Keystone Classes
# ---------------------------------------------------------------------------
class AnchorTable:
    """
    Dynamic anchor learning table that caches piecewise spline predictions.
    """
    def __init__(self):
        self._ptr = _lib.keystone_anchor_table_create()
        if not self._ptr:
            raise RuntimeError("Failed to create Keystone Anchor Table")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._ptr:
            _lib.keystone_anchor_table_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()

    @property
    def handle(self):
        return self._ptr

    @property
    def size(self) -> int:
        return int(_lib.keystone_anchor_table_size(self._ptr)) if self._ptr else 0


class KeystoneSearch:
    """
    Target-silicon-tuned search engine for sorted 64-bit arrays.
    """

    @staticmethod
    def detect_cpu_features() -> int:
        """Returns bitmask of detected CPU SIMD features."""
        return int(_lib.keystone_detect_cpu_features())

    @staticmethod
    def search(
        arr: Union[np.ndarray, list],
        key: int,
        table: Optional[AnchorTable] = None,
        tol: int = 4,
    ) -> int:
        """
        Searches for `key` in sorted int64 array `arr`.
        Returns 0-based index if found, or -1 if not found.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.int64)
        elif arr.dtype != np.int64:
            arr = arr.astype(np.int64)

        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        tbl_ptr = table.handle if table else None

        res = _lib.keystone_search(c_arr, len(arr), int(key), tbl_ptr, int(tol))
        return -1 if res == (2**64 - 1) or res >= len(arr) else int(res)

    @staticmethod
    def search_batch(
        arr: Union[np.ndarray, list],
        keys: Union[np.ndarray, list],
        table: Optional[AnchorTable] = None,
        tol: int = 4,
        threads: int = 0,
    ) -> List[int]:
        """
        Executes auto-calibrated batch lookup across `keys`.
        Returns a list of integer indices (-1 for misses).
        """
        if not isinstance(arr, np.ndarray) or arr.dtype != np.int64:
            arr = np.ascontiguousarray(np.array(arr, dtype=np.int64))
        if not isinstance(keys, np.ndarray) or keys.dtype != np.int64:
            keys = np.ascontiguousarray(np.array(keys, dtype=np.int64))

        n_keys = len(keys)
        if n_keys == 0:
            return []

        c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        tbl_ptr = table.handle if table else None

        items = (_CBatchItem * n_keys)()
        for i in range(n_keys):
            items[i].key = int(keys[i])
            items[i].ordinal = i
            items[i].result = 2**64 - 1

        pcfg = None
        if threads > 0:
            pcfg = _CParallelConfig()
            pcfg.num_threads = threads
            pcfg.use_thread_pool = 1
            pcfg.batch_chunk = 256

        _lib.keystone_search_batch_auto(
            c_arr,
            len(arr),
            items,
            n_keys,
            tbl_ptr,
            int(tol),
            ctypes.byref(pcfg) if pcfg else None,
        )

        out = [-1] * n_keys
        for i in range(n_keys):
            r = items[i].result
            out[items[i].ordinal] = -1 if r == (2**64 - 1) or r >= len(arr) else int(r)
        return out

    @staticmethod
    def search_batch_keys(
        arr: Union[np.ndarray, list],
        keys: Union[np.ndarray, list],
        table: Optional[AnchorTable] = None,
        tol: int = 4,
        threads: int = 0,
    ) -> np.ndarray:
        """
        Zero-copy batch lookup across `keys`.

        Uses the native keystone_search_keys_batch_auto API which accepts
        contiguous int64 key and uintp result arrays directly from NumPy,
        avoiding the per-key Python-level _CBatchItem construction and
        scatter loops.  For large batches (e.g. 1M queries) this eliminates
        ~2M Python iterations and is substantially faster than search_batch.

        Returns a NumPy int64 array of indices (-1 for misses).
        """
        if not isinstance(arr, np.ndarray) or arr.dtype != np.int64:
            arr = np.ascontiguousarray(np.array(arr, dtype=np.int64))
        if not isinstance(keys, np.ndarray) or keys.dtype != np.int64:
            keys = np.ascontiguousarray(np.array(keys, dtype=np.int64))

        n_keys = len(keys)
        if n_keys == 0:
            return np.full(0, -1, dtype=np.int64)

        c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        c_keys = keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        tbl_ptr = table.handle if table else None

        # results array: uintp (size_t) on the C side, we use uintp on
        # the Python side and convert to int64 for the -1 sentinel.
        results = np.full(n_keys, ctypes.c_size_t(-1).value, dtype=np.uintp)
        c_results = results.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))

        pcfg = None
        if threads > 0:
            pcfg = _CParallelConfig()
            pcfg.num_threads = threads
            pcfg.use_thread_pool = 1
            pcfg.batch_chunk = 256

        _lib.keystone_search_keys_batch_auto(
            c_arr,
            len(arr),
            c_keys,
            n_keys,
            c_results,
            tbl_ptr,
            int(tol),
            ctypes.byref(pcfg) if pcfg else None,
        )

        # Convert size_t results to int64, mapping KEYSTONE_NOT_FOUND to -1.
        not_found = ctypes.c_size_t(-1).value
        out = results.astype(np.int64)
        out[results == not_found] = -1
        out[results >= len(arr)] = -1
        return out

    @staticmethod
    def get_last_decision() -> Optional[BackendDecision]:
        """
        Returns metadata about the auto-backend calibration decision used in the last batch search.
        """
        c_dec = _CBackendDecision()
        if _lib.keystone_get_last_backend_decision(ctypes.byref(c_dec)) == 0:
            b_name = _lib.keystone_backend_name(c_dec.backend).decode("utf-8")
            s_name = _lib.keystone_decision_source_name(c_dec.decision_source).decode("utf-8")
            q_name = _lib.keystone_query_shape_name(c_dec.query_shape).decode("utf-8")
            return BackendDecision(
                backend=b_name,
                decision_source=s_name,
                query_shape=q_name,
                estimated_ns_per_key=float(c_dec.estimated_ns_per_key),
                p95_ns_per_key=float(c_dec.p95_ns_per_key),
                thread_count=int(c_dec.thread_count),
            )
        return None
