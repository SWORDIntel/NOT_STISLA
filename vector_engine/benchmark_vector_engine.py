#!/usr/bin/env python3
"""Benchmark KEYSTONE Vector Engine — scalar vs AVX vs brute force numpy."""
import numpy as np
import time
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from test_vector_engine import KeystoneVectorEngine, detect_features, SCALAR, AVX

DIM = 384
N_VECTORS = 10000
N_QUERIES = 100
K = 10

print(f"=== KEYSTONE Vector Engine Benchmark ===")
print(f"Vectors: {N_VECTORS}, Queries: {N_QUERIES}, Dim: {DIM}, K: {K}")
print(f"Features: {detect_features()}")
print()

# Generate test data
np.random.seed(42)
ids = np.arange(N_VECTORS, dtype=np.uint64)
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# Benchmark scalar
print("Scalar backend:")
eng_scalar = KeystoneVectorEngine(dim=DIM, capacity=N_VECTORS+100, force_backend=SCALAR)
print(f"  Backend: {eng_scalar.backend}")
t0 = time.time()
eng_scalar.upsert_batch(ids, vectors)
t_upsert_s = time.time() - t0
t0 = time.time()
for i in range(N_QUERIES):
    eng_scalar.search(queries[i], k=K)
t_search_s = time.time() - t0
print(f"  Upsert: {t_upsert_s*1000:.1f}ms ({N_VECTORS/t_upsert_s:.0f} vec/s)")
print(f"  Search: {t_search_s*1000:.1f}ms ({N_QUERIES/t_search_s:.0f} q/s)")
eng_scalar.close()

# Benchmark AVX
print("\nAVX backend:")
eng_avx = KeystoneVectorEngine(dim=DIM, capacity=N_VECTORS+100, force_backend=AVX)
print(f"  Backend: {eng_avx.backend}")
t0 = time.time()
eng_avx.upsert_batch(ids, vectors)
t_upsert_a = time.time() - t0
t0 = time.time()
for i in range(N_QUERIES):
    eng_avx.search(queries[i], k=K)
t_search_a = time.time() - t0
print(f"  Upsert: {t_upsert_a*1000:.1f}ms ({N_VECTORS/t_upsert_a:.0f} vec/s)")
print(f"  Search: {t_search_a*1000:.1f}ms ({N_QUERIES/t_search_a:.0f} q/s)")
eng_avx.close()

# Benchmark numpy brute force
print("\nNumPy brute force:")
t0 = time.time()
t_search_n = 0
for i in range(N_QUERIES):
    dots = vectors @ queries[i]
    top_k = np.argsort(-dots)[:K]
t_search_n = time.time() - t0
print(f"  Search: {t_search_n*1000:.1f}ms ({N_QUERIES/t_search_n:.0f} q/s)")

# Summary
print(f"\n=== Summary ===")
print(f"AVX vs Scalar speedup (search): {t_search_s/t_search_a:.1f}x")
print(f"AVX vs NumPy speedup (search):  {t_search_n/t_search_a:.1f}x")
print(f"Scalar per-query: {t_search_s/N_QUERIES*1000:.2f}ms")
print(f"AVX per-query:    {t_search_a/N_QUERIES*1000:.2f}ms")
print(f"NumPy per-query:  {t_search_n/N_QUERIES*1000:.2f}ms")
