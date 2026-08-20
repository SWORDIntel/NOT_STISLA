#include <cuda_runtime.h>
#include <stdio.h>
#include "keystone_cuda.h"

// Use __ldg to route reads through the read-only texture cache for high warp divergence efficiency
#if __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif

// ---------------------------------------------------------------------------
// Lock-free double-buffered device cache
//
// Two slots, each with its own CUDA stream.  Writers acquire a slot via CAS
// on the in_use field (0→1), copy the host array to the device buffer, then
// publish the slot as valid (in_use = 2) and update g_active_slot.  Readers
// check both slots for a matching host pointer + length and, on a hit, reuse
// the cached device pointer without copying.
//
// Synchronization is entirely via atomic operations and CUDA stream ordering:
//   * Before overwriting a slot's device buffer, the writer calls
//     cudaStreamSynchronize on that slot's stream so that any in-flight
//     kernels still reading the old contents have finished.
//   * The previously active slot is retired (marked free) only after its
//     stream is synchronized, guaranteeing no reader is still in flight.
//
// This eliminates the global spinlock that serialized all host-side batch
// submissions, allowing batches that hit different cache slots to execute
// concurrently on the GPU.
// ---------------------------------------------------------------------------
typedef struct {
    const int64_t* h_arr;      // host pointer (for comparison)
    int64_t* d_arr;            // device pointer
    size_t n;                  // array length
    volatile int in_use;       // 0=free, 1=being written, 2=valid
} keystone_cuda_cache_slot_t;

// __sync_swap is not available on all GCC versions (e.g. GCC 14+); use
// __sync_lock_test_and_set which provides identical atomic-exchange semantics
// (stores val into *ptr and returns the previous contents of *ptr).
#ifndef KEYSTONE_SYNC_SWAP
#define KEYSTONE_SYNC_SWAP(ptr, val) __sync_lock_test_and_set(ptr, val)
#endif

static keystone_cuda_cache_slot_t g_cache_slots[2];
static volatile int g_active_slot = 0;  // index of currently valid slot
static cudaStream_t g_streams[2];
static volatile int g_streams_init = 0; // 0=not init, 1=initializing, 2=ready

__global__ void keystone_search_kernel_scalar(
    const int64_t* __restrict__ arr,
    size_t n,
    keystone_batch_item_t* __restrict__ items,
    size_t num_items,
    unsigned long long* __restrict__ d_success_count)
{
    // Scalar Branchless Binary Search (Optimized for older GPUs / Pascal / low SM count)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    int64_t key = items[idx].key;
    size_t found_idx = KEYSTONE_NOT_FOUND;

    // Quick bounds check using texture cache
    if (n > 0 && key >= LDG(&arr[0]) && key <= LDG(&arr[n - 1])) {
        size_t lo = 0;
        size_t len = n;
        
        while (len > 1) {
            size_t half = len / 2;
            size_t mid = lo + half - 1;
            int64_t mid_val = LDG(&arr[mid]);
            
            // Branchless advance
            lo = (mid_val < key) ? (lo + half) : lo;
            len -= half;
        }
        
        if (LDG(&arr[lo]) == key) {
            found_idx = lo;
        }
    }

    items[idx].result = found_idx;
    if (found_idx != KEYSTONE_NOT_FOUND) {
        atomicAdd(d_success_count, 1ULL);
    }
}

__global__ void keystone_search_kernel_warp_cooperative(
    const int64_t* __restrict__ arr,
    size_t n,
    keystone_batch_item_t* __restrict__ items,
    size_t num_items,
    unsigned long long* __restrict__ d_success_count)
{
    // Warp-Cooperative 32-ary Search (Optimized for H200 / Hopper)
    // Instead of 1 thread = 1 query (which causes warp divergence and memory serialization),
    // we use 1 WARP (32 threads) = 1 query.
    // The warp divides the search space into 32 segments per iteration, reducing a 24-depth
    // binary search into a mere 5-depth 32-ary search. All memory reads are perfectly coalesced/parallel.

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % 32;
    unsigned int warp_id = (blockIdx.x * blockDim.x + tid) / 32;
    
    if (warp_id >= num_items) return;

    int64_t key = items[warp_id].key;
    size_t found_idx = KEYSTONE_NOT_FOUND;

    if (n > 0) {
        // Broadcast bounds check across the warp to avoid divergent reads
        int64_t bound_min = (lane_id == 0) ? LDG(&arr[0]) : 0;
        int64_t bound_max = (lane_id == 31) ? LDG(&arr[n - 1]) : 0;
        
        bound_min = __shfl_sync(0xFFFFFFFF, bound_min, 0);
        bound_max = __shfl_sync(0xFFFFFFFF, bound_max, 31);
        
        if (key >= bound_min && key <= bound_max) {
            size_t lo = 0;
            size_t hi = n;
            
            // N-ary search loop (N=32)
            while (hi - lo > 32) {
                size_t step = (hi - lo) / 32;
                size_t probe_idx = lo + lane_id * step;
                
                int64_t probe_val = LDG(&arr[probe_idx]);
                
                // Ballot creates a bitmask of all lanes where probe_val <= key
                unsigned int mask = __ballot_sync(0xFFFFFFFF, probe_val <= key);
                
                // The highest set bit tells us the exact segment the key falls into
                int highest_lane = 31 - __clz(mask); 
                
                lo = lo + highest_lane * step;
                hi = (highest_lane == 31) ? hi : (lo + step);
            }
            
            // Final phase: the remaining search space is <= 32 elements.
            // A single parallel read by the warp finds the exact match.
            size_t len = hi - lo;
            size_t probe_idx = lo + lane_id;
            
            int is_match = 0;
            if (lane_id < len && LDG(&arr[probe_idx]) == key) {
                is_match = 1;
            }
            
            unsigned int match_mask = __ballot_sync(0xFFFFFFFF, is_match);
            if (match_mask != 0) {
                // If there are multiple matches, __ffs gets the lowest index
                int match_lane = __ffs(match_mask) - 1;
                found_idx = lo + match_lane;
            }
        }
    }

    // Only lane 0 writes the result back to global memory
    if (lane_id == 0) {
        items[warp_id].result = found_idx;
        if (found_idx != KEYSTONE_NOT_FOUND) {
            atomicAdd(d_success_count, 1ULL);
        }
    }
}

extern "C" size_t keystone_search_batch_cuda(
    const int64_t* arr,
    size_t n,
    keystone_batch_item_t* items,
    size_t num_items)
{
    if (n == 0 || num_items == 0) return 0;

    // --- Lazy one-time initialization of per-slot CUDA streams ---
    if (g_streams_init != 2) {
        if (__sync_val_compare_and_swap(&g_streams_init, 0, 1) == 0) {
            cudaStreamCreate(&g_streams[0]);
            cudaStreamCreate(&g_streams[1]);
            __sync_synchronize();  // publish stream handles before flag
            g_streams_init = 2;
        } else {
            while (g_streams_init != 2) {
                /* spin briefly until initialization finishes */
            }
        }
    }

    int64_t* d_arr = NULL;
    cudaStream_t stream;
    bool used_temp = false;  // true when we fell back to a temporary buffer

    // --- Cache lookup: check both slots for a hit (same host pointer + length) ---
    for (int i = 0; i < 2; i++) {
        if (g_cache_slots[i].in_use == 2 &&
            g_cache_slots[i].h_arr == arr &&
            g_cache_slots[i].n == n) {
            // Cache hit — reuse the cached device pointer, no copy needed
            d_arr = g_cache_slots[i].d_arr;
            stream = g_streams[i];
            break;
        }
    }

    // --- Cache miss: try to acquire a free slot via CAS (0 → 1) ---
    if (d_arr == NULL) {
        int acquired = -1;
        for (int i = 0; i < 2; i++) {
            if (__sync_val_compare_and_swap(&g_cache_slots[i].in_use, 0, 1) == 0) {
                acquired = i;
                break;
            }
        }

        if (acquired >= 0) {
            // Ensure any in-flight kernels previously dispatched on this slot's
            // stream have finished reading before we overwrite the device buffer.
            cudaStreamSynchronize(g_streams[acquired]);

            // Reallocate the device buffer if it is missing or the length changed
            if (g_cache_slots[acquired].d_arr && g_cache_slots[acquired].n != n) {
                cudaFree(g_cache_slots[acquired].d_arr);
                g_cache_slots[acquired].d_arr = NULL;
            }
            if (!g_cache_slots[acquired].d_arr) {
                if (cudaMalloc((void**)&g_cache_slots[acquired].d_arr,
                               n * sizeof(int64_t)) != cudaSuccess) {
                    g_cache_slots[acquired].in_use = 0;  // release the slot
                    return 0;  // OOM
                }
            }

            // Copy the sorted array to the device buffer
            cudaMemcpy(g_cache_slots[acquired].d_arr, arr,
                       n * sizeof(int64_t), cudaMemcpyHostToDevice);
            g_cache_slots[acquired].h_arr = arr;
            g_cache_slots[acquired].n = n;

            // Publish: make the slot's contents visible, then mark it valid
            __sync_synchronize();
            g_cache_slots[acquired].in_use = 2;

            // Atomically update the active-slot index and retrieve the old value
            int old_active = KEYSTONE_SYNC_SWAP(&g_active_slot, acquired);

            // Retire the previously active slot: wait for any in-flight kernels
            // that are still reading from it, then mark it free for reuse.
            if (old_active != acquired) {
                cudaStreamSynchronize(g_streams[old_active]);
                __sync_val_compare_and_swap(&g_cache_slots[old_active].in_use, 2, 0);
            }

            d_arr = g_cache_slots[acquired].d_arr;
            stream = g_streams[acquired];
        } else {
            // --- Fallback: both slots are being written — allocate a temp buffer ---
            if (cudaMalloc((void**)&d_arr, n * sizeof(int64_t)) != cudaSuccess) {
                return 0;  // OOM
            }
            cudaMemcpy(d_arr, arr, n * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaStreamCreate(&stream);
            used_temp = true;
        }
    }

    // --- Allocate per-batch device buffers ---
    keystone_batch_item_t* d_items = NULL;
    unsigned long long* d_success_count = NULL;

    if (cudaMalloc((void**)&d_items, num_items * sizeof(keystone_batch_item_t)) != cudaSuccess) {
        if (used_temp) { cudaFree(d_arr); cudaStreamDestroy(stream); }
        return 0;
    }
    if (cudaMalloc((void**)&d_success_count, sizeof(unsigned long long)) != cudaSuccess) {
        cudaFree(d_items);
        if (used_temp) { cudaFree(d_arr); cudaStreamDestroy(stream); }
        return 0;
    }

    // Upload items and zero the counter asynchronously on the chosen stream
    cudaMemcpyAsync(d_items, items, num_items * sizeof(keystone_batch_item_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_success_count, 0, sizeof(unsigned long long), stream);

    // Adaptive Pathway Decision based on GPU architecture and capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Warp-cooperative search is heavily beneficial on Pascal (Compute 6.0) and newer
    // due to hardware sync primitives (__ballot_sync, __shfl_sync) and massive memory bandwidth.
    // On exceptionally old architectures (Compute < 6.0), scalar branchless achieves
    // better execution due to lack of warp-synchronous optimizations.
    bool use_warp_cooperative = (prop.major >= 6);

    if (use_warp_cooperative) {
        // Compute optimal thread blocks for Warp-Cooperative Launch
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks = (num_items + warps_per_block - 1) / warps_per_block;

        keystone_search_kernel_warp_cooperative<<<blocks, threads_per_block, 0, stream>>>(
            d_arr, n, d_items, num_items, d_success_count);
    } else {
        // Compute optimal thread blocks for Scalar Launch
        int threads_per_block = 256;
        int blocks = (num_items + threads_per_block - 1) / threads_per_block;

        keystone_search_kernel_scalar<<<blocks, threads_per_block, 0, stream>>>(
            d_arr, n, d_items, num_items, d_success_count);
    }

    // Download items back asynchronously
    cudaMemcpyAsync(items, d_items, num_items * sizeof(keystone_batch_item_t),
                    cudaMemcpyDeviceToHost, stream);

    unsigned long long h_success_count = 0;
    cudaMemcpyAsync(&h_success_count, d_success_count, sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost, stream);

    // Sync the stream to ensure all operations finish before returning
    cudaStreamSynchronize(stream);

    // Clean up temporary resources (cached slot resources are kept for reuse)
    if (used_temp) {
        cudaFree(d_arr);
        cudaStreamDestroy(stream);
    }

    cudaFree(d_items);
    cudaFree(d_success_count);

    return (size_t)h_success_count;
}
