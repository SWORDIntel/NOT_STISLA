#ifndef KEYSTONE_CUDA_H
#define KEYSTONE_CUDA_H

#include "../include/keystone.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform a batch search using CUDA.
 *
 * @param arr Pointer to the sorted array.
 * @param n Size of the array.
 * @param items Array of batch items to search for.
 * @param num_items Number of items in the batch.
 * @return Number of successful searches.
 */
size_t keystone_search_batch_cuda(
    const int64_t* arr,
    size_t n,
    keystone_batch_item_t* items,
    size_t num_items
);

/**
 * Versioned variant for callers that may mutate the host array in-place.
 *
 * The cache identity includes dataset_version, so bumping it forces a
 * fresh device copy even if the host pointer and size are unchanged.
 *
 * @param arr Pointer to the sorted array.
 * @param n Size of the array.
 * @param items Array of batch items to search for.
 * @param num_items Number of items in the batch.
 * @param dataset_version Caller-supplied generation counter; bump after
 *                        mutating arr in-place to invalidate the cache.
 * @return Number of successful searches.
 */
size_t keystone_search_batch_cuda_versioned(
    const int64_t* arr,
    size_t n,
    keystone_batch_item_t* items,
    size_t num_items,
    uint64_t dataset_version
);

/**
 * Invalidate the CUDA device-array cache.
 *
 * Forces the next keystone_search_batch_cuda[_versioned] call to re-upload
 * the host array.  Call this if you free or realloc the host array without
 * bumping the dataset_version.
 */
void keystone_cuda_cache_invalidate(void);

#ifdef __cplusplus
}
#endif

#endif // KEYSTONE_CUDA_H
