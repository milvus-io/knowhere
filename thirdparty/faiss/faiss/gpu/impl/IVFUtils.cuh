/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/utils/Tensor.cuh>

// A collection of utility functions for IVFPQ and IVFFlat, for
// post-processing and k-selecting the results
namespace faiss {
namespace gpu {

class GpuResources;

// This is warp divergence central, but this is really a final step
// and happening a small number of times
inline __device__ int binarySearchForBucket(
        int* prefixSumOffsets,
        int size,
        int val) {
    int start = 0;
    int end = size;

    while (end - start > 0) {
        int mid = start + (end - start) / 2;

        int midVal = prefixSumOffsets[mid];

        // Find the first bucket that we are <=
        if (midVal <= val) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    // We must find the bucket that it is in
    assert(start != size);

    return start;
}

inline __device__ Index::idx_t getUserIndex(
        int listId,
        int listOffset,
        void** listIndices,
        IndicesOptions opt) {
    Index::idx_t index = -1;

    if (opt == INDICES_32_BIT) {
        index = (Index::idx_t)((int*)listIndices[listId])[listOffset];
    } else if (opt == INDICES_64_BIT) {
        index = ((Index::idx_t*)listIndices[listId])[listOffset];
    } else {
        index = ((Index::idx_t)listId << 32 | (Index::idx_t) listOffset);
    }
    return index;
}

inline __device__ Index::idx_t getListIndex(
        int queryId,
        int offset,
        void** listIndices,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<int, 2, true>& topQueryToCentroid,
        IndicesOptions opt) {

    // In order to determine the actual user index, we need to first
    // determine what list it was in.
    // We do this by binary search in the prefix sum list.
    int probe = binarySearchForBucket(prefixSumOffsets[queryId].data(),
                                      prefixSumOffsets.getSize(1),
                                      offset);

    // This is then the probe for the query; we can find the actual
    // list ID from this
    int listId = topQueryToCentroid[queryId][probe];

    // Now, we need to know the offset within the list
    // We ensure that before the array (at offset -1), there is a 0 value
    int listStart = *(prefixSumOffsets[queryId][probe].data() - 1);
    int listOffset = offset - listStart;

    // This gives us our final index
    return getUserIndex(listId, listOffset, listIndices, opt);
}

/// Function for multi-pass scanning that collects the length of
/// intermediate results for all (query, probe) pair
void runCalcListOffsets(
        GpuResources* res,
        Tensor<int, 2, true>& topQueryToCentroid,
        thrust::device_vector<int>& listLengths,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        cudaStream_t stream);

/// Performs a first pass of k-selection on the results
void runPass1SelectLists(
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<int, 2, true>& coarseIndices,
        Tensor<uint8_t, 1, true>& bitset,
        Tensor<float, 1, true>& distance,
        int nprobe,
        int k,
        bool chooseLargest,
        Tensor<float, 3, true>& heapDistances,
        Tensor<int, 3, true>& heapIndices,
        cudaStream_t stream);

/// Performs a final pass of k-selection on the results, producing the
/// final indices
void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<int, 2, true>& heapIndices,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<int, 2, true>& topQueryToCentroid,
        int k,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
