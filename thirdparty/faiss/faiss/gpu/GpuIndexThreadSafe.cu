/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <algorithm>
#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t)256 * 1024 * 1024;

void GpuIndex::search_thread_safe(
        Index::idx_t n,
        const float* x,
        Index::idx_t k,
        int nprobe,
        float* distances,
        Index::idx_t* labels,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    // Maximum k-selection supported is based on the CUDA SDK
    FAISS_THROW_IF_NOT_FMT(
            k <= (Index::idx_t)getMaxKSelection(),
            "GPU index only supports k <= %d (requested %d)",
            getMaxKSelection(),
            (int)k); // select limitation

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // We guarantee that the searchImpl_ will be called with device-resident
    // pointers.

    // The input vectors may be too large for the GPU, but we still
    // assume that the output distances and labels are not.
    // Go ahead and make space for output distances and labels on the
    // GPU.
    // If we reach a point where all inputs are too big, we can add
    // another level of tiling.
    auto outDistances = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            distances,
            stream,
            {(int)n, (int)k});

    auto outLabels = toDeviceTemporary<Index::idx_t, 2>(
            resources_.get(), config_.device, labels, stream, {(int)n, (int)k});

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = (size_t)n * this->d * sizeof(float);

        if (dataSize >= minPagedSize_) {
            searchThreadSafeFromCpuPaged_(
                    n,
                    x,
                    k,
                    nprobe,
                    outDistances.data(),
                    outLabels.data(),
                    bitset);
            usePaged = true;
        }
    }

    if (!usePaged) {
        searchThreadSafeNonPaged_(
                n,
                x,
                k,
                nprobe,
                outDistances.data(),
                outLabels.data(),
                bitset);
    }

    // Copy back if necessary
    fromDevice<float, 2>(outDistances, distances, stream);
    fromDevice<Index::idx_t, 2>(outLabels, labels, stream);
}

void GpuIndex::searchThreadSafeNonPaged_(
        int n,
        const float* x,
        int k,
        int nprobe,
        float* outDistancesData,
        Index::idx_t* outIndicesData,
        const BitsetView bitset) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, (int)this->d});

    searchThreadSafeImpl_(n, vecs.data(), k, nprobe, outDistancesData, outIndicesData, bitset);
}

void GpuIndex::searchThreadSafeFromCpuPaged_(
        int n,
        const float* x,
        int k,
        int nprobe,
        float* outDistancesData,
        Index::idx_t* outIndicesData,
        const BitsetView bitset) const {
    Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
    Tensor<Index::idx_t, 2, true> outIndices(outIndicesData, {n, k});

    // Is pinned memory available?
    auto pinnedAlloc = resources_->getPinnedMemory();
    int pageSizeInVecs =
            (int)((pinnedAlloc.second / 2) / (sizeof(float) * this->d));

    if (!pinnedAlloc.first || pageSizeInVecs < 1) {
        // Just page without overlapping copy with compute
        int batchSize = utils::nextHighestPowerOf2(
                (int)((size_t)kNonPinnedPageSize / (sizeof(float) * this->d)));

        for (int cur = 0; cur < n; cur += batchSize) {
            int num = std::min(batchSize, n - cur);

            auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
            auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

            searchThreadSafeNonPaged_(
                    num,
                    x + (size_t)cur * this->d,
                    k,
                    nprobe,
                    outDistancesSlice.data(),
                    outIndicesSlice.data(),
                    bitset);
        }

        return;
    }

    //
    // Pinned memory is available, so we can overlap copy with compute.
    // We use two pinned memory buffers, and triple-buffer the
    // procedure:
    //
    // 1 CPU copy -> pinned
    // 2 pinned copy -> GPU
    // 3 GPU compute
    //
    // 1 2 3 1 2 3 ...   (pinned buf A)
    //   1 2 3 1 2 ...   (pinned buf B)
    //     1 2 3 1 ...   (pinned buf A)
    // time ->
    //
    auto defaultStream = resources_->getDefaultStream(config_.device);
    auto copyStream = resources_->getAsyncCopyStream(config_.device);

    FAISS_ASSERT(
            (size_t)pageSizeInVecs * this->d <=
            (size_t)std::numeric_limits<int>::max());

    float* bufPinnedA = (float*)pinnedAlloc.first;
    float* bufPinnedB = bufPinnedA + (size_t)pageSizeInVecs * this->d;
    float* bufPinned[2] = {bufPinnedA, bufPinnedB};

    // Reserve space on the GPU for the destination of the pinned buffer
    // copy
    DeviceTensor<float, 2, true> bufGpuA(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {(int)pageSizeInVecs, (int)this->d});
    DeviceTensor<float, 2, true> bufGpuB(
            resources_.get(),
            makeTempAlloc(AllocType::Other, defaultStream),
            {(int)pageSizeInVecs, (int)this->d});
    DeviceTensor<float, 2, true>* bufGpus[2] = {&bufGpuA, &bufGpuB};

    // Copy completion events for the pinned buffers
    std::unique_ptr<CudaEvent> eventPinnedCopyDone[2];

    // Execute completion events for the GPU buffers
    std::unique_ptr<CudaEvent> eventGpuExecuteDone[2];

    // All offsets are in terms of number of vectors; they remain within
    // int bounds (as this function only handles max in vectors)

    // Current start offset for buffer 1
    int cur1 = 0;
    int cur1BufIndex = 0;

    // Current start offset for buffer 2
    int cur2 = -1;
    int cur2BufIndex = 0;

    // Current start offset for buffer 3
    int cur3 = -1;
    int cur3BufIndex = 0;

    while (cur3 < n) {
        // Start async pinned -> GPU copy first (buf 2)
        if (cur2 != -1 && cur2 < n) {
            // Copy pinned to GPU
            int numToCopy = std::min(pageSizeInVecs, n - cur2);

            // Make sure any previous execution has completed before continuing
            auto& eventPrev = eventGpuExecuteDone[cur2BufIndex];
            if (eventPrev.get()) {
                eventPrev->streamWaitOnEvent(copyStream);
            }

            CUDA_VERIFY(cudaMemcpyAsync(
                    bufGpus[cur2BufIndex]->data(),
                    bufPinned[cur2BufIndex],
                    (size_t)numToCopy * this->d * sizeof(float),
                    cudaMemcpyHostToDevice,
                    copyStream));

            // Mark a completion event in this stream
            eventPinnedCopyDone[cur2BufIndex].reset(new CudaEvent(copyStream));

            // We pick up from here
            cur3 = cur2;
            cur2 += numToCopy;
            cur2BufIndex = (cur2BufIndex == 0) ? 1 : 0;
        }

        if (cur3 != -1 && cur3 < n) {
            // Process on GPU
            int numToProcess = std::min(pageSizeInVecs, n - cur3);

            // Make sure the previous copy has completed before continuing
            auto& eventPrev = eventPinnedCopyDone[cur3BufIndex];
            FAISS_ASSERT(eventPrev.get());

            eventPrev->streamWaitOnEvent(defaultStream);

            // Create tensor wrappers
            // DeviceTensor<float, 2, true> input(bufGpus[cur3BufIndex]->data(),
            //                                    {numToProcess, this->d});
            auto outDistancesSlice =
                    outDistances.narrowOutermost(cur3, numToProcess);
            auto outIndicesSlice =
                    outIndices.narrowOutermost(cur3, numToProcess);

            searchThreadSafeImpl_(
                    numToProcess,
                    bufGpus[cur3BufIndex]->data(),
                    k,
                    nprobe,
                    outDistancesSlice.data(),
                    outIndicesSlice.data(),
                    bitset);

            // Create completion event
            eventGpuExecuteDone[cur3BufIndex].reset(
                    new CudaEvent(defaultStream));

            // We pick up from here
            cur3BufIndex = (cur3BufIndex == 0) ? 1 : 0;
            cur3 += numToProcess;
        }

        if (cur1 < n) {
            // Copy CPU mem to CPU pinned
            int numToCopy = std::min(pageSizeInVecs, n - cur1);

            // Make sure any previous copy has completed before continuing
            auto& eventPrev = eventPinnedCopyDone[cur1BufIndex];
            if (eventPrev.get()) {
                eventPrev->cpuWaitOnEvent();
            }

            memcpy(bufPinned[cur1BufIndex],
                   x + (size_t)cur1 * this->d,
                   (size_t)numToCopy * this->d * sizeof(float));

            // We pick up from here
            cur2 = cur1;
            cur1 += numToCopy;
            cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
        }
    }
}

} // namespace gpu
} // namespace faiss
