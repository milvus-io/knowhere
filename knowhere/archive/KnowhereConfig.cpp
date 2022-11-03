// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "archive/KnowhereConfig.h"

#include <string>

#include "common/Log.h"
#include "faiss/Clustering.h"
#include "faiss/FaissHook.h"
#include "faiss/utils/distances.h"
#include "faiss/utils/utils.h"
#include "index/vector_index/Statistics.h"
#include "index/vector_index/helpers/Slice.h"
#ifdef KNOWHERE_WITH_DISKANN
#include "DiskANN/include/aio_context_pool.h"
#endif
#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace knowhere {

constexpr int64_t M_BYTE = 1024 * 1024;

std::string
KnowhereConfig::SetSimdType(const SimdType simd_type) {
    if (simd_type == SimdType::AUTO) {
        faiss::faiss_use_avx512 = true;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AUTO";
    } else if (simd_type == SimdType::AVX512) {
        faiss::faiss_use_avx512 = true;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AVX512";
    } else if (simd_type == SimdType::AVX2) {
        faiss::faiss_use_avx512 = false;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AVX2";
    } else if (simd_type == SimdType::SSE4_2) {
        faiss::faiss_use_avx512 = false;
        faiss::faiss_use_avx2 = false;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::SSE4_2";
    } else if (simd_type == SimdType::GENERIC) {
        faiss::faiss_use_avx512 = false;
        faiss::faiss_use_avx2 = false;
        faiss::faiss_use_sse4_2 = false;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::GENERIC";
    }

    std::string simd_str;
    faiss::hook_init(simd_str);
    LOG_KNOWHERE_INFO_ << "FAISS hook " << simd_str;
    return simd_str;
}

void
KnowhereConfig::SetIndexFileSliceSize(const int64_t size) {
    LOG_KNOWHERE_INFO_ << "Set knowhere::index_file_slice_size to " << size;
    knowhere::index_file_slice_size = size;
}

int64_t
KnowhereConfig::GetIndexFileSliceSize() {
    return knowhere::index_file_slice_size;
}

void
KnowhereConfig::SetBlasThreshold(const int64_t use_blas_threshold) {
    LOG_KNOWHERE_INFO_ << "Set faiss::distance_compute_blas_threshold to " << use_blas_threshold;
    faiss::distance_compute_blas_threshold = static_cast<int>(use_blas_threshold);
}

int64_t
KnowhereConfig::GetBlasThreshold() {
    return faiss::distance_compute_blas_threshold;
}

void
KnowhereConfig::SetEarlyStopThreshold(const double early_stop_threshold) {
    LOG_KNOWHERE_INFO_ << "Set faiss::early_stop_threshold to " << early_stop_threshold;
    faiss::early_stop_threshold = early_stop_threshold;
}

double
KnowhereConfig::GetEarlyStopThreshold() {
    return faiss::early_stop_threshold;
}

void
KnowhereConfig::SetClusteringType(const ClusteringType clustering_type) {
    LOG_KNOWHERE_INFO_ << "Set faiss::clustering_type to " << clustering_type;
    switch (clustering_type) {
        case ClusteringType::K_MEANS:
        default:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS;
            break;
        case ClusteringType::K_MEANS_PLUS_PLUS:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS_PLUS_PLUS;
            break;
    }
}

void
KnowhereConfig::SetStatisticsLevel(const int32_t stat_level) {
    LOG_KNOWHERE_INFO_ << "Set knowhere::STATISTICS_LEVEL to " << stat_level;
    knowhere::STATISTICS_LEVEL = stat_level;
    faiss::STATISTICS_LEVEL = stat_level;
}

void
KnowhereConfig::SetLogHandler() {
    faiss::LOG_ERROR_ = &knowhere::log_error_;
    faiss::LOG_WARNING_ = &knowhere::log_warning_;
}

void
KnowhereConfig::InitGPUResource(const std::vector<int64_t>& gpu_ids) {
#ifdef KNOWHERE_GPU_VERSION
    for (auto id : gpu_ids) {
        LOG_KNOWHERE_INFO_ << "init GPU resource for gpu id: " << id;
        // device_id, pinned_memory, temp_memory, resource_num
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(id, 256 * M_BYTE, 256 * M_BYTE, 2);
    }
#endif
}

void
KnowhereConfig::FreeGPUResource() {
#ifdef KNOWHERE_GPU_VERSION
    LOG_KNOWHERE_INFO_ << "free GPU resource";
    knowhere::FaissGpuResourceMgr::GetInstance().Free();  // Release gpu resources.
#endif
}

void
KnowhereConfig::SetAioContextPool(size_t num_ctx, size_t max_events) {
#ifdef KNOWHERE_WITH_DISKANN
    AioContextPool::InitGlobalAioPool(num_ctx, max_events);
#endif
}

}  // namespace knowhere
