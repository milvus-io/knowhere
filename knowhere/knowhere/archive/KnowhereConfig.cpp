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

#include <string>

#include "knowhere/archive/KnowhereConfig.h"
#include "knowhere/common/Log.h"

#if defined(__linux__) || defined(__MINGW64__)
#include "knowhere/index/vector_index/Statistics.h"
#include "NGT/lib/NGT/defines.h"
#include "faiss/Clustering.h"
#include "faiss/FaissHook.h"
#include "faiss/common.h"
#include "faiss/utils/distances.h"
#include "faiss/utils/utils.h"
#endif
#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace milvus {
namespace engine {

constexpr int64_t M_BYTE = 1024 * 1024;

std::string
KnowhereConfig::SetSimdType(const SimdType simd_type) {
#ifdef __APPLE__
    // return emtpy string if running on macos
    return "";
#elif defined(__linux__) || defined(__MINGW64__)
    if (simd_type == SimdType::AUTO) {
        faiss::faiss_use_avx512 = true;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_DEBUG_ << "FAISS expect simdType::AUTO";
    } else if (simd_type == SimdType::AVX512) {
        faiss::faiss_use_avx512 = true;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_DEBUG_ << "FAISS expect simdType::AVX512";
    } else if (simd_type == SimdType::AVX2) {
        faiss::faiss_use_avx512 = false;
        faiss::faiss_use_avx2 = true;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_DEBUG_ << "FAISS expect simdType::AVX2";
    } else if (simd_type == SimdType::SSE4_2) {
        faiss::faiss_use_avx512 = false;
        faiss::faiss_use_avx2 = false;
        faiss::faiss_use_sse4_2 = true;
        LOG_KNOWHERE_DEBUG_ << "FAISS expect simdType::SSE4_2";
    }

    std::string simd_str;
    faiss::hook_init(simd_str);
    LOG_KNOWHERE_DEBUG_ << "FAISS hook " << simd_str;
    return simd_str;
#else
    KNOWHERE_THROW_MSG("Unsupported SetSimdType on current platform!");
#endif
}

void
KnowhereConfig::SetBlasThreshold(const int64_t use_blas_threshold) {
#ifdef __APPLE__
    // do nothing
#elif defined(__linux__) || defined(__MINGW64__)
    faiss::distance_compute_blas_threshold = static_cast<int>(use_blas_threshold);
#else
    KNOWHERE_THROW_MSG("Unsupported SetBlasThreshold on current platform!");
#endif
}

void
KnowhereConfig::SetEarlyStopThreshold(const double early_stop_threshold) {
#ifdef __APPLE__
    // do nothing
#elif defined(__linux__) || defined(__MINGW64__)
    faiss::early_stop_threshold = early_stop_threshold;
#else
    KNOWHERE_THROW_MSG("Unsupported SetEarlyStopThreshold on current platform!");
#endif
}

void
KnowhereConfig::SetClusteringType(const ClusteringType clustering_type) {
#ifdef __APPLE__
    // do nothing
#elif defined(__linux__) || defined(__MINGW64__)
    switch (clustering_type) {
        case ClusteringType::K_MEANS:
        default:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS;
            break;
        case ClusteringType::K_MEANS_PLUS_PLUS:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS_PLUS_PLUS;
            break;
    }
#else
    KNOWHERE_THROW_MSG("Unsupported SetClusteringType on current platform!");
#endif
}

void
KnowhereConfig::SetStatisticsLevel(const int64_t stat_level) {
#ifdef __APPLE__
    // do nothing
#elif defined(__linux__) || defined(__MINGW64__)
    milvus::knowhere::STATISTICS_LEVEL = stat_level;
    faiss::STATISTICS_LEVEL = stat_level;
#else
    KNOWHERE_THROW_MSG("Unsupported SetStatisticsLevel on current platform!");
#endif
}

void
KnowhereConfig::SetLogHandler() {
#ifdef __APPLE__
    // do nothing
#elif defined(__linux__) || defined(__MINGW64__)
    faiss::LOG_ERROR_ = &knowhere::log_error_;
    faiss::LOG_WARNING_ = &knowhere::log_warning_;
    NGT_LOG_ERROR_ = &knowhere::log_error_;
    NGT_LOG_WARNING_ = &knowhere::log_warning_;
#else
    KNOWHERE_THROW_MSG("Unsupported SetLogHandler on current platform!");
#endif
}

#ifdef KNOWHERE_GPU_VERSION
void
KnowhereConfig::InitGPUResource(const std::vector<int64_t>& gpu_ids) {
    for (auto id : gpu_ids) {
        // device_id, pinned_memory, temp_memory, resource_num
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(id, 256 * M_BYTE, 256 * M_BYTE, 2);
    }
}

void
KnowhereConfig::FreeGPUResource() {
    knowhere::FaissGpuResourceMgr::GetInstance().Free();  // Release gpu resources.
}
#endif

}  // namespace engine
}  // namespace milvus
