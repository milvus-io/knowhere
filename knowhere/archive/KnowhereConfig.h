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

#pragma once

#include <string>
#include <vector>

namespace knowhere {

class KnowhereConfig {
 public:
    /**
     * set SIMD type
     */
    enum SimdType {
        AUTO = 0,  // enable all and depend on the system
        AVX512,    // only enable AVX512
        AVX2,      // only enable AVX2
        SSE4_2,    // only enable SSE4_2
        GENERIC,   // use arithmetic instead of SIMD
    };

    static std::string
    SetSimdType(const SimdType simd_type);

    /**
     * set knowhere index_file_slice_size
     */
    static void
    SetIndexFileSliceSize(const int64_t size);

    static int64_t
    GetIndexFileSliceSize();

    /**
     * Set openblas threshold
     *   if nq < use_blas_threshold, calculated by omp
     *   else, calculated by openblas
     */
    static void
    SetBlasThreshold(const int64_t use_blas_threshold);

    static int64_t
    GetBlasThreshold();

    /**
     * set Clustering early stop [0, 100]
     *   It is to reduce the number of iterations of K-means.
     *   Between each two iterations, if the optimization rate < early_stop_threshold, stop
     *   And if early_stop_threshold = 0, won't early stop
     */
    static void
    SetEarlyStopThreshold(const double early_stop_threshold);

    static double
    GetEarlyStopThreshold();

    /**
     * set Clustering type
     */
    enum ClusteringType {
        K_MEANS = 0,        // k-means (default)
        K_MEANS_PLUS_PLUS,  // k-means++
    };

    static void
    SetClusteringType(const ClusteringType clustering_type);

    /**
     * set Statistics Level [0, 3]
     */
    static void
    SetStatisticsLevel(const int32_t stat_level);

    // todo: add log level?
    /**
     * set Log handler
     */
    static void
    SetLogHandler();

    /**
     * init GPU Resource
     */
    static void
    InitGPUResource(const std::vector<int64_t>& gpu_ids);

    /**
     * free GPU Resource
     */
    static void
    FreeGPUResource();

    /**
     * The numebr of maximum parallel disk reads per thread. It should be set linearly proportional to `beam_width`.
     * Suggested ratio of this and `beam_width` is 2:1.
     * On Linux, the default limit of `aio-max-nr` is 65536, so the product of `num_threads` and `aio_maxnr` should
     * not exceed this value.
     * You can type `sudo sysctl -a | grep fs.aio-max-nr` on your terminal to see what is your default limit.
     * If you want to raise the default limit, you can type `sudo sysctl -w fs.aio-max-nr=X` on your terminal.
     */
    static void
    SetAioContextPool(size_t num_ctx, size_t max_events);
};

}  // namespace knowhere
