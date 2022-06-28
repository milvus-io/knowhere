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

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "index/vector_index/ConfAdapter.h"
#include "index/vector_index/helpers/IndexParameter.h"

#ifdef KNOWHERE_GPU_VERSION
#include "faiss/gpu/utils/DeviceUtils.h"
#endif

namespace knowhere {

static const int64_t MIN_NBITS = 1;
static const int64_t MAX_NBITS = 16;
static const int64_t DEFAULT_NBITS = 8;
static const int64_t MIN_NLIST = 1;
static const int64_t MAX_NLIST = 65536;
static const int64_t MIN_NPROBE = 1;
static const int64_t MAX_NPROBE = MAX_NLIST;
static const int64_t DEFAULT_MIN_DIM = 1;
static const int64_t DEFAULT_MAX_DIM = 32768;
static const int64_t NGT_MIN_EDGE_SIZE = 1;
static const int64_t NGT_MAX_EDGE_SIZE = 200;
static const int64_t HNSW_MIN_EFCONSTRUCTION = 8;
static const int64_t HNSW_MAX_EFCONSTRUCTION = 512;
static const int64_t HNSW_MIN_M = 4;
static const int64_t HNSW_MAX_M = 64;
static const int64_t HNSW_MAX_EF = 32768;

static const std::vector<MetricType> default_metric_array{metric::L2, metric::IP};
static const std::vector<MetricType> default_binary_metric_array{metric::HAMMING, metric::JACCARD,
                                                                 metric::TANIMOTO, metric::SUBSTRUCTURE,
                                                                 metric::SUPERSTRUCTURE};

inline void
CheckIntegerRange(const Config& cfg, const std::string& key, int64_t min, int64_t max) {
    if (!cfg.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }
    if (!cfg[key].is_number_integer()) {
        KNOWHERE_THROW_FORMAT("Param '%s' should be an integer", key.data());
    }
    int64_t value = GetValueFromConfig<int64_t>(cfg, key);
    if (value < min || value > max) {
        KNOWHERE_THROW_FORMAT("Param '%s'(%ld) is not in range [%ld, %ld]", key.data(), value, min, max);
    }
}

inline void
CheckFloatRange(const Config& cfg, const std::string& key, float min, float max) {
    if (!cfg.contains(key)) {
        KNOWHERE_THROW_FORMAT("Param '%s' not exist", key.data());
    }
    if (!cfg[key].is_number_float()) {
        KNOWHERE_THROW_FORMAT("Param '%s' should be a float", key.data());
    }
    float value = GetValueFromConfig<float>(cfg, key);
    if (value < min || value > max) {
        KNOWHERE_THROW_FORMAT("Param '%s'(%f) is not in range [%f, %f]", key.data(), value, min, max);
    }
}

inline void
CheckMetricType(const Config& cfg, const std::vector<MetricType>& metric_types) {
    auto type = GetMetaMetricType(cfg);
    if (std::find(metric_types.begin(), metric_types.end(), type) == metric_types.end()) {
        KNOWHERE_THROW_FORMAT("Metric type '%s' is not supported for this index type", type.c_str());
    }
}

bool
ConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckMetricType(cfg, default_metric_array);
    CheckIntegerRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM);
    return true;
}

bool
ConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    const int64_t DEFAULT_MIN_K = 1;
    const int64_t DEFAULT_MAX_K = 16384;
    CheckIntegerRange(cfg, meta::TOPK, DEFAULT_MIN_K - 1, DEFAULT_MAX_K);
    return true;
}

int64_t
MatchNlist(int64_t size, int64_t nlist) {
    const int64_t MIN_POINTS_PER_CENTROID = 39;

    if (nlist * MIN_POINTS_PER_CENTROID > size) {
        // nlist is too large, adjust to a proper value
        nlist = std::max(static_cast<int64_t>(1), size / MIN_POINTS_PER_CENTROID);
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nlist " << nlist;
    }
    return nlist;
}

int64_t
MatchNbits(int64_t size, int64_t nbits) {
    if (size < (1 << nbits)) {
        // nbits is too large, adjust to a proper value
        if (size >= (1 << 8)) {
            nbits = 8;
        } else if (size >= (1 << 4)) {
            nbits = 4;
        } else if (size >= (1 << 2)) {
            nbits = 2;
        } else {
            nbits = 1;
        }
        LOG_KNOWHERE_WARNING_ << "Row num " << size << " match nbits " << nbits;
    }
    return nbits;
}

bool
IVFConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::NLIST, MIN_NLIST, MAX_NLIST);

    // auto tune params
    auto rows = GetMetaRows(cfg);
    auto nlist = GetIndexParamNlist(cfg);
    SetIndexParamNlist(cfg, MatchNlist(rows, nlist));

    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
IVFConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    int64_t max_nprobe = MAX_NPROBE;
#ifdef KNOWHERE_GPU_VERSION
    if (mode == IndexMode::MODE_GPU) {
        max_nprobe = faiss::gpu::getMaxKSelection();
    }
#endif
    CheckIntegerRange(cfg, indexparam::NPROBE, MIN_NPROBE, max_nprobe);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
IVFSQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    SetIndexParamNbits(cfg, DEFAULT_NBITS);
    return IVFConfAdapter::CheckTrain(cfg, mode);
}

bool
IVFPQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!IVFConfAdapter::CheckTrain(cfg, mode)) {
        return false;
    }
    CheckIntegerRange(cfg, indexparam::NBITS, MIN_NBITS, MAX_NBITS);

    auto rows = GetMetaRows(cfg);
    auto nbits = cfg.count(indexparam::NBITS) ? GetIndexParamNbits(cfg) : DEFAULT_NBITS;
    SetIndexParamNbits(cfg, MatchNbits(rows, nbits));

    auto m = GetIndexParamM(cfg);
    auto dimension = GetMetaDim(cfg);

    IndexMode ivfpq_mode = mode;
    return CheckPQParams(dimension, m, nbits, ivfpq_mode);
}

bool
IVFPQConfAdapter::CheckPQParams(int64_t dimension, int64_t m, int64_t nbits, IndexMode& mode) {
#ifdef KNOWHERE_GPU_VERSION
    if (mode == IndexMode::MODE_GPU && !IVFPQConfAdapter::CheckGPUPQParams(dimension, m, nbits)) {
        mode = IndexMode::MODE_CPU;
    }
#endif
    if (mode == IndexMode::MODE_CPU && !IVFPQConfAdapter::CheckCPUPQParams(dimension, m)) {
        return false;
    }
    return true;
}

bool
IVFPQConfAdapter::CheckGPUPQParams(int64_t dimension, int64_t m, int64_t nbits) {
    /*
     * Faiss 1.6
     * Only 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32 dims per sub-quantizer are currently supported with
     * no precomputed codes. Precomputed codes supports any number of dimensions, but will involve memory overheads.
     */
    static const std::vector<int64_t> support_dim_per_subquantizer{32, 28, 24, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1};
    static const std::vector<int64_t> support_subquantizer{96, 64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 8, 4, 3, 2, 1};

    if (!CheckCPUPQParams(dimension, m)) {
        return false;
    }

    int64_t sub_dim = dimension / m;
    return (std::find(std::begin(support_subquantizer), std::end(support_subquantizer), m) !=
            support_subquantizer.end()) &&
           (std::find(std::begin(support_dim_per_subquantizer), std::end(support_dim_per_subquantizer), sub_dim) !=
            support_dim_per_subquantizer.end()) &&
           (nbits == 8);
}

bool
IVFPQConfAdapter::CheckCPUPQParams(int64_t dimension, int64_t m) {
    return (dimension % m == 0);
}

bool
IVFHNSWConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    // HNSW param check
    CheckIntegerRange(cfg, indexparam::EFCONSTRUCTION, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION);
    CheckIntegerRange(cfg, indexparam::HNSW_M, HNSW_MIN_M, HNSW_MAX_M);

    // IVF param check
    CheckIntegerRange(cfg, indexparam::NLIST, MIN_NLIST, MAX_NLIST);

    // auto tune params
    auto rows = GetMetaRows(cfg);
    auto nlist = GetIndexParamNlist(cfg);
    SetIndexParamNlist(cfg, MatchNlist(rows, nlist));

    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
IVFHNSWConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    // HNSW param check
    CheckIntegerRange(cfg, indexparam::EF, GetMetaTopk(cfg), HNSW_MAX_EF);
    // IVF param check
    CheckIntegerRange(cfg, indexparam::NPROBE, MIN_NPROBE, MAX_NPROBE);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
HNSWConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EFCONSTRUCTION, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION);
    CheckIntegerRange(cfg, indexparam::HNSW_M, HNSW_MIN_M, HNSW_MAX_M);
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
HNSWConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EF, GetMetaTopk(cfg), HNSW_MAX_EF);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWFlatConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EFCONSTRUCTION, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION);
    CheckIntegerRange(cfg, indexparam::HNSW_M, HNSW_MIN_M, HNSW_MAX_M);
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWFlatConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EF, GetMetaTopk(cfg), HNSW_MAX_EF);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWPQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EFCONSTRUCTION, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION);
    CheckIntegerRange(cfg, indexparam::HNSW_M, HNSW_MIN_M, HNSW_MAX_M);

    auto dimension = GetMetaDim(cfg);
    if (!IVFPQConfAdapter::CheckCPUPQParams(dimension, GetIndexParamPQM(cfg))) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWPQConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EF, GetMetaTopk(cfg), HNSW_MAX_EF);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWSQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EFCONSTRUCTION, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION);
    CheckIntegerRange(cfg, indexparam::HNSW_M, HNSW_MIN_M, HNSW_MAX_M);
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWSQConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EF, GetMetaTopk(cfg), HNSW_MAX_EF);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
BinIDMAPConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckMetricType(cfg, default_binary_metric_array);
    CheckIntegerRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM);
    return true;
}

bool
BinIVFConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    static const std::vector<MetricType> metric_array{metric::HAMMING, metric::JACCARD, metric::TANIMOTO};
    CheckMetricType(cfg, metric_array);
    CheckIntegerRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM);
    CheckIntegerRange(cfg, indexparam::NLIST, MIN_NLIST, MAX_NLIST);

    // auto tune params
    auto rows = GetMetaRows(cfg);
    auto nlist = GetIndexParamNlist(cfg);
    SetIndexParamNlist(cfg, MatchNlist(rows, nlist));
    return true;
}

bool
ANNOYConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    // too large of n_trees takes much time, if there is real requirement, change this threshold.
    static int64_t MIN_N_TREES = 1;
    static int64_t MAX_N_TREES = 1024;
    CheckIntegerRange(cfg, indexparam::N_TREES, MIN_N_TREES, MAX_N_TREES);
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
ANNOYConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    static int64_t MIN_SEARCH_K = std::numeric_limits<int64_t>::min();
    static int64_t MAX_SEARCH_K = std::numeric_limits<int64_t>::max();
    CheckIntegerRange(cfg, indexparam::SEARCH_K, MIN_SEARCH_K, MAX_SEARCH_K);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

#ifdef KNOWHERE_SUPPORT_NGT
bool
NGTPANNGConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    CheckIntegerRange(cfg, indexparam::FORCEDLY_PRUNED_EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    CheckIntegerRange(cfg, indexparam::SELECTIVELY_PRUNED_EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    if (GetValueFromConfig<int64_t>(cfg, indexparam::SELECTIVELY_PRUNED_EDGE_SIZE) >=
        GetValueFromConfig<int64_t>(cfg, indexparam::FORCEDLY_PRUNED_EDGE_SIZE)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
NGTPANNGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::MAX_SEARCH_EDGES, -1, NGT_MAX_EDGE_SIZE);
    CheckFloatRange(cfg, indexparam::EPSILON, -1.0, 1.0);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
NGTONNGConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    CheckIntegerRange(cfg, indexparam::OUTGOING_EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    CheckIntegerRange(cfg, indexparam::INCOMING_EDGE_SIZE, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE);
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
NGTONNGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    CheckIntegerRange(cfg, indexparam::MAX_SEARCH_EDGES, -1, NGT_MAX_EDGE_SIZE);
    CheckFloatRange(cfg, indexparam::EPSILON, -1.0, 1.0);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}
#endif

#ifdef KNOWHERE_SUPPORT_NSG
bool
NSGConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    const int64_t MIN_KNNG = 5;
    const int64_t MAX_KNNG = 300;
    const int64_t MIN_SEARCH_LENGTH = 10;
    const int64_t MAX_SEARCH_LENGTH = 300;
    const int64_t MIN_OUT_DEGREE = 5;
    const int64_t MAX_OUT_DEGREE = 300;
    const int64_t MIN_CANDIDATE_POOL_SIZE = 50;
    const int64_t MAX_CANDIDATE_POOL_SIZE = 1000;

    CheckMetricType(cfg, default_metric_array);
    CheckIntegerRange(cfg, indexparam::KNNG, MIN_KNNG, MAX_KNNG);
    CheckIntegerRange(cfg, indexparam::SEARCH_LENGTH, MIN_SEARCH_LENGTH, MAX_SEARCH_LENGTH);
    CheckIntegerRange(cfg, indexparam::OUT_DEGREE, MIN_OUT_DEGREE, MAX_OUT_DEGREE);
    CheckIntegerRange(cfg, indexparam::CANDIDATE, MIN_CANDIDATE_POOL_SIZE, MAX_CANDIDATE_POOL_SIZE);

    // auto tune params
    auto rows = GetMetaRows(cfg);
    SetIndexParamNlist(cfg, MatchNlist(rows, 8192));

    int64_t nprobe = GetIndexParamNlist(cfg) * 0.1;
    SetIndexParamNprobe(cfg, (nprobe < 1) ? 1 : nprobe);
    return true;
}

bool
NSGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    static int64_t MIN_SEARCH_LENGTH = 1;
    static int64_t MAX_SEARCH_LENGTH = 300;
    CheckIntegerRange(cfg, indexparam::SEARCH_LENGTH, MIN_SEARCH_LENGTH, MAX_SEARCH_LENGTH);
    return ConfAdapter::CheckSearch(cfg, type, mode);
}
#endif

}  // namespace knowhere
