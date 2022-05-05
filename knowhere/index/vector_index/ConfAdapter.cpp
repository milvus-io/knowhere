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
#include <memory>
#include <string>
#include <vector>

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

static const std::vector<std::string> default_metrics_array{Metric::L2, Metric::IP};
static const std::vector<std::string> default_binary_metrics_array{Metric::HAMMING, Metric::JACCARD, Metric::TANIMOTO,
                                                                   Metric::SUBSTRUCTURE, Metric::SUPERSTRUCTURE};
inline bool
CheckIntByRange(const Config& cfg, const std::string& key, int64_t min, int64_t max) {
    return (cfg.contains(key) && cfg[key].is_number_integer() && cfg[key].get<int64_t>() >= min &&
            cfg[key].get<int64_t>() <= max);
}

inline bool
CheckFloatByRange(const Config& cfg, const std::string& key, int64_t min, int64_t max) {
    return (cfg.contains(key) && cfg[key].is_number_float() && cfg[key].get<float>() >= min &&
            cfg[key].get<float>() <= max);
}

inline bool
CheckStrByValues(const Config& cfg, const std::string& key, const std::vector<std::string>& container) {
    return (cfg.contains(key) && cfg[key].is_string() &&
            std::find(container.begin(), container.end(), cfg[key].get<std::string>()) != container.end());
}

bool
ConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM)) {
        return false;
    }
    if (!CheckStrByValues(cfg, Metric::TYPE, default_metrics_array)) {
        return false;
    }
    return true;
}

bool
ConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    const int64_t DEFAULT_MIN_K = 1;
    const int64_t DEFAULT_MAX_K = 16384;
    return CheckIntByRange(cfg, meta::TOPK, DEFAULT_MIN_K - 1, DEFAULT_MAX_K);
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
    if (!CheckIntByRange(cfg, IndexParams::nlist, MIN_NLIST, MAX_NLIST)) {
        return false;
    }

    // auto tune params
    auto rows = cfg[meta::ROWS].get<int64_t>();
    auto nlist = cfg[IndexParams::nlist].get<int64_t>();
    cfg[IndexParams::nlist] = MatchNlist(rows, nlist);

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
    if (!CheckIntByRange(cfg, IndexParams::nprobe, MIN_NPROBE, max_nprobe)) {
        return false;
    }

    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
IVFSQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    cfg[IndexParams::nbits] = DEFAULT_NBITS;
    return IVFConfAdapter::CheckTrain(cfg, mode);
}

bool
IVFPQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!IVFConfAdapter::CheckTrain(cfg, mode)) {
        return false;
    }

    if (!CheckIntByRange(cfg, IndexParams::nbits, MIN_NBITS, MAX_NBITS)) {
        return false;
    }

    auto rows = cfg[meta::ROWS].get<int64_t>();
    auto nbits = cfg.count(IndexParams::nbits) ? cfg[IndexParams::nbits].get<int64_t>() : DEFAULT_NBITS;
    cfg[IndexParams::nbits] = MatchNbits(rows, nbits);

    auto m = cfg[IndexParams::m].get<int64_t>();
    auto dimension = cfg[meta::DIM].get<int64_t>();

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
    if (!CheckIntByRange(cfg, IndexParams::efConstruction, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::M, HNSW_MIN_M, HNSW_MAX_M)) {
        return false;
    }

    // IVF param check
    if (!CheckIntByRange(cfg, IndexParams::nlist, MIN_NLIST, MAX_NLIST)) {
        return false;
    }

    // auto tune params
    auto rows = cfg[meta::ROWS].get<int64_t>();
    auto nlist = cfg[IndexParams::nlist].get<int64_t>();
    cfg[IndexParams::nlist] = MatchNlist(rows, nlist);

    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
IVFHNSWConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    // HNSW param check
    if (!CheckIntByRange(cfg, IndexParams::ef, cfg[meta::TOPK], HNSW_MAX_EF)) {
        return false;
    }

    // IVF param check
    if (!CheckIntByRange(cfg, IndexParams::nprobe, MIN_NPROBE, MAX_NPROBE)) {
        return false;
    }

    return ConfAdapter::CheckSearch(cfg, type, mode);
}

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

    if (!CheckStrByValues(cfg, Metric::TYPE, default_metrics_array)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::knng, MIN_KNNG, MAX_KNNG)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::search_length, MIN_SEARCH_LENGTH, MAX_SEARCH_LENGTH)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::out_degree, MIN_OUT_DEGREE, MAX_OUT_DEGREE)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::candidate, MIN_CANDIDATE_POOL_SIZE, MAX_CANDIDATE_POOL_SIZE)) {
        return false;
    }

    // auto tune params
    cfg[IndexParams::nlist] = MatchNlist(cfg[meta::ROWS].get<int64_t>(), 8192);

    int64_t nprobe = int(cfg[IndexParams::nlist].get<int64_t>() * 0.1);
    cfg[IndexParams::nprobe] = nprobe < 1 ? 1 : nprobe;

    return true;
}

bool
NSGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    static int64_t MIN_SEARCH_LENGTH = 1;
    static int64_t MAX_SEARCH_LENGTH = 300;

    if (!CheckIntByRange(cfg, IndexParams::search_length, MIN_SEARCH_LENGTH, MAX_SEARCH_LENGTH)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
HNSWConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::efConstruction, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::M, HNSW_MIN_M, HNSW_MAX_M)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
HNSWConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::ef, cfg[meta::TOPK], HNSW_MAX_EF)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWFlatConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::efConstruction, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::M, HNSW_MIN_M, HNSW_MAX_M)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWFlatConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::ef, cfg[meta::TOPK], HNSW_MAX_EF)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWPQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::efConstruction, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::M, HNSW_MIN_M, HNSW_MAX_M)) {
        return false;
    }

    auto dimension = cfg[meta::DIM].get<int64_t>();
    if (!IVFPQConfAdapter::CheckCPUPQParams(dimension, cfg[IndexParams::PQM].get<int64_t>())) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWPQConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::ef, cfg[meta::TOPK], HNSW_MAX_EF)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
RHNSWSQConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::efConstruction, HNSW_MIN_EFCONSTRUCTION, HNSW_MAX_EFCONSTRUCTION)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::M, HNSW_MIN_M, HNSW_MAX_M)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
RHNSWSQConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::ef, cfg[meta::TOPK], HNSW_MAX_EF)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
BinIDMAPConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM)) {
        return false;
    }
    if (!CheckStrByValues(cfg, Metric::TYPE, default_binary_metrics_array)) {
        return false;
    }
    return true;
}

bool
BinIVFConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    static const std::vector<std::string> metrics_array{Metric::HAMMING, Metric::JACCARD, Metric::TANIMOTO};

    if (!CheckIntByRange(cfg, meta::DIM, DEFAULT_MIN_DIM, DEFAULT_MAX_DIM)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::nlist, MIN_NLIST, MAX_NLIST)) {
        return false;
    }
    if (!CheckStrByValues(cfg, Metric::TYPE, metrics_array)) {
        return false;
    }

    // auto tune params
    auto rows = cfg[meta::ROWS].get<int64_t>();
    auto nlist = cfg[IndexParams::nlist].get<int64_t>();
    cfg[IndexParams::nlist] = MatchNlist(rows, nlist);

    return true;
}

bool
ANNOYConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    static int64_t MIN_NTREES = 1;
    // too large of n_trees takes much time, if there is real requirement, change this threshold.
    static int64_t MAX_NTREES = 1024;

    if (!CheckIntByRange(cfg, IndexParams::n_trees, MIN_NTREES, MAX_NTREES)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
ANNOYConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    static int64_t MIN_SEARCH_K = std::numeric_limits<int64_t>::min();
    static int64_t MAX_SEARCH_K = std::numeric_limits<int64_t>::max();
    if (!CheckIntByRange(cfg, IndexParams::search_k, MIN_SEARCH_K, MAX_SEARCH_K)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
NGTPANNGConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::forcedly_pruned_edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::selectively_pruned_edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (cfg[IndexParams::selectively_pruned_edge_size].get<int64_t>() >=
        cfg[IndexParams::forcedly_pruned_edge_size].get<int64_t>()) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
NGTPANNGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::max_search_edges, -1, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckFloatByRange(cfg, IndexParams::epsilon, -1.0, 1.0)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

bool
NGTONNGConfAdapter::CheckTrain(Config& cfg, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::outgoing_edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckIntByRange(cfg, IndexParams::incoming_edge_size, NGT_MIN_EDGE_SIZE, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    return ConfAdapter::CheckTrain(cfg, mode);
}

bool
NGTONNGConfAdapter::CheckSearch(Config& cfg, const IndexType type, const IndexMode mode) {
    if (!CheckIntByRange(cfg, IndexParams::max_search_edges, -1, NGT_MAX_EDGE_SIZE)) {
        return false;
    }
    if (!CheckFloatByRange(cfg, IndexParams::epsilon, -1.0, 1.0)) {
        return false;
    }
    return ConfAdapter::CheckSearch(cfg, type, mode);
}

}  // namespace knowhere
