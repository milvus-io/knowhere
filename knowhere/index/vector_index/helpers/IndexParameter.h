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

#include <algorithm>
#include <string>
#include <unordered_map>

#include "faiss/MetricType.h"
#include "knowhere/common/Config.h"
#include "knowhere/common/Exception.h"

namespace knowhere {

namespace meta {
constexpr const char* SLICE_SIZE = "SLICE_SIZE";
constexpr const char* METRIC_TYPE = "metric_type";
constexpr const char* DIM = "dim";
constexpr const char* TENSOR = "tensor";
constexpr const char* ROWS = "rows";
constexpr const char* IDS = "ids";
constexpr const char* DISTANCE = "distance";
constexpr const char* LIMS = "lims";
constexpr const char* TOPK = "k";
constexpr const char* RADIUS = "radius";
constexpr const char* INPUT_IDS = "input_ids";
constexpr const char* OUTPUT_TENSOR = "output_tensor";
constexpr const char* DEVICE_ID = "gpu_id";
constexpr const char* BUILD_THREAD_NUM = "build_thread_num";
constexpr const char* QUERY_THREAD_NUM = "query_thread_num";
};  // namespace meta

namespace indexparam {
// IVF Params
constexpr const char* NPROBE = "nprobe";
constexpr const char* NLIST = "nlist";
constexpr const char* NBITS = "nbits";   // PQ/SQ
constexpr const char* M = "m";           // PQ param for IVFPQ
constexpr const char* PQ_M = "PQM";      // PQ param for RHNSWPQ
// HNSW Params
constexpr const char* EFCONSTRUCTION = "efConstruction";
constexpr const char* HNSW_M = "M";
constexpr const char* EF = "ef";
constexpr const char* HNSW_K = "range_k";
// Annoy Params
constexpr const char* N_TREES = "n_trees";
constexpr const char* SEARCH_K = "search_k";
#ifdef KNOWHERE_SUPPORT_NGT
// NGT Params
constexpr const char* EDGE_SIZE = "edge_size";
// NGT Search Params
constexpr const char* EPSILON = "epsilon";
constexpr const char* MAX_SEARCH_EDGES = "max_search_edges";
// NGT_PANNG Params
constexpr const char* FORCEDLY_PRUNED_EDGE_SIZE = "forcedly_pruned_edge_size";
constexpr const char* SELECTIVELY_PRUNED_EDGE_SIZE = "selectively_pruned_edge_size";
// NGT_ONNG Params
constexpr const char* OUTGOING_EDGE_SIZE = "outgoing_edge_size";
constexpr const char* INCOMING_EDGE_SIZE = "incoming_edge_size";
#endif
#ifdef KNOWHERE_SUPPORT_NSG
// NSG Params
constexpr const char* KNNG = "knng";
constexpr const char* SEARCH_LENGTH = "search_length";
constexpr const char* OUT_DEGREE = "out_degree";
constexpr const char* CANDIDATE = "candidate_pool_size";
#endif
}  // namespace indexparam

using MetricType = std::string;

namespace metric {
constexpr const char* IP = "IP";
constexpr const char* L2 = "L2";
constexpr const char* HAMMING = "HAMMING";
constexpr const char* JACCARD = "JACCARD";
constexpr const char* TANIMOTO = "TANIMOTO";
constexpr const char* SUBSTRUCTURE = "SUBSTRUCTURE";
constexpr const char* SUPERSTRUCTURE = "SUPERSTRUCTURE";
}  // namespace metric

///////////////////////////////////////////////////////////////////////////////
inline bool
CheckKeyInConfig(const Config& cfg, const std::string& key) {
    return cfg.contains(key);
}

template <typename T>
inline T
GetValueFromConfig(const Config& cfg, const std::string& key) {
    return cfg.at(key).get<T>();
}

template <typename T>
inline void
SetValueToConfig(Config& cfg, const std::string& key, const T value) {
    cfg[key] = value;
}

#define DEFINE_CONFIG_GETTER(func_name, key, T) \
inline T func_name(const Config& cfg) {         \
    return GetValueFromConfig<T>(cfg, key);     \
}

#define DEFINE_CONFIG_SETTER(func_name, key, T) \
inline void func_name(Config& cfg, T value) {   \
    SetValueToConfig<T>(cfg, key, (T)(value));  \
}

///////////////////////////////////////////////////////////////////////////////
// APIs to access meta
DEFINE_CONFIG_GETTER(GetMetaSliceSize, meta::SLICE_SIZE, int64_t)
DEFINE_CONFIG_SETTER(SetMetaSliceSize, meta::SLICE_SIZE, int64_t)

DEFINE_CONFIG_GETTER(GetMetaMetricType, meta::METRIC_TYPE, std::string)
DEFINE_CONFIG_SETTER(SetMetaMetricType, meta::METRIC_TYPE, std::string)

DEFINE_CONFIG_GETTER(GetMetaRows, meta::ROWS, int64_t)
DEFINE_CONFIG_SETTER(SetMetaRows, meta::ROWS, int64_t)

DEFINE_CONFIG_GETTER(GetMetaDim, meta::DIM, int64_t)
DEFINE_CONFIG_SETTER(SetMetaDim, meta::DIM, int64_t)

DEFINE_CONFIG_GETTER(GetMetaTopk, meta::TOPK, int64_t)
DEFINE_CONFIG_SETTER(SetMetaTopk, meta::TOPK, int64_t)

DEFINE_CONFIG_GETTER(GetMetaRadius, meta::RADIUS, float)
DEFINE_CONFIG_SETTER(SetMetaRadius, meta::RADIUS, float)

DEFINE_CONFIG_GETTER(GetMetaDeviceID, meta::DEVICE_ID, int64_t)
DEFINE_CONFIG_SETTER(SetMetaDeviceID, meta::DEVICE_ID, int64_t)

DEFINE_CONFIG_GETTER(GetMetaBuildThreadNum, meta::BUILD_THREAD_NUM, int64_t)
DEFINE_CONFIG_SETTER(SetMetaBuildThreadNum, meta::BUILD_THREAD_NUM, int64_t)

DEFINE_CONFIG_GETTER(GetMetaQueryThreadNum, meta::QUERY_THREAD_NUM, int64_t)
DEFINE_CONFIG_SETTER(SetMetaQueryThreadNum, meta::QUERY_THREAD_NUM, int64_t)

///////////////////////////////////////////////////////////////////////////////
// APIs to access indexparam

DEFINE_CONFIG_GETTER(GetIndexParamNprobe, indexparam::NPROBE, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamNprobe, indexparam::NPROBE, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamNlist, indexparam::NLIST, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamNlist, indexparam::NLIST, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamNbits, indexparam::NBITS, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamNbits, indexparam::NBITS, int64_t)

// PQ param for IVFPQ
DEFINE_CONFIG_GETTER(GetIndexParamM, indexparam::M, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamM, indexparam::M, int64_t)

// PQ param for RHNSWPQ
DEFINE_CONFIG_GETTER(GetIndexParamPQM, indexparam::PQ_M, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamPQM, indexparam::PQ_M, int64_t)

// HNSW Params
DEFINE_CONFIG_GETTER(GetIndexParamEfConstruction, indexparam::EFCONSTRUCTION, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamEfConstruction, indexparam::EFCONSTRUCTION, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamHNSWM, indexparam::HNSW_M, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamHNSWM, indexparam::HNSW_M, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamEf, indexparam::EF, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamEf, indexparam::EF, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamHNSWK, indexparam::HNSW_K, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamHNSWK, indexparam::HNSW_K, int64_t)

// Annoy Params
DEFINE_CONFIG_GETTER(GetIndexParamNtrees, indexparam::N_TREES, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamNtrees, indexparam::N_TREES, int64_t)

DEFINE_CONFIG_GETTER(GetIndexParamSearchK, indexparam::SEARCH_K, int64_t)
DEFINE_CONFIG_SETTER(SetIndexParamSearchK, indexparam::SEARCH_K, int64_t)

///////////////////////////////////////////////////////////////////////////////
// other
static const std::unordered_map<knowhere::MetricType, faiss::MetricType> metric_map = {
    {metric::L2, faiss::MetricType::METRIC_L2},
    {metric::IP, faiss::MetricType::METRIC_INNER_PRODUCT},
    {metric::JACCARD, faiss::MetricType::METRIC_Jaccard},
    {metric::TANIMOTO, faiss::MetricType::METRIC_Tanimoto},
    {metric::HAMMING, faiss::MetricType::METRIC_Hamming},
    {metric::SUBSTRUCTURE, faiss::MetricType::METRIC_Substructure},
    {metric::SUPERSTRUCTURE, faiss::MetricType::METRIC_Superstructure},
};

inline faiss::MetricType
GetFaissMetricType(const MetricType& type) {
    try {
        std::string type_str = type;
        std::transform(type_str.begin(), type_str.end(), type_str.begin(), toupper);
        return metric_map.at(type_str);
    } catch (...) {
        KNOWHERE_THROW_FORMAT("Metric type '%s' invalid", type.data());
    }
}

inline faiss::MetricType
GetFaissMetricType(const Config& cfg) {
    return GetFaissMetricType(GetMetaMetricType(cfg));
}

}  // namespace knowhere
