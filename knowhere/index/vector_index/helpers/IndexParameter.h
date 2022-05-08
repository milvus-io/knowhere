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

#include "knowhere/common/Config.h"
#include "knowhere/common/MetricType.h"

namespace knowhere {

using MetaType = std::string_view;

namespace meta {
constexpr MetaType METRIC_TYPE = "metric_type";
constexpr MetaType DIM = "dim";
constexpr MetaType TENSOR = "tensor";
constexpr MetaType ROWS = "rows";
constexpr MetaType IDS = "ids";
constexpr MetaType DISTANCE = "distance";
constexpr MetaType LIMS = "lims";
constexpr MetaType TOPK = "k";
constexpr MetaType RADIUS = "radius";
constexpr MetaType DEVICE_ID = "gpu_id";
};  // namespace meta

using IndexParamType = std::string_view;

namespace indexparam {
// IVF Params
constexpr IndexParamType NPROBE = "nprobe";
constexpr IndexParamType NLIST = "nlist";
constexpr IndexParamType NBITS = "nbits";   // PQ/SQ
constexpr IndexParamType M = "m";           // PQ param for IVFPQ
constexpr IndexParamType PQ_M = "PQM";      // PQ param for RHNSWPQ
// HNSW Params
constexpr IndexParamType EFCONSTRUCTION = "efConstruction";
constexpr IndexParamType HNSW_M = "M";
constexpr IndexParamType EF = "ef";
// Annoy Params
constexpr IndexParamType N_TREES = "n_trees";
constexpr IndexParamType SEARCH_K = "search_k";
#ifdef KNOWHERE_SUPPORT_NGT
// NGT Params
constexpr IndexParamType EDGE_SIZE = "edge_size";
// NGT Search Params
constexpr IndexParamType EPSILON = "epsilon";
constexpr IndexParamType MAX_SEARCH_EDGES = "max_search_edges";
// NGT_PANNG Params
constexpr IndexParamType FORCEDLY_PRUNED_EDGE_SIZE = "forcedly_pruned_edge_size";
constexpr IndexParamType SELECTIVELY_PRUNED_EDGE_SIZE = "selectively_pruned_edge_size";
// NGT_ONNG Params
constexpr IndexParamType OUTGOING_EDGE_SIZE = "outgoing_edge_size";
constexpr IndexParamType INCOMING_EDGE_SIZE = "incoming_edge_size";
#endif
#ifdef KNOWHERE_SUPPORT_NSG
// NSG Params
constexpr IndexParamType KNNG = "knng";
constexpr IndexParamType SEARCH_LENGTH = "search_length";
constexpr IndexParamType OUT_DEGREE = "out_degree";
constexpr IndexParamType CANDIDATE = "candidate_pool_size";
#endif
}  // namespace indexparam

using MetricType = std::string_view;

namespace metric {
constexpr MetricType IP = "IP";
constexpr MetricType L2 = "L2";
constexpr MetricType HAMMING = "HAMMING";
constexpr MetricType JACCARD = "JACCARD";
constexpr MetricType TANIMOTO = "TANIMOTO";
constexpr MetricType SUBSTRUCTURE = "SUBSTRUCTURE";
constexpr MetricType SUPERSTRUCTURE = "SUPERSTRUCTURE";
}  // namespace metric

///////////////////////////////////////////////////////////////////////////////
template<typename T>
T
GetValueFromConfig(const Config& cfg, const std::string& key) {
    return cfg.at(key).get<T>();
}

template<typename T>
void
SetValueToConfig(Config& cfg, const std::string& key, const T value) {
    cfg[key] = value;
}

#define DEFINE_GETTER(func_name, key, T)                    \
inline T func_name(const Config& cfg) {                     \
    return GetValueFromConfig<T>(cfg, std::string(key));    \
}

#define DEFINE_SETTER(func_name, key, T1, T2)               \
inline void func_name(Config& cfg, T1 value) {              \
    SetValueToConfig<T2>(cfg, std::string(key), T2(value)); \
}

///////////////////////////////////////////////////////////////////////////////
// APIs to access meta

DEFINE_GETTER(GetMetaMetricType, meta::METRIC_TYPE, std::string)
DEFINE_SETTER(SetMetaMetricType, meta::METRIC_TYPE, MetricType , std::string)

DEFINE_GETTER(GetMetaRows, meta::ROWS, int64_t)
DEFINE_SETTER(SetMetaRows, meta::ROWS, int64_t, int64_t)

DEFINE_GETTER(GetMetaDim, meta::DIM, int64_t)
DEFINE_SETTER(SetMetaDim, meta::DIM, int64_t, int64_t)

DEFINE_GETTER(GetMetaTopk, meta::TOPK, int64_t)
DEFINE_SETTER(SetMetaTopk, meta::TOPK, int64_t, int64_t)

DEFINE_GETTER(GetMetaRadius, meta::RADIUS, float)
DEFINE_SETTER(SetMetaRadius, meta::RADIUS, float, float)

DEFINE_GETTER(GetMetaDeviceID, meta::DEVICE_ID, int64_t)
DEFINE_SETTER(SetMetaDeviceID, meta::DEVICE_ID, int64_t , int64_t)

///////////////////////////////////////////////////////////////////////////////
// APIs to access indexparam

DEFINE_GETTER(GetIndexParamNprobe, indexparam::NPROBE, int64_t)
DEFINE_SETTER(SetIndexParamNprobe, indexparam::NPROBE, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamNlist, indexparam::NLIST, int64_t)
DEFINE_SETTER(SetIndexParamNlist, indexparam::NLIST, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamNbits, indexparam::NBITS, int64_t)
DEFINE_SETTER(SetIndexParamNbits, indexparam::NBITS, int64_t, int64_t)

// PQ param for IVFPQ
DEFINE_GETTER(GetIndexParamM, indexparam::M, int64_t)
DEFINE_SETTER(SetIndexParamM, indexparam::M, int64_t, int64_t)

// PQ param for RHNSWPQ
DEFINE_GETTER(GetIndexParamPQM, indexparam::PQ_M, int64_t)
DEFINE_SETTER(SetIndexParamPQM, indexparam::PQ_M, int64_t, int64_t)

// HNSW Params
DEFINE_GETTER(GetIndexParamEfConstruction, indexparam::EFCONSTRUCTION, int64_t)
DEFINE_SETTER(SetIndexParamEfConstruction, indexparam::EFCONSTRUCTION, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamHNSWM, indexparam::HNSW_M, int64_t)
DEFINE_SETTER(SetIndexParamHNSWM, indexparam::HNSW_M, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamEf, indexparam::EF, int64_t)
DEFINE_SETTER(SetIndexParamEf, indexparam::EF, int64_t, int64_t)

// Annoy Params
DEFINE_GETTER(GetIndexParamNtrees, indexparam::N_TREES, int64_t)
DEFINE_SETTER(SetIndexParamNtrees, indexparam::N_TREES, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamSearchK, indexparam::SEARCH_K, int64_t)
DEFINE_SETTER(SetIndexParamSearchK, indexparam::SEARCH_K, int64_t, int64_t)

///////////////////////////////////////////////////////////////////////////////
// other
faiss::MetricType GetMetricType(const std::string& type);

inline faiss::MetricType
GetMetricType(const Config& cfg) {
    return GetMetricType(GetMetaMetricType(cfg));
}

}  // namespace knowhere
