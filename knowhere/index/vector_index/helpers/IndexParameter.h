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

namespace IndexParams {
// IVF Params
constexpr IndexParamType nprobe = "nprobe";
constexpr IndexParamType nlist = "nlist";
constexpr IndexParamType PQM = "pqm";      // PQ
constexpr IndexParamType nbits = "nbits";  // PQ/SQ
// HNSW Params
constexpr IndexParamType efConstruction = "efConstruction";
constexpr IndexParamType M = "M";
constexpr IndexParamType ef = "ef";
// Annoy Params
constexpr IndexParamType n_trees = "n_trees";
constexpr IndexParamType search_k = "search_k";
#ifdef KNOWHERE_SUPPORT_NGT
// NGT Params
constexpr IndexParamType edge_size = "edge_size";
// NGT Search Params
constexpr IndexParamType epsilon = "epsilon";
constexpr IndexParamType max_search_edges = "max_search_edges";
// NGT_PANNG Params
constexpr IndexParamType forcedly_pruned_edge_size = "forcedly_pruned_edge_size";
constexpr IndexParamType selectively_pruned_edge_size = "selectively_pruned_edge_size";
// NGT_ONNG Params
constexpr IndexParamType outgoing_edge_size = "outgoing_edge_size";
constexpr IndexParamType incoming_edge_size = "incoming_edge_size";
#endif
#ifdef KNOWHERE_SUPPORT_NSG
// NSG Params
constexpr IndexParamType knng = "knng";
constexpr IndexParamType search_length = "search_length";
constexpr IndexParamType out_degree = "out_degree";
constexpr IndexParamType candidate = "candidate_pool_size";
#endif
}  // namespace IndexParams

using MetricType = std::string_view;

namespace MetricEnum {
constexpr MetricType IP = "IP";
constexpr MetricType L2 = "L2";
constexpr MetricType HAMMING = "HAMMING";
constexpr MetricType JACCARD = "JACCARD";
constexpr MetricType TANIMOTO = "TANIMOTO";
constexpr MetricType SUBSTRUCTURE = "SUBSTRUCTURE";
constexpr MetricType SUPERSTRUCTURE = "SUPERSTRUCTURE";
}  // namespace MetricEnum

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

faiss::MetricType GetMetricType(const Config& cfg);

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
// APIs to access IndexParams

DEFINE_GETTER(GetIndexParamNprobe, IndexParams::nprobe, int64_t)
DEFINE_SETTER(SetIndexParamNprobe, IndexParams::nprobe, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamNlist, IndexParams::nlist, int64_t)
DEFINE_SETTER(SetIndexParamNlist, IndexParams::nlist, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamPQM, IndexParams::PQM, int64_t)
DEFINE_SETTER(SetIndexParamPQM, IndexParams::PQM, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamNbits, IndexParams::nbits, int64_t)
DEFINE_SETTER(SetIndexParamNbits, IndexParams::nbits, int64_t, int64_t)

// HNSW Params
DEFINE_GETTER(GetIndexParamEfConstruction, IndexParams::efConstruction, int64_t)
DEFINE_SETTER(SetIndexParamEfConstruction, IndexParams::efConstruction, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamM, IndexParams::M, int64_t)
DEFINE_SETTER(SetIndexParamM, IndexParams::M, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamEf, IndexParams::ef, int64_t)
DEFINE_SETTER(SetIndexParamEf, IndexParams::ef, int64_t, int64_t)

// Annoy Params
DEFINE_GETTER(GetIndexParamNtrees, IndexParams::n_trees, int64_t)
DEFINE_SETTER(SetIndexParamNtrees, IndexParams::n_trees, int64_t, int64_t)

DEFINE_GETTER(GetIndexParamSearchK, IndexParams::search_k, int64_t)
DEFINE_SETTER(SetIndexParamSearchK, IndexParams::search_k, int64_t, int64_t)

}  // namespace knowhere
