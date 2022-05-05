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

namespace IndexParams {

// IVF Params
constexpr const char* nprobe = "nprobe";
constexpr const char* nlist = "nlist";
constexpr const char* m = "m";          // PQ
constexpr const char* nbits = "nbits";  // PQ/SQ

// NSG Params
constexpr const char* knng = "knng";
constexpr const char* search_length = "search_length";
constexpr const char* out_degree = "out_degree";
constexpr const char* candidate = "candidate_pool_size";

// HNSW Params
constexpr const char* efConstruction = "efConstruction";
constexpr const char* M = "M";
constexpr const char* ef = "ef";

// Annoy Params
constexpr const char* n_trees = "n_trees";
constexpr const char* search_k = "search_k";

// PQ Params
constexpr const char* PQM = "PQM";

// NGT Params
constexpr const char* edge_size = "edge_size";
// NGT Search Params
constexpr const char* epsilon = "epsilon";
constexpr const char* max_search_edges = "max_search_edges";
// NGT_PANNG Params
constexpr const char* forcedly_pruned_edge_size = "forcedly_pruned_edge_size";
constexpr const char* selectively_pruned_edge_size = "selectively_pruned_edge_size";
// NGT_ONNG Params
constexpr const char* outgoing_edge_size = "outgoing_edge_size";
constexpr const char* incoming_edge_size = "incoming_edge_size";
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

faiss::MetricType
GetMetricType(const Config& cfg);

inline std::string
GetMetaMetricType(const Config& cfg) {
    return GetValueFromConfig<std::string>(cfg, std::string(meta::METRIC_TYPE));
}

inline void
SetMetaMetricType(Config& cfg, MetricType value) {
    SetValueToConfig<std::string>(cfg, std::string(meta::METRIC_TYPE), std::string(value));
}

inline int64_t
GetMetaRows(const Config& cfg) {
    return GetValueFromConfig<int64_t>(cfg, std::string(meta::ROWS));
}

inline void
SetMetaRows(Config& cfg, int64_t value) {
    SetValueToConfig<int64_t>(cfg, std::string(meta::ROWS), value);
}

inline int64_t
GetMetaDim(const Config& cfg) {
    return GetValueFromConfig<int64_t>(cfg, std::string(meta::DIM));
}

inline void
SetMetaDim(Config& cfg, int64_t value) {
    SetValueToConfig<int64_t>(cfg, std::string(meta::DIM), value);
}

inline int64_t
GetMetaTopk(const Config& cfg) {
    return GetValueFromConfig<int64_t>(cfg, std::string(meta::TOPK));
}

inline void
SetMetaTopk(Config& cfg, int64_t value) {
    SetValueToConfig<int64_t>(cfg, std::string(meta::TOPK), value);
}

inline float
GetMetaRadius(const Config& cfg) {
    return GetValueFromConfig<float>(cfg, std::string(meta::RADIUS));
}

inline void
SetMetaRadius(Config& cfg, float value) {
    SetValueToConfig<float>(cfg, std::string(meta::RADIUS), value);
}

inline int64_t
GetMetaDeviceID(const Config& cfg) {
    return GetValueFromConfig<int64_t>(cfg, std::string(meta::DEVICE_ID));
}

inline void
SetMetaDeviceID(Config& cfg, int64_t value) {
    SetValueToConfig<int64_t>(cfg, std::string(meta::DEVICE_ID), value);
}

}  // namespace knowhere
