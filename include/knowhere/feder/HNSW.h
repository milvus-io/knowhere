// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace knowhere::feder::hnsw {

///////////////////////////////////////////////////////////////////////////////
// index view
struct NodeInfo {
    int64_t id_;
    std::vector<int64_t> neighbors_;

    NodeInfo() = default;

    explicit NodeInfo(const int64_t id, std::vector<int64_t>&& neighbors) : id_(id), neighbors_(std::move(neighbors)) {
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(NodeInfo, id_, neighbors_)
};

class LevelLinkGraph {
 public:
    LevelLinkGraph() = default;

    explicit LevelLinkGraph(int64_t level) : level_(level) {
    }

    int64_t
    GetLevel() {
        return level_;
    }

    const std::vector<NodeInfo>&
    GetNodes() const {
        return nodes_;
    }

    void
    AddNodeInfo(int64_t id, std::vector<int64_t>&& links) {
        nodes_.emplace_back(NodeInfo(id, std::move(links)));
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LevelLinkGraph, level_, nodes_)

 private:
    int64_t level_;
    std::vector<NodeInfo> nodes_;
};

class HNSWMeta {
 public:
    HNSWMeta() = default;

    explicit HNSWMeta(int64_t ef_construction, int64_t M, int64_t num_elem, int64_t num_levels, int64_t enter_point_id,
                      int64_t num_overview_levels)
        : ef_construction_(ef_construction),
          M_(M),
          num_elem_(num_elem),
          num_levels_(num_levels),
          enter_point_id_(enter_point_id),
          num_overview_levels_(num_overview_levels) {
    }

    int64_t
    GetEfConstruction() {
        return ef_construction_;
    }

    int64_t
    GetM() {
        return M_;
    }

    int64_t
    GetNumElem() {
        return num_elem_;
    }

    int64_t
    GetNumLevel() {
        return num_levels_;
    }

    int64_t
    GetEnterPointId() {
        return enter_point_id_;
    }

    int64_t
    GetNumOverviewLevels() {
        return num_overview_levels_;
    }

    const std::vector<LevelLinkGraph>&
    GetOverviewHierGraph() const {
        return overview_hier_graph_;
    }

    void
    AddLevelLinkGraph(int64_t level) {
        overview_hier_graph_.push_back(LevelLinkGraph(level));
    }

    void
    AddNodeInfo(int64_t level, int64_t id, std::vector<int64_t>&& links) {
        auto& curr = overview_hier_graph_.back();
        assert(curr.GetLevel() == level);
        curr.AddNodeInfo(id, std::move(links));
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(HNSWMeta, ef_construction_, M_, num_elem_, num_levels_, enter_point_id_,
                                   num_overview_levels_, overview_hier_graph_);

 private:
    int64_t ef_construction_;
    int64_t M_;

    int64_t num_elem_;
    int64_t num_levels_;

    int64_t enter_point_id_;
    int64_t num_overview_levels_;

    std::vector<LevelLinkGraph> overview_hier_graph_;
};

///////////////////////////////////////////////////////////////////////////////
// search view
class LevelVisitRecord {
 public:
    LevelVisitRecord() = default;

    explicit LevelVisitRecord(int64_t level) : level_(level) {
    }

    int64_t
    GetLevel() {
        return level_;
    }

    const std::vector<std::tuple<int64_t, int64_t, float>>&
    GetRecords() const {
        return records_;
    }

    void
    AddVisitRecord(int64_t id_from, int64_t id_to, float distance) {
        records_.emplace_back(std::make_tuple(id_from, id_to, distance));
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LevelVisitRecord, level_, records_)

 private:
    int64_t level_;
    std::vector<std::tuple<int64_t, int64_t, float>> records_;
};

class HNSWVisitInfo {
 public:
    HNSWVisitInfo() = default;

    const std::vector<LevelVisitRecord>&
    GetInfos() const {
        return infos_;
    }

    void
    AddLevelVisitRecord(int64_t level) {
        infos_.emplace_back(LevelVisitRecord(level));
    }

    void
    AddVisitRecord(int64_t level, int64_t id_from, int64_t id_to, float dist) {
        auto& curr = infos_.back();
        assert(curr.GetLevel() == level);
        curr.AddVisitRecord(id_from, id_to, dist);
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(HNSWVisitInfo, infos_)

 private:
    std::vector<LevelVisitRecord> infos_;
};

struct FederResult {
    HNSWVisitInfo visit_info_;
    std::unordered_set<int64_t> id_set_;
};
using FederResultUniq = std::unique_ptr<FederResult>;

}  // namespace knowhere::feder::hnsw
