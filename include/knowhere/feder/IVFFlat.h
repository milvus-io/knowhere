// Copyright (C) 2019-2023 Zilliz. All rights reserved.
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

namespace knowhere::feder::ivfflat {

///////////////////////////////////////////////////////////////////////////////
// index view
struct ClusterInfo {
    int64_t id_;
    std::vector<int64_t> node_ids_;
    std::vector<float> centroid_vec_;

    ClusterInfo() = default;

    explicit ClusterInfo(const int64_t id, const int64_t* node_id_addr, const int64_t node_num,
                         const float* centroid_addr, const int64_t dim)
        : id_(id) {
        node_ids_.resize(node_num);
        centroid_vec_.resize(dim);
        std::copy_n(node_id_addr, node_num, node_ids_.data());
        std::copy_n(centroid_addr, dim, centroid_vec_.data());
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ClusterInfo, id_, node_ids_, centroid_vec_)
};

class IVFFlatMeta {
 public:
    IVFFlatMeta() = default;

    explicit IVFFlatMeta(int64_t nlist, int64_t dim, int64_t ntotal) : nlist_(nlist), dim_(dim), ntotal_(ntotal) {
    }

    int64_t
    GetNlist() {
        return nlist_;
    }

    int64_t
    GetDim() {
        return dim_;
    }

    int64_t
    GetNtotal() {
        return ntotal_;
    }

    const std::vector<ClusterInfo>&
    GetClusters() const {
        return clusters_;
    }

    void
    AddCluster(const int64_t id, const int64_t* node_id_addr, const int64_t node_num, const float* centroid_addr,
               const int64_t dim) {
        assert(dim == dim_);
        clusters_.emplace_back(ClusterInfo(id, node_id_addr, node_num, centroid_addr, dim));
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(IVFFlatMeta, nlist_, dim_, ntotal_, clusters_);

 private:
    int64_t nlist_;
    int64_t dim_;
    int64_t ntotal_;
    std::vector<ClusterInfo> clusters_;
};

}  // namespace knowhere::feder::ivfflat
