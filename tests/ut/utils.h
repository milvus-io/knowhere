// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "common/range_util.h"
#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"

namespace {
using IdDisPair = std::pair<int64_t, float>;
struct DisPairLess {
    bool
    operator()(const IdDisPair& p1, const IdDisPair& p2) {
        return p1.second < p2.second;
    }
};
};  // namespace
inline std::unique_ptr<knowhere::DataSet>
GenDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(0.0, 100.0);

    auto ds = std::make_unique<knowhere::DataSet>();
    ds->SetRows(rows);
    ds->SetDim(dim);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
    ds->SetTensor(ts);
    return ds;
}

inline std::unique_ptr<knowhere::DataSet>
GetKNNGroundTruth(const knowhere::DataSet& base, const knowhere::DataSet& query, const std::string& metric,
                  const int topk, const knowhere::BitsetView bitset = nullptr) {
    auto nb = base.GetRows();
    auto nq = query.GetRows();
    auto ndim = base.GetDim();
    auto base_p = reinterpret_cast<const float*>(base.GetTensor());
    auto query_p = reinterpret_cast<const float*>(query.GetTensor());

    float* dis = new float[nq * topk];
    int64_t* ids = new int64_t[nq * topk];
    for (uint32_t query_index = 0; query_index < nq; ++query_index) {
        std::priority_queue<IdDisPair, std::vector<IdDisPair>, DisPairLess> pq;
        for (int64_t row = 0; row < nb; ++row) {
            if (!bitset.empty() && bitset.test(row)) {
                continue;
            }
            float dis = 0;
            for (uint32_t dim = 0; dim < ndim; ++dim) {  // for every dim
                if (metric == knowhere::metric::IP) {
                    dis -= (base_p[ndim * row + dim] * query_p[query_index * ndim + dim]);
                } else {
                    dis += ((base_p[ndim * row + dim] - query_p[query_index * ndim + dim]) *
                            (base_p[ndim * row + dim] - query_p[query_index * ndim + dim]));
                }
            }
            if (pq.size() < topk) {
                pq.push(std::make_pair(row, dis));
            } else if (pq.top().second > dis) {
                pq.pop();
                pq.push(std::make_pair(row, dis));
            }
        }
        int pq_size = pq.size();
        uint32_t index = 0;
        for (; index < pq_size; ++index) {
            auto& id_dis_pair = pq.top();
            ids[query_index * topk + index] = id_dis_pair.first;
            dis[query_index * topk + index] = id_dis_pair.second;
            pq.pop();
        }
        for (; index < topk; ++index) {
            ids[query_index * topk + index] = -1;
            dis[query_index * topk + index] = -1;
        }
    }
    auto gt = std::make_unique<knowhere::DataSet>();
    gt->SetDim(topk);
    gt->SetRows(nq);
    gt->SetIds(ids);
    gt->SetDistance(dis);
    return gt;
}

inline std::unique_ptr<knowhere::DataSet>
GetRangeSearchGroundTruth(const knowhere::DataSet& base, const knowhere::DataSet& query, const std::string& metric,
                          const float radius_low_bound, const float radius_high_bound,
                          const knowhere::BitsetView bitset = nullptr) {
    auto nb = base.GetRows();
    auto nq = query.GetRows();
    auto ndim = base.GetDim();
    auto base_p = reinterpret_cast<const float*>(base.GetTensor());
    auto query_p = reinterpret_cast<const float*>(query.GetTensor());
    bool is_ip = (metric == knowhere::metric::IP);

    std::vector<std::vector<float>> dis_v(nq, std::vector<float>());
    std::vector<std::vector<int64_t>> ids_v(nq, std::vector<int64_t>());
    for (uint32_t query_index = 0; query_index < nq; ++query_index) {
        for (int64_t row = 0; row < nb; ++row) {
            if (!bitset.empty() && bitset.test(row)) {
                continue;
            }
            float distance = 0;
            if (metric == knowhere::metric::IP) {
                for (uint32_t dim = 0; dim < ndim; ++dim) {
                    distance += (base_p[ndim * row + dim] * query_p[query_index * ndim + dim]);
                }
            } else {
                for (uint32_t dim = 0; dim < ndim; ++dim) {
                    distance += ((base_p[ndim * row + dim] - query_p[query_index * ndim + dim]) *
                                 (base_p[ndim * row + dim] - query_p[query_index * ndim + dim]));
                }
            }
            if (knowhere::distance_in_range(distance, radius_low_bound, radius_high_bound, is_ip)) {
                ids_v[query_index].emplace_back(row);
                dis_v[query_index].emplace_back(distance);
            }
        }
    }
    int64_t* ids_p = nullptr;
    float* distances_p = nullptr;
    size_t* lims_p = nullptr;
    knowhere::GetRangeSearchResult(dis_v, ids_v, is_ip, nq, radius_low_bound, radius_high_bound, distances_p, ids_p,
                                   lims_p);

    auto gt = std::make_unique<knowhere::DataSet>();
    gt->SetRows(nq);
    gt->SetIds(ids_p);
    gt->SetDistance(distances_p);
    gt->SetLims(lims_p);
    return gt;
}

inline float
GetKNNRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result) {
    REQUIRE(ground_truth.GetDim() >= result.GetDim());

    auto nq = result.GetRows();
    auto gt_k = ground_truth.GetDim();
    auto res_k = result.GetDim();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();

    uint32_t matched_num = 0;
    for (auto i = 0; i < nq; ++i) {
        std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + res_k);
        std::vector<int64_t> ids_1(res_ids + i * res_k, res_ids + i * res_k + res_k);

        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());

        std::vector<int64_t> v(nq * 2);
        std::vector<int64_t>::iterator it;
        it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it - v.begin());
        matched_num += v.size();
    }
    return ((float)matched_num) / ((float)nq * res_k);
}

inline float
GetRangeSearchRecall(const knowhere::DataSet& gt, const knowhere::DataSet& result) {
    uint32_t nq = result.GetRows();
    auto res_ids_p = result.GetIds();
    auto res_lims_p = result.GetLims();
    auto gt_ids_p = gt.GetIds();
    auto gt_lims_p = gt.GetLims();
    uint32_t ninter = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        std::set<int64_t> inter;
        std::set<int64_t> res_ids_set(res_ids_p + res_lims_p[i], res_ids_p + res_lims_p[i + 1]);
        std::set<int64_t> gt_ids_set(gt_ids_p + gt_lims_p[i], gt_ids_p + gt_lims_p[i + 1]);
        std::set_intersection(res_ids_set.begin(), res_ids_set.end(), gt_ids_set.begin(), gt_ids_set.end(),
                              std::inserter(inter, inter.begin()));
        ninter += inter.size();
    }

    float recall = ninter * 1.0f / gt_lims_p[nq];
    float precision = ninter * 1.0f / res_lims_p[nq];

    return (1 + precision) * recall / 2;
}
