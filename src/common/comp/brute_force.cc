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

#include "knowhere/comp/brute_force.h"

#include <vector>

#include "common/metric.h"
#include "common/range_util.h"
#include "faiss/utils/BinaryDistance.h"
#include "faiss/utils/distances.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"

namespace knowhere {

/** knowhere wrapper API to call faiss brute force search for all metric types
 */

class BruteForceConfig : public BaseConfig {};

expected<DataSetPtr, Status>
BruteForce::Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                   const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    auto load_res = Config::Load(cfg, config, knowhere::SEARCH);
    if (load_res != Status::success) {
        return unexpected(load_res);
    }

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return unexpected(Status::invalid_metric_type);
    }

    int topk = cfg.k;
    auto labels = new int64_t[nq * topk];
    auto distances = new float[nq * topk];

    auto faiss_metric_type = metric_type.value();
    auto pool = ThreadPool::GetGlobalThreadPool();
    std::vector<std::future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.push_back(pool->push([&, index = i] {
            ThreadPool::ScopedOmpSetter setter(1);
            auto cur_labels = labels + topk * index;
            auto cur_distances = distances + topk * index;
            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_maxheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_L2sqr(cur_query, (const float*)xb, dim, 1, nb, &buf, nullptr, bitset);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_minheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_inner_product(cur_query, (const float*)xb, dim, 1, nb, &buf, bitset);
                    break;
                }
                case faiss::METRIC_Jaccard:
                case faiss::METRIC_Tanimoto: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::float_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, cur_distances};
                    binary_distance_knn_hc(faiss::METRIC_Jaccard, &res, cur_query, (const uint8_t*)xb, nb, dim / 8,
                                           bitset);

                    if (faiss_metric_type == faiss::METRIC_Tanimoto) {
                        for (int i = 0; i < topk; i++) {
                            cur_distances[i] = faiss::Jaccard_2_Tanimoto(cur_distances[i]);
                        }
                    }
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    std::vector<int32_t> int_distances(topk);
                    faiss::int_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, int_distances.data()};
                    binary_distance_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)cur_query, (const uint8_t*)xb,
                                           nb, dim / 8, bitset);
                    for (int i = 0; i < topk; ++i) {
                        cur_distances[i] = int_distances[i];
                    }
                    break;
                }
                case faiss::METRIC_Substructure:
                case faiss::METRIC_Superstructure: {
                    // only matched ids will be chosen, not to use heap
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    binary_distance_knn_mc(faiss_metric_type, cur_query, (const uint8_t*)xb, 1, nb, topk, dim / 8,
                                           cur_distances, cur_labels, bitset);
                    break;
                }
                default:
                    return Status::invalid_metric_type;
            }
            return Status::success;
        }));
    }
    for (auto& fut : futs) {
        auto ret = fut.get();
        if (ret != Status::success) {
            return unexpected(ret);
        }
    }
    return GenResultDataSet(nq, cfg.k, labels, distances);
}

Status
BruteForce::SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                          const Json& config, const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    auto load_res = Config::Load(cfg, config, knowhere::SEARCH);
    if (load_res != Status::success) {
        return load_res;
    }

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return Status::invalid_metric_type;
    }

    int topk = cfg.k;
    auto labels = ids;
    auto distances = dis;

    auto faiss_metric_type = metric_type.value();
    auto pool = ThreadPool::GetGlobalThreadPool();
    std::vector<std::future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.push_back(pool->push([&, index = i] {
            ThreadPool::ScopedOmpSetter setter(1);
            auto cur_labels = labels + topk * index;
            auto cur_distances = distances + topk * index;
            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_maxheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_L2sqr(cur_query, (const float*)xb, dim, 1, nb, &buf, nullptr, bitset);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::float_minheap_array_t buf{(size_t)1, (size_t)topk, cur_labels, cur_distances};
                    faiss::knn_inner_product(cur_query, (const float*)xb, dim, 1, nb, &buf, bitset);
                    break;
                }
                case faiss::METRIC_Jaccard:
                case faiss::METRIC_Tanimoto: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::float_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, cur_distances};
                    binary_distance_knn_hc(faiss::METRIC_Jaccard, &res, cur_query, (const uint8_t*)xb, nb, dim / 8,
                                           bitset);

                    if (faiss_metric_type == faiss::METRIC_Tanimoto) {
                        for (int i = 0; i < topk; i++) {
                            cur_distances[i] = faiss::Jaccard_2_Tanimoto(distances[i]);
                        }
                    }
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    std::vector<int32_t> int_distances(topk);
                    faiss::int_maxheap_array_t res = {size_t(1), size_t(topk), cur_labels, int_distances.data()};
                    binary_distance_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)cur_query, (const uint8_t*)xb,
                                           nb, dim / 8, bitset);
                    for (int i = 0; i < topk; ++i) {
                        cur_distances[i] = int_distances[i];
                    }
                    break;
                }
                case faiss::METRIC_Substructure:
                case faiss::METRIC_Superstructure: {
                    // only matched ids will be chosen, not to use heap
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    binary_distance_knn_mc(faiss_metric_type, cur_query, (const uint8_t*)xb, 1, nb, topk, dim / 8,
                                           cur_distances, cur_labels, bitset);
                    break;
                }
                default:
                    return Status::invalid_metric_type;
            }
            return Status::success;
        }));
    }
    for (auto& fut : futs) {
        auto ret = fut.get();
        if (ret != Status::success) {
            return ret;
        }
    }
    return Status::success;
}

/** knowhere wrapper API to call faiss brute force range search for all metric types
 */
expected<DataSetPtr, Status>
BruteForce::RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                        const BitsetView& bitset) {
    auto xb = base_dataset->GetTensor();
    auto nb = base_dataset->GetRows();
    auto dim = base_dataset->GetDim();

    auto xq = query_dataset->GetTensor();
    auto nq = query_dataset->GetRows();

    BruteForceConfig cfg;
    auto load_res = Config::Load(cfg, config, knowhere::RANGE_SEARCH);
    if (load_res != Status::success) {
        return unexpected(load_res);
    }

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return unexpected(Status::invalid_metric_type);
    }

    auto radius = cfg.radius;
    bool is_ip = false;
    float range_filter = cfg.range_filter;

    auto faiss_metric_type = metric_type.value();
    auto pool = ThreadPool::GetGlobalThreadPool();

    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);
    std::vector<size_t> result_size(nq);
    std::vector<size_t> result_lims(nq + 1);
    std::vector<std::future<Status>> futs;
    futs.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        futs.push_back(pool->push([&, index = i] {
            ThreadPool::ScopedOmpSetter setter(1);
            faiss::RangeSearchResult res(1);
            switch (faiss_metric_type) {
                case faiss::METRIC_L2: {
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::range_search_L2sqr(cur_query, (const float*)xb, dim, 1, nb, radius, &res, bitset);
                    break;
                }
                case faiss::METRIC_INNER_PRODUCT: {
                    is_ip = true;
                    auto cur_query = (const float*)xq + dim * index;
                    faiss::range_search_inner_product(cur_query, (const float*)xb, dim, 1, nb, radius, &res, bitset);
                    break;
                }
                case faiss::METRIC_Jaccard: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(
                        faiss::METRIC_Jaccard, cur_query, (const uint8_t*)xb, 1, nb, radius, dim / 8, &res, bitset);
                    break;
                }
                case faiss::METRIC_Tanimoto: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(
                        faiss::METRIC_Tanimoto, cur_query, (const uint8_t*)xb, 1, nb, radius, dim / 8, &res, bitset);
                    break;
                }
                case faiss::METRIC_Hamming: {
                    auto cur_query = (const uint8_t*)xq + (dim / 8) * index;
                    faiss::binary_range_search<faiss::CMin<int, int64_t>, int>(faiss::METRIC_Hamming, cur_query,
                                                                               (const uint8_t*)xb, 1, nb, (int)radius,
                                                                               dim / 8, &res, bitset);
                    break;
                }
                default:
                    return Status::invalid_metric_type;
            }
            auto elem_cnt = res.lims[1];
            result_dist_array[index].resize(elem_cnt);
            result_id_array[index].resize(elem_cnt);
            result_size[index] = elem_cnt;
            for (size_t j = 0; j < elem_cnt; j++) {
                result_dist_array[index][j] = res.distances[j];
                result_id_array[index][j] = res.labels[j];
            }
            if (cfg.range_filter != defaultRangeFilter) {
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                range_filter);
            }
            return Status::success;
        }));
    }
    for (auto& fut : futs) {
        auto ret = fut.get();
        if (ret != Status::success) {
            return unexpected(ret);
        }
    }

    int64_t* ids = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;
    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius, range_filter, distances, ids, lims);
    return GenResultDataSet(nq, ids, distances, lims);
}
}  // namespace knowhere
