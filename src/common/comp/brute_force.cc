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
    Config::Load(cfg, config, knowhere::SEARCH);

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return unexpected(Status::invalid_metric_type);
    }

    int topk = cfg.k;
    auto labels = new int64_t[nq * topk];
    auto distances = new float[nq * topk];

    auto faiss_metric_type = metric_type.value();
    switch (faiss_metric_type) {
        case faiss::METRIC_L2: {
            faiss::float_maxheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
            faiss::knn_L2sqr((const float*)xq, (const float*)xb, dim, nq, nb, &buf, nullptr, bitset);
            break;
        }
        case faiss::METRIC_INNER_PRODUCT: {
            faiss::float_minheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
            faiss::knn_inner_product((const float*)xq, (const float*)xb, dim, nq, nb, &buf, bitset);
            break;
        }
        case faiss::METRIC_Jaccard:
        case faiss::METRIC_Tanimoto: {
            faiss::float_maxheap_array_t res = {size_t(nq), size_t(topk), labels, distances};
            binary_distance_knn_hc(faiss::METRIC_Jaccard, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8,
                                   bitset);

            if (faiss_metric_type == faiss::METRIC_Tanimoto) {
                for (int i = 0; i < topk * nq; i++) {
                    distances[i] = faiss::Jaccard_2_Tanimoto(distances[i]);
                }
            }
            break;
        }
        case faiss::METRIC_Hamming: {
            std::vector<int32_t> int_distances(nq * topk);
            faiss::int_maxheap_array_t res = {size_t(nq), size_t(topk), labels, int_distances.data()};
            binary_distance_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8,
                                   bitset);
            for (int i = 0; i < nq * topk; ++i) {
                distances[i] = int_distances[i];
            }
            break;
        }
        case faiss::METRIC_Substructure:
        case faiss::METRIC_Superstructure: {
            // only matched ids will be chosen, not to use heap
            binary_distance_knn_mc(faiss_metric_type, (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, topk, dim / 8,
                                   distances, labels, bitset);
            break;
        }
        default:
            return unexpected(Status::invalid_metric_type);
    }

    DataSetPtr results = std::make_shared<DataSet>();
    results->SetIds(labels);
    results->SetDistance(distances);
    return results;
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
    Config::Load(cfg, config, knowhere::SEARCH);

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return Status::invalid_metric_type;
    }

    int topk = cfg.k;
    auto labels = ids;
    auto distances = dis;

    auto faiss_metric_type = metric_type.value();
    switch (faiss_metric_type) {
        case faiss::METRIC_L2: {
            faiss::float_maxheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
            faiss::knn_L2sqr((const float*)xq, (const float*)xb, dim, nq, nb, &buf, nullptr, bitset);
            break;
        }
        case faiss::METRIC_INNER_PRODUCT: {
            faiss::float_minheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
            faiss::knn_inner_product((const float*)xq, (const float*)xb, dim, nq, nb, &buf, bitset);
            break;
        }
        case faiss::METRIC_Jaccard:
        case faiss::METRIC_Tanimoto: {
            faiss::float_maxheap_array_t res = {size_t(nq), size_t(topk), labels, distances};
            binary_distance_knn_hc(faiss::METRIC_Jaccard, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8,
                                   bitset);

            if (faiss_metric_type == faiss::METRIC_Tanimoto) {
                for (int i = 0; i < topk * nq; i++) {
                    distances[i] = faiss::Jaccard_2_Tanimoto(distances[i]);
                }
            }
            break;
        }
        case faiss::METRIC_Hamming: {
            std::vector<int32_t> int_distances(nq * topk);
            faiss::int_maxheap_array_t res = {size_t(nq), size_t(topk), labels, int_distances.data()};
            binary_distance_knn_hc(faiss::METRIC_Hamming, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8,
                                   bitset);
            for (int i = 0; i < nq * topk; ++i) {
                distances[i] = int_distances[i];
            }
            break;
        }
        case faiss::METRIC_Substructure:
        case faiss::METRIC_Superstructure: {
            // only matched ids will be chosen, not to use heap
            binary_distance_knn_mc(faiss_metric_type, (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, topk, dim / 8,
                                   distances, labels, bitset);
            break;
        }
        default:
            return Status::invalid_metric_type;
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
    Config::Load(cfg, config, knowhere::RANGE_SEARCH);

    auto metric_type = Str2FaissMetricType(cfg.metric_type);
    if (!metric_type.has_value()) {
        return unexpected(Status::invalid_metric_type);
    }

    auto low_bound = cfg.radius_low_bound;
    auto high_bound = cfg.radius_high_bound;

    faiss::RangeSearchResult res(nq);
    auto faiss_metric_type = metric_type.value();
    switch (faiss_metric_type) {
        case faiss::METRIC_L2:
            faiss::range_search_L2sqr((const float*)xq, (const float*)xb, dim, nq, nb, high_bound, &res, bitset);
            break;
        case faiss::METRIC_INNER_PRODUCT:
            faiss::range_search_inner_product((const float*)xq, (const float*)xb, dim, nq, nb, low_bound, &res, bitset);
            break;
        case faiss::METRIC_Jaccard:
            faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(faiss::METRIC_Jaccard, (const uint8_t*)xq,
                                                                           (const uint8_t*)xb, nq, nb, high_bound,
                                                                           dim / 8, &res, bitset);
            break;
        case faiss::METRIC_Tanimoto:
            faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(faiss::METRIC_Tanimoto, (const uint8_t*)xq,
                                                                           (const uint8_t*)xb, nq, nb, high_bound,
                                                                           dim / 8, &res, bitset);
            break;
        case faiss::METRIC_Hamming:
            faiss::binary_range_search<faiss::CMin<int, int64_t>, int>(faiss::METRIC_Hamming, (const uint8_t*)xq,
                                                                       (const uint8_t*)xb, nq, nb, (int)high_bound,
                                                                       dim / 8, &res, bitset);
            break;
        default:
            return unexpected(Status::invalid_metric_type);
    }

    int64_t* labels = nullptr;
    float* distances = nullptr;
    size_t* lims = nullptr;

    GetRangeSearchResult(res, (faiss_metric_type == faiss::METRIC_INNER_PRODUCT), nq, low_bound, high_bound, distances,
                         labels, lims, bitset);

    DataSetPtr results = std::make_shared<DataSet>();
    results->SetRows(nq);
    results->SetIds(labels);
    results->SetDistance(distances);
    results->SetLims(lims);
    return results;
}
}  // namespace knowhere
