// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "faiss/utils/BinaryDistance.h"
#include "faiss/utils/distances.h"
#include "knowhere/archive/BruteForce.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/RangeUtil.h"

namespace knowhere {

/** knowhere wrapper API to call faiss brute force search for all metric types
 */
DatasetPtr
BruteForce::Search(const DatasetPtr base_dataset,
                   const DatasetPtr query_dataset,
                   const Config& config,
                   const faiss::BitsetView bitset) {
    auto xb = GetDatasetTensor(base_dataset);
    auto nb = GetDatasetRows(base_dataset);
    auto dim = GetDatasetDim(base_dataset);

    auto xq = GetDatasetTensor(query_dataset);
    auto nq = GetDatasetRows(query_dataset);

    auto metric_type = GetMetaMetricType(config);
    auto topk = GetMetaTopk(config);

    auto labels = new int64_t[nq * topk];
    auto distances = new float[nq * topk];

    auto faiss_metric_type = GetFaissMetricType(metric_type);
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
            KNOWHERE_THROW_MSG("BruteForce search not support metric type: " + metric_type);
    }
    return GenResultDataset(labels, distances);
}

/** knowhere wrapper API to call faiss brute force range search for all metric types
 */
DatasetPtr
BruteForce::RangeSearch(const DatasetPtr base_dataset,
                        const DatasetPtr query_dataset,
                        const Config& config,
                        const faiss::BitsetView bitset) {
    auto xb = GetDatasetTensor(base_dataset);
    auto nb = GetDatasetRows(base_dataset);
    auto dim = GetDatasetDim(base_dataset);

    auto xq = GetDatasetTensor(query_dataset);
    auto nq = GetDatasetRows(query_dataset);

    auto metric_type = GetMetaMetricType(config);
    auto low_bound = GetMetaRadiusLowBound(config);
    auto high_bound = GetMetaRadiusHighBound(config);

    faiss::RangeSearchResult res(nq);
    auto faiss_metric_type = GetFaissMetricType(metric_type);
    switch (faiss_metric_type) {
        case faiss::METRIC_L2:
            faiss::range_search_L2sqr((const float*)xq, (const float*)xb, dim, nq, nb, high_bound, &res, bitset);
            break;
        case faiss::METRIC_INNER_PRODUCT:
            faiss::range_search_inner_product((const float*)xq, (const float*)xb, dim, nq, nb, low_bound, &res, bitset);
            break;
        case faiss::METRIC_Jaccard:
            faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(faiss::METRIC_Jaccard,
                    (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, high_bound, dim / 8, &res, bitset);
            break;
        case faiss::METRIC_Tanimoto:
            faiss::binary_range_search<faiss::CMin<float, int64_t>, float>(faiss::METRIC_Tanimoto,
                    (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, high_bound, dim / 8, &res, bitset);
            break;
        case faiss::METRIC_Hamming:
            faiss::binary_range_search<faiss::CMin<int, int64_t>, int>(faiss::METRIC_Hamming,
                    (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, (int)high_bound, dim / 8, &res, bitset);
            break;
        default:
            KNOWHERE_THROW_MSG("BruteForce range search not support metric type: " + metric_type);
    }

    float* distances = nullptr;
    int64_t* labels = nullptr;
    size_t* lims = nullptr;
    GetRangeSearchResult(res, (faiss_metric_type == faiss::METRIC_INNER_PRODUCT), nq, low_bound, high_bound,
                         distances, labels, lims, bitset);

    return GenResultDataset(labels, distances, lims);
}

}  // namespace knowhere
