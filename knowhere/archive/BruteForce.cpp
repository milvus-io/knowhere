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

namespace knowhere {

// copy from faiss/IndexBinaryFlat.cpp::IndexBinaryFlat::search()
// disable lint to make further migration easier
void
BruteForceSearch(const knowhere::MetricType& metric_type,
                 const void* xb,
                 const void* xq,
                 const int64_t dim,
                 const int64_t nb,
                 const int64_t nq,
                 const int64_t topk,
                 int64_t* labels,
                 float* distances,
                 const faiss::BitsetView bitset) {
    if (metric_type == knowhere::metric::L2) {
        faiss::float_maxheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
        faiss::knn_L2sqr((const float*)xq, (const float*)xb, dim, nq, nb, &buf, nullptr, bitset);
    } else if (metric_type == knowhere::metric::IP) {
        faiss::float_minheap_array_t buf{(size_t)nq, (size_t)topk, labels, distances};
        faiss::knn_inner_product((const float*)xq, (const float*)xb, dim, nq, nb, &buf, bitset);
    } else if (metric_type == knowhere::metric::JACCARD || metric_type == knowhere::metric::TANIMOTO) {
        faiss::float_maxheap_array_t res = {size_t(nq), size_t(topk), labels, distances};
        binary_distance_knn_hc(
            faiss::METRIC_Jaccard, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8, bitset);

        if (metric_type == knowhere::metric::TANIMOTO) {
            for (int i = 0; i < topk * nq; i++) {
                distances[i] = faiss::Jaccard_2_Tanimoto(distances[i]);
            }
        }
    } else if (metric_type == knowhere::metric::HAMMING) {
        std::vector<int32_t> int_distances(nq * topk);
        faiss::int_maxheap_array_t res = {size_t(nq), size_t(topk), labels, int_distances.data()};
        binary_distance_knn_hc(
            faiss::METRIC_Hamming, &res, (const uint8_t*)xq, (const uint8_t*)xb, nb, dim / 8, bitset);
        for (int i = 0; i < nq * topk; ++i) {
            distances[i] = int_distances[i];
        }
    } else if (metric_type == knowhere::metric::SUBSTRUCTURE || metric_type == knowhere::metric::SUPERSTRUCTURE) {
        // only matched ids will be chosen, not to use heap
        auto faiss_metric_type = knowhere::GetFaissMetricType(metric_type);
        binary_distance_knn_mc(faiss_metric_type, (const uint8_t*)xq, (const uint8_t*)xb, nq, nb, topk, dim / 8,
                               distances, labels, bitset);
    } else {
        KNOWHERE_THROW_MSG("BruteForce search not support metric type: " + metric_type);
    }
}

}  // namespace knowhere
