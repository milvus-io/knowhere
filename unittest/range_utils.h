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

#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>

#include <faiss/utils/BinaryDistance.h>
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/RangeUtil.h"
#include "knowhere/utils/BitsetView.h"
#include "knowhere/utils/distances_simd.h"

inline float float_vec_dist(
    const knowhere::MetricType metric_type,
    const float* pa,
    const float* pb,
    const size_t dim) {
    assert(metric_type == knowhere::metric::L2 || metric_type == knowhere::metric::IP);
    if (metric_type == knowhere::metric::L2) {
        return faiss::fvec_L2sqr_ref(pa, pb, dim);
    } else {
        return faiss::fvec_inner_product_ref(pa, pb, dim);
    }
}

inline float binary_vec_dist(
    const knowhere::MetricType metric_type,
    const uint8_t* pa,
    const uint8_t* pb,
    const size_t code_size) {
    assert(metric_type == knowhere::metric::HAMMING ||
           metric_type == knowhere::metric::JACCARD ||
           metric_type == knowhere::metric::TANIMOTO);
    if (metric_type == knowhere::metric::HAMMING) {
        return faiss::xor_popcnt(pa, pb, code_size);
    } else if (metric_type == knowhere::metric::JACCARD) {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        return 1.0 - (float)and_value / or_value;
    } else {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        auto v = 1.0 - (float)and_value / or_value;
        return (-log2(1 - v));
    }
}

inline void RunFloatRangeSearchBF(
    std::vector<int64_t>& golden_ids,
    std::vector<float>& golden_distances,
    std::vector<size_t>& golden_lims,
    const knowhere::MetricType metric_type,
    const float* xb,
    const int64_t nb,
    const float* xq,
    const int64_t nq,
    const int64_t dim,
    const float low_bound,
    const float high_bound,
    const faiss::BitsetView bitset) {

    bool is_ip = (metric_type == knowhere::metric::IP);
    std::vector<std::vector<int64_t>> ids_v(nq);
    std::vector<std::vector<float>> distances_v(nq);

#pragma omp parallel for
    for (auto i = 0; i < nq; ++i) {
        const float* pq = xq + i * dim;
        for (auto j = 0; j < nb; ++j) {
            if (bitset.empty() || !bitset.test(j)) {
                const float* pb = xb + j * dim;
                auto dist = float_vec_dist(metric_type, pq, pb, dim);
                if (knowhere::distance_in_range(dist, low_bound, high_bound, is_ip)) {
                    ids_v[i].push_back(j);
                    distances_v[i].push_back(dist);
                }
            }
        }
    }

    golden_lims.push_back(0);
    for (auto i = 0; i < nq; ++i) {
        golden_ids.insert(golden_ids.end(), ids_v[i].begin(), ids_v[i].end());
        golden_distances.insert(golden_distances.end(), distances_v[i].begin(), distances_v[i].end());
        golden_lims.push_back(golden_ids.size());
    }
}

inline void RunBinaryRangeSearchBF(
    std::vector<int64_t>& golden_ids,
    std::vector<float>& golden_distances,
    std::vector<size_t>& golden_lims,
    const knowhere::MetricType metric_type,
    const uint8_t* xb,
    const int64_t nb,
    const uint8_t* xq,
    const int64_t nq,
    const int64_t dim,
    const float low_bound,
    const float high_bound,
    const faiss::BitsetView bitset) {

    std::vector<std::vector<int64_t>> ids_v(nq);
    std::vector<std::vector<float>> distances_v(nq);

#pragma omp parallel for
    for (auto i = 0; i < nq; ++i) {
        const uint8_t* pq = xq + i * dim / 8;
        for (auto j = 0; j < nb; ++j) {
            if (bitset.empty() || !bitset.test(j)) {
                const uint8_t* pb = xb + j * dim / 8;
                auto dist = binary_vec_dist(metric_type, pq, pb, dim/8);
                if (knowhere::distance_in_range(dist, low_bound, high_bound, false)) {
                    ids_v[i].push_back(j);
                    distances_v[i].push_back(dist);
                }
            }
        }
    }

    golden_lims.push_back(0);
    for (auto i = 0; i < nq; ++i) {
        golden_ids.insert(golden_ids.end(), ids_v[i].begin(), ids_v[i].end());
        golden_distances.insert(golden_distances.end(), distances_v[i].begin(), distances_v[i].end());
        golden_lims.push_back(golden_ids.size());
    }
}

inline void CheckRangeSearchResult(
    const knowhere::DatasetPtr& result,
    const knowhere::MetricType& metric_type,
    const int64_t nq,
    const float low_bound,
    const float high_bound,
    const int64_t* golden_ids,
    const size_t* golden_lims,
    const bool is_idmap,
    const faiss::BitsetView bitset) {

    bool is_ip = (metric_type == knowhere::metric::IP);
    auto lims = knowhere::GetDatasetLims(result);
    auto ids = knowhere::GetDatasetIDs(result);
    auto distances = knowhere::GetDatasetDistance(result);

    printf("Range search num (%4ld / %4ld)\n", lims[nq], golden_lims[nq]);
    for (int i = 0; i < nq; i++) {
        size_t golden_size = golden_lims[i+1] - golden_lims[i];
        size_t size = lims[i+1] - lims[i];
        if (is_idmap) {
            ASSERT_EQ(size, golden_size);
        }

        int64_t recall_cnt = 0;
        std::unordered_set<int64_t> golden_ids_set(golden_ids + golden_lims[i],
                                                   golden_ids + golden_lims[i+1]);
        for (int j = lims[i]; j < lims[i+1]; j++) {
            ASSERT_TRUE(bitset.empty() || !bitset.test(ids[j]));
            bool hit = (golden_ids_set.count(ids[j]) == 1);
            if (hit) {
                recall_cnt++;
            } else if (is_idmap) {
                // only IDMAP always hit
                ASSERT_TRUE(hit);
            }
            ASSERT_TRUE(knowhere::distance_in_range(distances[j], low_bound, high_bound, is_ip));
        }

        int64_t accuracy_cnt = 0;
        std::vector<int64_t> ids_array(ids + lims[i], ids + lims[i+1]);
        std::unordered_set<int64_t> ids_set(ids_array.begin(), ids_array.end());
        for (size_t j = golden_lims[i]; j < golden_lims[i+1]; j++) {
            bool hit = (ids_set.count(golden_ids[j]) == 1);
            if (hit) {
                accuracy_cnt++;
            } else if (is_idmap) {
                // only IDMAP always hit
                ASSERT_TRUE(hit);
            }
        }

        printf("\tNo.%2d: recall = %.2f (%4ld / %4ld),  accuracy = %.2f (%4ld / %4ld)\n", i,
               (recall_cnt * 1.0f / golden_size), recall_cnt, golden_size,
               (accuracy_cnt * 1.0f / size), accuracy_cnt, size);
    }
}
