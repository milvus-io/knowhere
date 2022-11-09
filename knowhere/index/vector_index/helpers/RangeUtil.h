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

#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/MetricType.h>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/utils/BitsetView.h"

namespace knowhere {

inline bool
distance_in_range(const float dist, const float low_bound, const float high_bound, const bool is_ip) {
    return ((is_ip && low_bound < dist && dist <= high_bound) || (!is_ip && low_bound <= dist && dist < high_bound));
}

///////////////////////////////////////////////////////////////////////////////
// For Faiss index types
inline void
CountValidRangeSearchResult(const faiss::RangeSearchResult& res,
                            const bool is_ip,
                            const int64_t nq,
                            const float low_bound,
                            const float high_bound,
                            size_t*& lims) {
    lims = new size_t[nq + 1];
    lims[0] = 0;
    for (int64_t i = 0; i < nq; i++) {
        int64_t valid = 0;
        for (size_t j = res.lims[i]; j < res.lims[i + 1]; j++) {
            if (distance_in_range(res.distances[j], low_bound, high_bound, is_ip)) {
                valid++;
            }
        }
        lims[i + 1] = lims[i] + valid;
    }
}

inline void
FilterRangeSearchResultForOneNq(const int64_t i_size,
                                const float* i_distances,
                                 const int64_t* i_labels,
                                 const bool is_ip,
                                 const float low_bound,
                                 const float high_bound,
                                 const int64_t o_size,
                                 float* o_distances,
                                int64_t* o_labels,
                                const faiss::BitsetView bitset) {
    size_t num = 0;
    for (size_t i = 0; i < i_size; i++) {
        auto dis = i_distances[i];
        auto id = i_labels[i];
        KNOWHERE_THROW_IF_NOT_MSG(bitset.empty() || !bitset.test(id), "bitset invalid");
        KNOWHERE_THROW_IF_NOT_FMT((is_ip && dis > low_bound) || (!is_ip && dis < high_bound),
                                  "distance %f invalid, is_ip %s, low_bound %f, high_bound %f",
                                  dis, (is_ip ? "true" : "false"), low_bound, high_bound);
        if (distance_in_range(dis, low_bound, high_bound, is_ip)) {
            o_labels[num] = id;
            o_distances[num] = dis;
            num++;
        }
    }
    KNOWHERE_THROW_IF_NOT_FMT(num == o_size, "%ld not equal %ld", num, o_size);
}

inline void
GetRangeSearchResult(const faiss::RangeSearchResult& res,
                     const bool is_ip,
                     const int64_t nq,
                     const float low_bound,
                     const float high_bound,
                     float*& distances,
                     int64_t*& labels,
                     size_t*& lims,
                     const faiss::BitsetView bitset) {
    CountValidRangeSearchResult(res, is_ip, nq, low_bound, high_bound, lims);

    size_t total_valid = lims[nq];
    LOG_KNOWHERE_DEBUG_ << "Range search metric type: " << (is_ip ? "IP" : "L2") << ", low_bound " << low_bound
                        << ", high_bound " << high_bound << ", total result num: " << total_valid;

    distances = new float[total_valid];
    labels = new int64_t[total_valid];

#pragma omp parallel for
    for (auto i = 0; i < nq; i++) {
        FilterRangeSearchResultForOneNq(res.lims[i + 1] - res.lims[i],
                                        res.distances + res.lims[i],
                                        res.labels + res.lims[i],
                                        is_ip,
                                        low_bound,
                                        high_bound,
                                        lims[i + 1] - lims[i],
                                        distances + lims[i],
                                        labels + lims[i],
                                        bitset);
    }
}

///////////////////////////////////////////////////////////////////////////////
// for HNSW and DiskANN
inline void
FilterRangeSearchResultForOneNq(std::vector<float>& distances,
                                std::vector<int64_t>& labels,
                                const bool is_ip,
                                const float low_bound,
                                const float high_bound) {
    KNOWHERE_THROW_IF_NOT_FMT(distances.size() == labels.size(),
                              "distances' size %ld not equal to labels' size %ld", distances.size(), labels.size());
    auto len = distances.size();
    size_t valid_cnt = 0;
    for (size_t i = 0; i < len; i++) {
        auto dist = distances[i];
        auto id = labels[i];
        if (distance_in_range(dist, low_bound, high_bound, is_ip)) {
            distances[valid_cnt] = dist;
            labels[valid_cnt] = id;
            valid_cnt++;
        }
    }
    if (valid_cnt != distances.size()) {
        distances.resize(valid_cnt);
        labels.resize(valid_cnt);
    }
}

inline void
GetRangeSearchResult(const std::vector<std::vector<float>>& result_distances,
                     const std::vector<std::vector<int64_t>>& result_labels,
                     const bool is_ip,
                     const int64_t nq,
                     const float low_bound,
                     const float high_bound,
                     float*& distances,
                     int64_t*& labels,
                     size_t*& lims) {
    KNOWHERE_THROW_IF_NOT_FMT(result_distances.size() == nq,
                              "result distances size %ld not equal to %ld", result_distances.size(), nq);
    KNOWHERE_THROW_IF_NOT_FMT(result_labels.size() == nq,
                              "result labels size %ld not equal to %ld", result_labels.size(), nq);

    lims = new size_t[nq + 1];
    lims[0] = 0;
    // all distances must be in range scope
    for (int64_t i = 0; i < nq; i++) {
        lims[i + 1] = lims[i] + result_distances[i].size();
    }

    size_t total_valid = lims[nq];
    LOG_KNOWHERE_DEBUG_ << "Range search metric type: " << (is_ip ? "IP" : "L2") << ", low_bound " << low_bound
                        << ", high_bound " << high_bound << ", total result num: " << total_valid;

    distances = new float[total_valid];
    labels = new int64_t[total_valid];

    for (auto i = 0; i < nq; i++) {
        std::copy_n(result_distances[i].data(), lims[i + 1] - lims[i], distances + lims[i]);
        std::copy_n(result_labels[i].data(), lims[i + 1] - lims[i], labels + lims[i]);
    }
}

}  // namespace knowhere
