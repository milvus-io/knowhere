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

#include <faiss/impl/AuxIndexStructures.h>

#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

inline bool
distance_in_range(const float dist, const float low_bound, const float high_bound, const bool is_ip) {
    return ((is_ip && low_bound < dist && dist <= high_bound) || (!is_ip && low_bound <= dist && dist < high_bound));
}

void
GetRangeSearchResult(const faiss::RangeSearchResult& res, const bool is_ip, const int64_t nq, const float low_bound,
                     const float high_bound, float*& distances, int64_t*& labels, size_t*& lims,
                     const BitsetView& bitset);

void
FilterRangeSearchResultForOneNq(std::vector<float>& distances, std::vector<int64_t>& labels, const bool is_ip,
                                const float low_bound, const float high_bound);

void
GetRangeSearchResult(const std::vector<std::vector<float>>& result_distances,
                     const std::vector<std::vector<int64_t>>& result_labels, const bool is_ip, const int64_t nq,
                     const float low_bound, const float high_bound, float*& distances, int64_t*& labels, size_t*& lims);

}  // namespace knowhere
