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

#include <memory>
#include <string>
#include <vector>
#include "faiss/impl/AuxIndexStructures.h"
#include "knowhere/common/Typedef.h"

namespace knowhere {

using idx_t = int64_t;

// BufferPool (inner classes)
using BufferList = faiss::BufferList;
using BufferListPtr = std::shared_ptr<BufferList>;

/*
 * Class: Dynamic result set (merged results)
 */
struct RangeSearchResult {
    enum class SortType { None = 0, AscOrder, DescOrder };

    std::shared_ptr<idx_t[]> labels;     /// result for query i is labels[lims[i]:lims[i + 1]]
    std::shared_ptr<float[]> distances;  /// corresponding distances, not sorted
    size_t count;  /// size of the result buffer's size, when reaches this size, auto start a new buffer

    void
    AllocImpl();

    void
    SortImpl(SortType type = SortType::AscOrder);

 private:
    template<bool asc>
    void
    quick_sort(int64_t start, int64_t end);
};

/*
 * Class: Dynamic result collector
 * Example:
    DynamicResultCollector collector;
    for (auto &seg: segments) {
        auto seg_rst = seg.QueryByDistance(xxx);
        collector.append(seg_rst);
    }
    auto rst = collector.merge();
 */
struct RangeSearchResultHandler {
 public:
    /*
     * Merge the results of segments
     * Notes: Now, we apply limit before sort.
     *        It can be updated if you don't expect the policy.
     */
    RangeSearchResult
    Merge(size_t limit = 10000, RangeSearchResult::SortType type = RangeSearchResult::SortType::None);

    /*
     * Collect the results of segments
     */
    void
    Append(std::vector<BufferListPtr>& seg_result);

 private:
    std::vector<std::vector<BufferListPtr>> seg_results;  /// unmerged results of every segments
};

void
ExchangeDataset(std::vector<BufferListPtr>& dst, std::vector<faiss::RangeSearchPartialResult*>& src);

}  // namespace knowhere
