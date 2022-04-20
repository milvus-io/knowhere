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

#include <cstring>
#include <string>
#include <utility>

#include "common/Exception.h"
#include "index/vector_index/helpers/DynamicResultSet.h"

namespace knowhere {

/***********************************************************************
 * DynamicResultSet
 ***********************************************************************/

void
RangeSearchResult::AllocImpl() {
    if (count <= 0) {
        KNOWHERE_THROW_MSG("RangeSearchResult::do_alloction failed because of count <= 0");
    }
    labels = std::shared_ptr<idx_t[]>(new idx_t[count]);
    distances = std::shared_ptr<float[]>(new float[count]);
}

void
RangeSearchResult::SortImpl(SortType type) {
    switch (type) {
        case SortType::AscOrder:
            quick_sort<true>(0, count - 1);
            break;
        case SortType::DescOrder:
            quick_sort<false>(0, count - 1);
            break;
        default:
            KNOWHERE_THROW_MSG("invalid sort type!");
    }
}

template <bool is_asc>
void
RangeSearchResult::quick_sort(int64_t start, int64_t end) {
    if (start >= end) return;
    int64_t l = start, r = end;
    int64_t x_id = labels[start];
    float x_dist = distances[start];
    while (l < r) {
        // search from right to left, find the first num less than x
        while (l < r && ((is_asc && distances[r] >= x_dist) || (!is_asc && distances[r] <= x_dist))) {
            r--;
        }
        if (l < r) {
            labels[l] = labels[r];
            distances[l] = distances[r];
            l++;
        }

        // search from left to right, find the first num bigger than x
        while (l < r && ((is_asc && distances[l] <= x_dist) || (!is_asc && distances[l] >= x_dist))) {
            l++;
        }
        if (l < r) {
            labels[r] = labels[l];
            distances[r] = distances[l];
            r--;
        }
    }
    labels[r] = x_id;
    distances[r] = x_dist;
    quick_sort<is_asc>(start, r-1);
    quick_sort<is_asc>(r+1, end);
}

RangeSearchResult
RangeSearchResultHandler::Merge(size_t limit, RangeSearchResult::SortType type) {
    if (limit <= 0) {
        KNOWHERE_THROW_MSG("limit must > 0!");
    }
    RangeSearchResult ret;
    auto seg_num = seg_results.size();
    std::vector<size_t> boundaries(seg_num + 1, 0);
#pragma omp parallel for
    for (auto i = 0; i < seg_num; ++i) {
        for (auto& pseg : seg_results[i]) {
            boundaries[i] += (pseg->buffer_size * pseg->buffers.size() - pseg->buffer_size + pseg->wp);
        }
    }
    for (size_t i = 0, ofs = 0; i <= seg_num; ++i) {
        auto bn = boundaries[i];
        boundaries[i] = ofs;
        ofs += bn;
        //        boundaries[i] += boundaries[i - 1];
    }
    ret.count = boundaries[seg_num] <= limit ? boundaries[seg_num] : limit;
    ret.AllocImpl();

    // abandon redundancy answers randomly
    // abandon strategy: keep the top limit sequentially
    int32_t pos = 1;
    for (int i = 1; i < boundaries.size(); ++i) {
        if (boundaries[i] >= ret.count) {
            pos = i;
            break;
        }
    }
    pos--;  // last segment id
    // full copy
#pragma omp parallel for
    for (auto i = 0; i < pos; ++i) {
        for (auto& pseg : seg_results[i]) {
            auto len = pseg->buffers.size() * pseg->buffer_size - pseg->buffer_size + pseg->wp;
            pseg->copy_range(0, len, ret.labels.get() + boundaries[i], ret.distances.get() + boundaries[i]);
            boundaries[i] += len;
        }
    }
    // partial copy
    auto last_len = ret.count - boundaries[pos];
    for (auto& pseg : seg_results[pos]) {
        auto len = pseg->buffers.size() * pseg->buffer_size - pseg->buffer_size + pseg->wp;
        auto ncopy = last_len > len ? len : last_len;
        pseg->copy_range(0, ncopy, ret.labels.get() + boundaries[pos], ret.distances.get() + boundaries[pos]);
        boundaries[pos] += ncopy;
        last_len -= ncopy;
        if (last_len <= 0) {
            break;
        }
    }

    if (type != RangeSearchResult::SortType::None) {
        ret.SortImpl(type);
    }
    return ret;
}

void
RangeSearchResultHandler::Append(std::vector<BufferListPtr>& seg_result) {
    seg_results.emplace_back(std::move(seg_result));
}

// TODO: this API seems useless, should be removed
void
ExchangeDataset(std::vector<BufferListPtr>& dst, std::vector<faiss::RangeSearchPartialResult*>& src) {
    for (auto& s : src) {
        auto bptr = std::make_shared<BufferList>(s->res->buffer_size);
        bptr->wp = s->wp;
        bptr->buffers.resize(s->buffers.size());
        for (auto i = 0; i < s->buffers.size(); ++i) {
            bptr->buffers[i].ids = s->buffers[i].ids;
            bptr->buffers[i].dis = s->buffers[i].dis;
            s->buffers[i].ids = nullptr;
            s->buffers[i].dis = nullptr;
        }
        delete s->res;
        delete s;
        dst.push_back(bptr);
    }
}

}  // namespace knowhere
