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

#pragma once

#include <optional>
#include <queue>
#include <utility>

namespace knowhere {

// Maintain intermediate top-k results via maxheap
// TODO: this naive implementation might be optimzed later
//     1. Based on top-k and pushed element count to swtich strategy
//     2. Combine `pop` and `push` operation to `replace`
template <typename DisT, typename IdT>
class ResultMaxHeap {
 public:
    explicit ResultMaxHeap(size_t k) : k_(k) {}

    inline std::optional<std::pair<DisT, IdT>>
    Pop() {
        if (pq.empty()) {
            return std::nullopt;
        }
        std::optional<std::pair<DisT, IdT>> res = pq.top();
        pq.pop();
        return res;
    }

    inline void
    Push(DisT dis, IdT id) {
        if (pq.size() < k_) {
            pq.emplace(dis, id);
            return;
        }

        if (dis < pq.top().first) {
            pq.pop();
            pq.emplace(dis, id);
        }
    }

    inline size_t
    Size() {
        return pq.size();
    }

 private:
    size_t k_;
    std::priority_queue<std::pair<DisT, IdT>> pq;
};

}  // namespace knowhere
