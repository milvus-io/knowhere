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

#include <list>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace knowhere {

template <typename key_t, typename value_t>
class lru_cache {
 public:
    using key_value_pair_t = std::pair<key_t, value_t>;
    using list_iterator_t = typename std::list<key_value_pair_t>::iterator;

    explicit lru_cache(size_t cap = kDefaultSize) : capacity(cap) {
    }

    void
    put(const key_t& key, const value_t& value) {
        std::unique_lock lk(mtx);
        auto it = map.find(key);
        list.push_front(key_value_pair_t(key, value));
        if (it != map.end()) {
            list.erase(it->second);
            map.erase(it);
        }
        map[key] = list.begin();
        if (map.size() > capacity) {
            auto last = list.end();
            last--;
            map.erase(last->first);
            list.pop_back();
        }
    }

    bool
    try_get(const key_t& key, value_t& val) {
        std::unique_lock lk(mtx);
        auto it = map.find(key);
        if (it == map.end()) {
            return false;
        } else {
            list.splice(list.begin(), list, it->second);
            val = it->second->second;
            return true;
        }
    }

    static uint64_t
    hash_vec(const float* x, size_t d) {
        uint64_t h = 0;
        for (size_t i = 0; i < d; ++i) {
            h = h * 13331 + *(uint32_t*)(x + i);
        }
        return h;
    }

 private:
    std::list<key_value_pair_t> list;
    std::unordered_map<key_t, list_iterator_t> map;
    size_t capacity;
    std::mutex mtx;
    constexpr static size_t kDefaultSize = 10000;
};

}  // namespace knowhere
