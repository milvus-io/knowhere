// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

using Key = std::string_view;
using Value = std::any;
using ValuePtr = std::shared_ptr<Value>;

class Dataset {
 public:
    Dataset() = default;
    ~Dataset() {
        for (auto const& d : data_) {
            if (d.first == meta::IDS) {
                auto ids = Get<const int64_t*>(meta::IDS);
                // the space of ids must be allocated through malloc
                free((void*)ids);
            }
            if (d.first == meta::DISTANCE) {
                auto distances = Get<const float*>(meta::DISTANCE);
                // the space of distance must be allocated through malloc
                free((void*)distances);
            }
            if (d.first == meta::LIMS) {
                auto lims = Get<const size_t*>(meta::LIMS);
                // the space of lims must be allocated through malloc
                free((void*)lims);
            }
        }
    }

    template <typename T>
    void
    Set(const Key& k, T&& v) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[k] = std::make_shared<Value>(std::forward<T>(v));
    }

    template <typename T>
    T
    Get(const Key& k) {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<T>(*(data_.at(k)));
    }

 private:
    std::mutex mutex_;
    std::unordered_map<Key, ValuePtr, std::hash<Key>, std::equal_to<>> data_;
};
using DatasetPtr = std::shared_ptr<Dataset>;

}  // namespace knowhere
