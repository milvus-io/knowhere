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

#ifndef BINARYSET_H
#define BINARYSET_H

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace knowhere {

class BinarySet {
 public:
    BinarySet() : data_(nullptr), size_(0) {
    }

    BinarySet(std::unique_ptr<uint8_t[]>& data, uint64_t size) {
        Clear();
        Set(data, size);
    }

    template <typename T>
    BinarySet(T&& binset) {
        Clear();
        size_ = binset.size_;
        data_.swap(binset.data_);
        binset.size_ = 0;
    }

    template <typename T>
    BinarySet&
    operator=(T&& binset) {
        Clear();
        size_ = binset.size_;
        data_.swap(binset.data_);
        binset.size_ = 0;
        return *this;
    }

    void
    Set(std::unique_ptr<uint8_t[]>& data, uint64_t size) {
        data_ = std::move(data);
        size_ = size;
    }

    void
    Clear() {
        if (data_ != nullptr) {
            data_.reset(nullptr);
            size_ = 0;
        }
    }

    std::unique_ptr<uint8_t[]>
    Release() {
        size_ = 0;
        return std::move(data_);
    }

    const uint8_t*
    GetData() const {
        return data_.get();
    }

    const uint64_t
    GetSize() const {
        return size_;
    }

 private:
    std::unique_ptr<uint8_t[]> data_;
    uint64_t size_;
};

}  // namespace knowhere

#endif /* BINARYSET_H */
