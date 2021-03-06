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
#include <utility>

#include <faiss/IndexBinary.h>

#include "knowhere/common/BinarySet.h"
#include "knowhere/common/Dataset.h"
#include "knowhere/index/IndexType.h"

namespace knowhere {

class FaissBaseBinaryIndex {
 protected:
    explicit FaissBaseBinaryIndex(std::shared_ptr<faiss::IndexBinary> index) : index_(std::move(index)) {
    }

    virtual BinarySet
    SerializeImpl(const IndexType& type);

    virtual void
    LoadImpl(const BinarySet& index_binary, const IndexType& type);

 public:
    std::shared_ptr<faiss::IndexBinary> index_ = nullptr;
};

}  // namespace knowhere
