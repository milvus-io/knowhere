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
#include <mutex>
#include <utility>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/FaissBaseIndex.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#include <faiss/index_io.h>
#include <faiss/IndexRHNSW.h>

namespace knowhere {

class IndexRHNSW : public VecIndex, public FaissBaseIndex {
 public:
    IndexRHNSW() : FaissBaseIndex(nullptr) {
        index_type_ = IndexEnum::INVALID;
        stats = std::make_shared<RHNSWStatistics>(index_type_);
    }

    explicit IndexRHNSW(std::shared_ptr<faiss::Index> index) : FaissBaseIndex(std::move(index)) {
        index_type_ = IndexEnum::INVALID;
        stats = std::make_shared<RHNSWStatistics>(index_type_);
    }

    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& index_binary) override;

    void
    Train(const DatasetPtr& dataset_ptr, const Config& config) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    void
    UpdateIndexSize() override;

#if 0
    StatisticsPtr
    GetStatistics() override;

    void
    ClearStatistics() override;
#endif
};

}  // namespace knowhere
