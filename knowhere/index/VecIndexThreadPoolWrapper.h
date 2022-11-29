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

#include <memory>
#include <utility>

#include "knowhere/common/ThreadPool.h"
#include "knowhere/index/VecIndex.h"

namespace knowhere {

/**
 * @brief This class is a Wrapper for VecIndex, it will use a global thread pool for all Query and RangeQuery API calls.
 * 
 */
class VecIndexThreadPoolWrapper : public VecIndex {
 public:
    explicit VecIndexThreadPoolWrapper(std::unique_ptr<VecIndex> index)
        : VecIndexThreadPoolWrapper(std::move(index), ThreadPool::GetGlobalThreadPool()) {
    }

    explicit VecIndexThreadPoolWrapper(std::unique_ptr<VecIndex> index, std::shared_ptr<ThreadPool> thread_pool)
        : index_(std::move(index)), thread_pool_(thread_pool) {
    }

    BinarySet
    Serialize(const Config& config) override {
        return index_->Serialize(config);
    }

    void
    Load(const BinarySet& index_binary) override {
        index_->Load(index_binary);
    }

    void
    Train(const DatasetPtr& dataset, const Config& config) override {
        index_->Train(dataset, config);
    }

    void
    AddWithoutIds(const DatasetPtr& dataset, const Config& config) override {
        index_->AddWithoutIds(dataset, config);
    }

    bool
    Prepare(const Config& config) override {
        return index_->Prepare(config);
    }

    DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config) override {
        return index_->GetVectorById(dataset, config);
    }

    DatasetPtr
    Query(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) override {
        return thread_pool_->push([&]() { return this->index_->Query(dataset, config, bitset); }).get();
    }

    DatasetPtr
    QueryByRange(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) override {
        return thread_pool_->push([&]() { return this->index_->QueryByRange(dataset, config, bitset); }).get();
    }

    DatasetPtr
    GetIndexMeta(const Config& config) override {
        return index_->GetIndexMeta(config);
    }

    int64_t
    Size() override {
        return index_->Size();
    }

    int64_t
    Dim() override {
        return index_->Dim();
    }

    int64_t
    Count() override {
        return index_->Count();
    }

    StatisticsPtr
    GetStatistics() override {
        return index_->GetStatistics();
    }

    void
    ClearStatistics() override {
        index_->ClearStatistics();
    }

    IndexType
    index_type() const override {
        return index_->index_type();
    }

    IndexMode
    index_mode() const override {
        return index_->index_mode();
    }

 private:
    std::unique_ptr<VecIndex> index_;
    std::shared_ptr<ThreadPool> thread_pool_;
};

}  // namespace knowhere
