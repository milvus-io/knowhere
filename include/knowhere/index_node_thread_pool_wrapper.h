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

#ifndef INDEX_NODE_THREAD_POOL_WRAPPER_H
#define INDEX_NODE_THREAD_POOL_WRAPPER_H

#include "knowhere/comp/thread_pool.h"
#include "knowhere/index_node.h"

namespace knowhere {

class IndexNodeThreadPoolWrapper : public IndexNode {
 public:
    explicit IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node)
        : IndexNodeThreadPoolWrapper(std::move(index_node), ThreadPool::GetGlobalThreadPool()) {
    }

    explicit IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node, std::shared_ptr<ThreadPool> thread_pool)
        : index_node_(std::move(index_node)), thread_pool_(thread_pool) {
    }

    Status
    Build(const DataSet& dataset, const Config& cfg) {
        return index_node_->Build(dataset, cfg);
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) {
        return index_node_->Train(dataset, cfg);
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) {
        return index_node_->Add(dataset, cfg);
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
        return thread_pool_->push([&]() { return this->index_node_->Search(dataset, cfg, bitset); }).get();
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
        return thread_pool_->push([&]() { return this->index_node_->RangeSearch(dataset, cfg, bitset); }).get();
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset) const {
        return index_node_->GetVectorByIds(dataset);
    }

    bool
    HasRawData(const std::string& metric_type) const {
        return index_node_->HasRawData(metric_type);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const {
        return index_node_->GetIndexMeta(cfg);
    }

    Status
    Serialize(BinarySet& binset) const {
        return index_node_->Serialize(binset);
    }

    Status
    Deserialize(const BinarySet& binset) {
        return index_node_->Deserialize(binset);
    }

    Status
    DeserializeFromFile(const std::string& filename, const LoadConfig& config) {
        return index_node_->DeserializeFromFile(filename, config);
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const {
        return index_node_->CreateConfig();
    }

    int64_t
    Dim() const {
        return index_node_->Dim();
    }

    int64_t
    Size() const {
        return index_node_->Size();
    }

    int64_t
    Count() const {
        return index_node_->Count();
    }

    std::string
    Type() const {
        return index_node_->Type();
    }

 private:
    std::unique_ptr<IndexNode> index_node_;
    std::shared_ptr<ThreadPool> thread_pool_;
};

}  // namespace knowhere

#endif /* INDEX_NODE_THREAD_POOL_WRAPPER_H */
