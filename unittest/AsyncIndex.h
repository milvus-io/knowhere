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

#include <future>

#include "common/Log.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/IndexAnnoy.h"
#include "knowhere/index/vector_index/IndexBinaryIDMAP.h"
#include "knowhere/index/vector_index/IndexBinaryIVF.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/IndexIVFSQ.h"
#ifdef KNOWHERE_WITH_DISKANN
#include "LocalFileManager.h"
#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#endif

namespace knowhere {

class AsyncIndex : public VecIndex {
 public:
    AsyncIndex(std::string type) {
        std::transform(type.begin(), type.end(), type.begin(), toupper);
        if (type == IndexEnum::INDEX_ANNOY) {
            index_ = std::make_unique<knowhere::IndexAnnoy>();
        } else if (type == IndexEnum::INDEX_FAISS_BIN_IDMAP) {
            index_ = std::make_unique<knowhere::BinaryIDMAP>();
        } else if (type == IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
            index_ = std::make_unique<knowhere::BinaryIVF>();
        } else if (type == IndexEnum::INDEX_FAISS_IDMAP) {
            index_ = std::make_unique<knowhere::IDMAP>();
        } else if (type == IndexEnum::INDEX_HNSW) {
            index_ = std::make_unique<knowhere::IndexHNSW>();
        } else if (type == IndexEnum::INDEX_FAISS_IVFFLAT) {
            index_ = std::make_unique<knowhere::IVF>();
        } else if (type == IndexEnum::INDEX_FAISS_IVFPQ) {
            index_ = std::make_unique<knowhere::IVFPQ>();
        } else if (type == IndexEnum::INDEX_FAISS_IVFSQ8) {
            index_ = std::make_unique<knowhere::IVFSQ>();
        } else {
            KNOWHERE_THROW_FORMAT("Invalid index type %s", std::string(type).c_str());
        }
    }

#ifdef KNOWHERE_WITH_DISKANN
    AsyncIndex(std::string type, std::string index_prefix, std::string metric_type) {
        std::transform(metric_type.begin(), metric_type.end(), metric_type.begin(), toupper);
        if (type == "diskann_f") {
            index_ = std::make_unique<knowhere::IndexDiskANN<float>>(index_prefix, metric_type,
                                                                     std::make_shared<LocalFileManager>());
        } else if (type == "disann_ui8") {
            index_ = std::make_unique<knowhere::IndexDiskANN<uint8_t>>(index_prefix, metric_type,
                                                                       std::make_shared<LocalFileManager>());
        } else if (type == "diskann_i8") {
            index_ = std::make_unique<knowhere::IndexDiskANN<int8_t>>(index_prefix, metric_type,
                                                                      std::make_shared<LocalFileManager>());
        } else {
            KNOWHERE_THROW_FORMAT("Invalid index type %s", std::string(type).c_str());
        }
    }
    bool
    Prepare(const Config& config) {
        return index_->Prepare(config);
    }

#endif

    std::vector<std::future<DatasetPtr>>&
    GetFeatureStack() {
        static std::vector<std::future<DatasetPtr>> stk;
        return stk;
    }

    void
    QueryAsync(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) {
        std::vector<std::future<DatasetPtr>>& stk = GetFeatureStack();
        stk.emplace_back(std::async(std::launch::async, [this, &dataset, &config, bitset]() {
            return index_->Query(dataset, config, bitset);
        }));
    }

    DatasetPtr
    Sync() {
        std::vector<std::future<DatasetPtr>>& stk = GetFeatureStack();
        assert(!stk.empty());
        auto res = stk.back().get();
        stk.pop_back();
        return res;
    }

    BinarySet
    Serialize(const Config& config) override {
        return index_->Serialize(config);
    }

    void
    Load(const BinarySet& index_binary) override {
        return index_->Load(index_binary);
    }

    void
    BuildAll(const DatasetPtr& dataset_ptr, const Config& config) override {
        index_->BuildAll(dataset_ptr, config);
    }

    void
    Train(const DatasetPtr& dataset_ptr, const Config& config) override {
        index_->Train(dataset_ptr, config);
    }

    void
    AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) override {
        return index_->AddWithoutIds(dataset_ptr, config);
    }

    DatasetPtr
    Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) override {
        return index_->Query(dataset_ptr, config, bitset);
    }

    int64_t
    Count() override {
        return index_->Count();
    }

    int64_t
    Dim() override {
        return index_->Dim();
    }

    int64_t
    Size() override {
        return index_->Size();
    }

 private:
    std::unique_ptr<VecIndex> index_;
};
}  // namespace knowhere
