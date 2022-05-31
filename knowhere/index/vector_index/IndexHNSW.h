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

#include "hnswlib/hnswlib/hnswlib.h"

#include "knowhere/common/Exception.h"
#include "knowhere/index/VecIndex.h"

namespace knowhere {

class IndexHNSW : public VecIndex {
 public:
    IndexHNSW() {
        index_type_ = IndexEnum::INDEX_HNSW;
        stats = std::make_shared<LibHNSWStatistics>(index_type_);
    }

    BinarySet
    Serialize(const Config&) override;

    void
    Load(const BinarySet&) override;

    void
    Train(const DatasetPtr&, const Config&) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    GetVectorById(const DatasetPtr&, const Config&) override;

    DatasetPtr
    Query(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

    DatasetPtr
    QueryByRange(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    int64_t
    Size() override;

#if 0
    void
    ClearStatistics() override;
#endif

 private:
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

}  // namespace knowhere
