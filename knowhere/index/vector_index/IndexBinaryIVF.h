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
#include <vector>

#include <faiss/IndexIVF.h>

#include "knowhere/common/Exception.h"
#include "knowhere/index/VecIndex.h"
#include "knowhere/index/vector_index/FaissBaseBinaryIndex.h"

namespace knowhere {

class BinaryIVF : public VecIndex, public FaissBaseBinaryIndex {
 public:
    BinaryIVF() : FaissBaseBinaryIndex(nullptr) {
        index_type_ = IndexEnum::INDEX_FAISS_BIN_IVFFLAT;
        stats = std::make_shared<IVFStatistics>(index_type_);
    }

    explicit BinaryIVF(std::shared_ptr<faiss::IndexBinary> index) : FaissBaseBinaryIndex(std::move(index)) {
        index_type_ = IndexEnum::INDEX_FAISS_BIN_IVFFLAT;
        stats = std::make_shared<IVFStatistics>(index_type_);
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
    StatisticsPtr
    GetStatistics() override;

    void
    ClearStatistics() override;
#endif

 protected:
    virtual std::shared_ptr<faiss::IVFSearchParameters>
    GenParams(const Config& config);

    virtual void
    QueryImpl(int64_t n,
              const uint8_t* data,
              int64_t k,
              float* distances,
              int64_t* labels,
              const Config& config,
              const faiss::BitsetView bitset);

    virtual void
    QueryByRangeImpl(int64_t n,
                     const uint8_t* data,
                     float radius,
                     float*& distances,
                     int64_t*& labels,
                     size_t*& lims,
                     const Config& config,
                     const faiss::BitsetView bitset);
};

using BinaryIVFIndexPtr = std::shared_ptr<BinaryIVF>;

}  // namespace knowhere
