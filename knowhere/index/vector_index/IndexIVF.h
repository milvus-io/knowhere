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
#include <vector>

#include <faiss/IndexIVF.h>

#include "knowhere/common/Typedef.h"
#include "knowhere/index/VecIndex.h"
#include "knowhere/index/vector_index/FaissBaseIndex.h"

namespace knowhere {

class IVF : public VecIndex, public FaissBaseIndex {
 public:
    IVF() : FaissBaseIndex(nullptr) {
        index_type_ = IndexEnum::INDEX_FAISS_IVFFLAT;
        stats = std::make_shared<IVFStatistics>(index_type_);
    }

    explicit IVF(std::shared_ptr<faiss::Index> index) : FaissBaseIndex(std::move(index)) {
        index_type_ = IndexEnum::INDEX_FAISS_IVFFLAT;
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

    virtual void
    Seal();

    virtual VecIndexPtr
    CopyCpuToGpu(const int64_t, const Config&);

    virtual void
    GenGraph(const float* data, const int64_t k, GraphType& graph, const Config& config);

 protected:
    virtual std::shared_ptr<faiss::IVFSearchParameters>
    GenParams(const Config&);

    virtual void
    QueryImpl(int64_t, const float*, int64_t, float*, int64_t*, const Config&, const faiss::BitsetView);

    virtual void
    QueryByRangeImpl(int64_t, const float*, float, float*&, int64_t*&, size_t*&, const Config&,
                     const faiss::BitsetView);

    void
    SealImpl() override;
};

using IVFPtr = std::shared_ptr<IVF>;

}  // namespace knowhere
