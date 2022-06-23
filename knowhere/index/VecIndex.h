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

#include "knowhere/common/Dataset.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Typedef.h"
#include "knowhere/index/Index.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/Statistics.h"
#include "knowhere/index/vector_index/helpers/Slice.h"
#include "knowhere/utils/BitsetView.h"

namespace knowhere {

#define RAW_DATA "RAW_DATA"
#define QUANTIZATION_DATA "QUANTIZATION_DATA"

class VecIndex : public Index {
 public:
    virtual void
    BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
        Train(dataset_ptr, config);
        AddWithoutIds(dataset_ptr, config);
    }

    virtual void
    Train(const DatasetPtr& dataset, const Config& config) = 0;

    virtual void
    AddWithoutIds(const DatasetPtr& dataset, const Config& config) = 0;

    virtual DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config) {
        KNOWHERE_THROW_MSG("GetVectorById not supported yet");
    }

    virtual DatasetPtr
    Query(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) = 0;

    virtual DatasetPtr
    QueryByRange(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) {
        KNOWHERE_THROW_MSG("QueryByRange not supported yet");
    }

    virtual int64_t
    Dim() = 0;

    virtual int64_t
    Count() = 0;

    virtual StatisticsPtr
    GetStatistics() {
        return stats;
    }

    virtual void
    ClearStatistics() {
    }

    virtual IndexType
    index_type() const {
        return index_type_;
    }

    virtual IndexMode
    index_mode() const {
        return index_mode_;
    }

 protected:
    IndexType index_type_ = "";
    IndexMode index_mode_ = IndexMode::MODE_CPU;
    StatisticsPtr stats = nullptr;
};

using VecIndexPtr = std::shared_ptr<VecIndex>;

}  // namespace knowhere
