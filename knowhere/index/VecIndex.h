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
#include <climits>

#include "knowhere/common/Dataset.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Typedef.h"
#include "knowhere/common/Utils.h"
#include "knowhere/index/Index.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/Statistics.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/utils/BitsetView.h"
#ifdef KNOWHERE_WITH_DISKANN
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#endif

namespace knowhere {

#define RAW_DATA "RAW_DATA"
#define QUANTIZATION_DATA "QUANTIZATION_DATA"

const int64_t kSanityCheckNumberOfQueries = 1;

class VecIndex : public Index {
 public:
    virtual void
    BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
        Train(dataset_ptr, config);
        AddWithoutIds(dataset_ptr, config);

        // sanity check
        auto dim_on_storage = Dim();
        Config sanity_check_config = GenSanityCheckConfig(config);
        if (IndexEnum::INDEX_FAISS_BIN_IDMAP == index_type_ || IndexEnum::INDEX_FAISS_BIN_IVFFLAT == index_type_) {
            auto num_bits = CHAR_BIT * sizeof(float);
            dim_on_storage = (dim_on_storage + num_bits - 1) / num_bits;
        }

#ifdef KNOWHERE_WITH_DISKANN
        if (IndexEnum::INDEX_DISKANN == index_type_) {
            sanity_check_config = GenSanityCheckDiskANNConfig(sanity_check_config);
            Prepare(sanity_check_config);
        }
#endif
        std::vector<float> query_data(dim_on_storage, 0);
        auto query_dataset = GenDataset(kSanityCheckNumberOfQueries, Dim(), query_data.data());
        Query(query_dataset, sanity_check_config, nullptr);
    }

    virtual void
    Train(const DatasetPtr& dataset, const Config& config) = 0;

    virtual void
    AddWithoutIds(const DatasetPtr& dataset, const Config& config) = 0;

    /**
     * @brief Prepare the Index ready for query.
     *
     * @return true if the index is well prepared.
     * @return false if any error.
     */
    virtual bool
    Prepare(const Config& /* unused */) {
        KNOWHERE_THROW_MSG("Prepare not supported yet");
    }

    virtual DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config) {
        KNOWHERE_THROW_MSG("GetVectorById not supported yet");
    }

    /**
     * @brief TopK Query. if the result size is smaller than K, this API will fill the return ids with -1 and distances
     * with FLOAT_MIN or FLOAT_MAX depends on the metric type.
     *
     */
    virtual DatasetPtr
    Query(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) = 0;

    virtual DatasetPtr
    QueryByRange(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) {
        KNOWHERE_THROW_MSG("QueryByRange not supported yet");
    }

    virtual DatasetPtr
    GetIndexMeta(const Config& config) {
        KNOWHERE_THROW_MSG("GetIndexMeta not supported yet");
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
