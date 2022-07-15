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
#include <string>

#include "DiskANN/include/pq_flash_index.h"
#include "knowhere/common/FileManager.h"
#include "knowhere/index/VecIndex.h"

namespace knowhere {

template <typename T>
class IndexDiskANN : public VecIndex {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "DiskANN only support float, int8 and uint8");

 public:
    IndexDiskANN(std::string index_prefix, MetricType metric_type, std::unique_ptr<FileManager> file_manager);

    IndexDiskANN(const IndexDiskANN& index_diskann) = delete;

    IndexDiskANN&
    operator=(const IndexDiskANN& index_diskann) = delete;

    BinarySet
    Serialize(const Config& /* unused */) override {
        KNOWHERE_THROW_MSG("DiskANN doesn't support Seialize.");
    }

    void
    Load(const BinarySet& /* unused */) override {
        KNOWHERE_THROW_MSG("DiskANN doesn't support Load.");
    }

    /**
     * @brief Due to legacy reasons, Train() and AddWithoutIds() are bonded together. We will put the building work in
     * AddWIthoutIds() and leave Train() empty for now.
     */
    void
    Train(const DatasetPtr& /* unused */, const Config& /* unused */) override{};

    /**
     * @brief This API will build DiskANN.
     */
    void
    AddWithoutIds(const DatasetPtr& data_set, const Config& config) override;

    bool
    BuildDiskIndex(const Config& config) override;

    bool
    Prepare(const Config& config) override;

    DatasetPtr
    GetVectorById(const DatasetPtr& /* unused */, const Config& /* unused */) override {
        KNOWHERE_THROW_MSG("DiskANN doesn't support GetVectorById.");
    }

    DatasetPtr
    Query(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

    DatasetPtr
    QueryByRange(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    int64_t
    Size() override {
        KNOWHERE_THROW_MSG("DiskANN doesn't support Size for now.");
    }

    bool
    IsPrepared() {
        return is_prepared_;
    }

 private:
    std::string index_prefix_;

    diskann::Metric metric_;
    std::unique_ptr<FileManager> file_manager_;

    std::unique_ptr<diskann::PQFlashIndex<T>> pq_flash_index_;

    bool is_prepared_ = false;
    std::mutex preparation_lock_;
};

}  // namespace knowhere
