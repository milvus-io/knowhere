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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "index/vector_index/IndexRHNSWFlat.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace knowhere {

IndexRHNSWFlat::IndexRHNSWFlat(int d, int M, MetricType metric) {
    faiss::MetricType mt = GetFaissMetricType(metric);
    index_ = std::shared_ptr<faiss::Index>(new faiss::IndexRHNSWFlat(d, M, mt));
}

BinarySet
IndexRHNSWFlat::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        auto res_set = IndexRHNSW::Serialize(config);
        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get());
        if (real_idx == nullptr) {
            KNOWHERE_THROW_MSG("index is not a faiss::IndexRHNSWFlat");
        }

        int64_t meta_info[3] = {real_idx->storage->metric_type, real_idx->storage->d, real_idx->storage->ntotal};
        auto meta_space = new uint8_t[sizeof(meta_info)];
        memcpy(meta_space, meta_info, sizeof(meta_info));
        std::shared_ptr<uint8_t[]> space_sp(meta_space, std::default_delete<uint8_t[]>());
        res_set.Append("META", space_sp, sizeof(meta_info));

        Disassemble(res_set, config);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSWFlat::Load(const BinarySet& index_binary) {
    try {
        Assemble(const_cast<BinarySet&>(index_binary));
        IndexRHNSW::Load(index_binary);

        int64_t meta_info[3];  // = {metric_type, dim, ntotal}
        auto meta_data = index_binary.GetByName("META");
        memcpy(meta_info, meta_data->data.get(), meta_data->size);

        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get());
        real_idx->storage =
            new faiss::IndexFlat(static_cast<faiss::idx_t>(meta_info[1]), static_cast<faiss::MetricType>(meta_info[0]));
        auto binary_data = index_binary.GetByName(RAW_DATA);
        real_idx->storage->add(meta_info[2], reinterpret_cast<const float*>(binary_data->data.get()));
        real_idx->init_hnsw();
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSWFlat::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    try {
        GET_TENSOR_DATA_DIM(dataset_ptr)
        faiss::MetricType metric_type = GetFaissMetricType(config);
        int32_t efConstruction = GetIndexParamEfConstruction(config);
        int32_t hnsw_m = GetIndexParamHNSWM(config);

        auto idx = new faiss::IndexRHNSWFlat(int(dim), hnsw_m, metric_type);
        idx->hnsw.efConstruction = efConstruction;
        index_ = std::shared_ptr<faiss::Index>(idx);
        index_->train(rows, reinterpret_cast<const float*>(p_data));
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

int64_t
IndexRHNSWFlat::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return dynamic_cast<faiss::IndexRHNSWFlat*>(index_.get())->cal_size();
}

}  // namespace knowhere
