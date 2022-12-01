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

#include <string>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/MetaIndexes.h>

#include "common/Exception.h"
#include "common/Utils.h"
#include "index/vector_index/IndexBinaryIDMAP.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/RangeUtil.h"

namespace knowhere {

BinarySet
BinaryIDMAP::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    auto ret = SerializeImpl(index_type_);
    return ret;
}

void
BinaryIDMAP::Load(const BinarySet& index_binary) {
    LoadImpl(index_binary, index_type_);
}

DatasetPtr
BinaryIDMAP::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    uint8_t* p_x = nullptr;
    auto release_when_exception = [&]() {
        if (p_x != nullptr) {
            delete[] p_x;
        }
    };

    try {
        p_x = new uint8_t[(dim / 8) * rows];
        auto bin_idmap_index = dynamic_cast<faiss::IndexBinaryFlat*>(index_.get());
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < bin_idmap_index->ntotal, "invalid id %ld", id);
            bin_idmap_index->reconstruct(id, p_x + i * dim / 8);
        }
        return GenResultDataset(p_x);
    } catch (faiss::FaissException& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
BinaryIDMAP::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GET_TENSOR_DATA(dataset_ptr)

    utils::SetQueryOmpThread(config);
    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    auto release_when_exception = [&]() {
        if (p_id != nullptr) {
            delete[] p_id;
        }
        if (p_dist != nullptr) {
            delete[] p_dist;
        }
    };

    try {
        auto k = GetMetaTopk(config);
        p_id = new int64_t[k * rows];
        p_dist = new float[k * rows];

        QueryImpl(rows, reinterpret_cast<const uint8_t*>(p_data), k, p_dist, p_id, config, bitset);

        return GenResultDataset(p_id, p_dist);
    } catch (faiss::FaissException& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
BinaryIDMAP::QueryByRange(const DatasetPtr& dataset,
                          const Config& config,
                          const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GET_TENSOR_DATA(dataset)

    utils::SetQueryOmpThread(config);

    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    size_t* p_lims = nullptr;

    auto release_when_exception = [&]() {
        if (p_id != nullptr) {
            delete[] p_id;
        }
        if (p_dist != nullptr) {
            delete[] p_dist;
        }
        if (p_lims != nullptr) {
            delete[] p_lims;
        }
    };

    try {
        QueryByRangeImpl(rows, reinterpret_cast<const uint8_t*>(p_data), p_dist, p_id, p_lims, config, bitset);
        return GenResultDataset(p_id, p_dist, p_lims);
    } catch (faiss::FaissException& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        release_when_exception();
        KNOWHERE_THROW_MSG(e.what());
    }
}

int64_t
BinaryIDMAP::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
BinaryIDMAP::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

void
BinaryIDMAP::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const uint8_t*>(p_data));
}

void
BinaryIDMAP::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetBuildOmpThread(config);
    faiss::MetricType metric_type = GetFaissMetricType(config);
    auto index = std::make_shared<faiss::IndexBinaryFlat>(dim, metric_type);
    index_ = index;
}

void
BinaryIDMAP::QueryImpl(int64_t n,
                       const uint8_t* data,
                       int64_t k,
                       float* distances,
                       int64_t* labels,
                       const Config& config,
                       const faiss::BitsetView bitset) {
    auto i_distances = reinterpret_cast<int32_t*>(distances);
    index_->search(n, data, k, i_distances, labels, bitset);

    // convert int32 to float for hamming
    if (index_->metric_type == faiss::METRIC_Hamming) {
        int64_t num = n * k;
        for (int64_t i = 0; i < num; i++) {
            distances[i] = static_cast<float>(i_distances[i]);
        }
    }
}

void
BinaryIDMAP::QueryByRangeImpl(int64_t n,
                              const uint8_t* data,
                              float*& distances,
                              int64_t*& labels,
                              size_t*& lims,
                              const Config& config,
                              const faiss::BitsetView bitset) {
    auto index = dynamic_cast<faiss::IndexBinaryFlat*>(index_.get());
    float low_bound = GetMetaRadiusLowBound(config);
    float high_bound = GetMetaRadiusHighBound(config);

    faiss::RangeSearchResult res(n);
    index->range_search(n, data, high_bound, &res, bitset);
    GetRangeSearchResult(res, false, n, low_bound, high_bound, distances, labels, lims, bitset);
}

}  // namespace knowhere
