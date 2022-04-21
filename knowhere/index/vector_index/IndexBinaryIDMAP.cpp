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
#include "index/vector_index/IndexBinaryIDMAP.h"
#include "index/vector_index/adapter/VectorAdapter.h"

namespace knowhere {

BinarySet
BinaryIDMAP::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    auto ret = SerializeImpl(index_type_);
    Disassemble(ret, config);
    return ret;
}

void
BinaryIDMAP::Load(const BinarySet& index_binary) {
    Assemble(const_cast<BinarySet&>(index_binary));
    LoadImpl(index_binary, index_type_);
}

DatasetPtr
BinaryIDMAP::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GET_TENSOR_DATA(dataset_ptr)

    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    auto release_when_exception = [&]() {
        if (p_id != nullptr) {
            free(p_id);
        }
        if (p_dist != nullptr) {
            free(p_dist);
        }
    };

    try {
        auto k = config[meta::TOPK].get<int64_t>();
        auto elems = rows * k;
        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        p_id = static_cast<int64_t*>(malloc(p_id_size));
        p_dist = static_cast<float*>(malloc(p_dist_size));

        QueryImpl(rows, reinterpret_cast<const uint8_t*>(p_data), k, p_dist, p_id, config, bitset);

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::IDS, p_id);
        ret_ds->Set(meta::DISTANCE, p_dist);
        return ret_ds;
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

    auto radius = config[meta::RADIUS].get<float>();

    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    size_t* p_lims = nullptr;

    auto release_when_exception = [&]() {
        if (p_id != nullptr) {
            free(p_id);
        }
        if (p_dist != nullptr) {
            free(p_dist);
        }
        if (p_lims != nullptr) {
            free(p_lims);
        }
    };

    try {
        QueryByRangeImpl(rows, reinterpret_cast<const uint8_t*>(p_data), radius, p_dist, p_id, p_lims, config, bitset);

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::IDS, p_id);
        ret_ds->Set(meta::DISTANCE, p_dist);
        ret_ds->Set(meta::LIMS, p_lims);
        return ret_ds;
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

    faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());
    auto index = std::make_shared<faiss::IndexBinaryFlat>(dim, metric_type);
    index_ = index;
}

const uint8_t*
BinaryIDMAP::GetRawVectors() {
    try {
        auto flat_index = dynamic_cast<faiss::IndexBinaryFlat*>(index_.get());
        return flat_index->xb.data();
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
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

    // for hamming, need to transform int32 to float
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
                              float radius,
                              float*& distances,
                              int64_t*& labels,
                              size_t*& lims,
                              const Config& config,
                              const faiss::BitsetView bitset) {
    auto binary_idmap_index = dynamic_cast<faiss::IndexBinaryFlat*>(index_.get());

    faiss::RangeSearchResult res(n);
    binary_idmap_index->range_search(n, data, radius, &res, bitset);

    distances = res.distances;
    labels = res.labels;
    lims = res.lims;

    res.distances = nullptr;
    res.labels = nullptr;
    res.lims = nullptr;
}

}  // namespace knowhere
