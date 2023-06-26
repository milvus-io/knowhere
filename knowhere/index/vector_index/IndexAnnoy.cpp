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

#include "IndexAnnoy.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace knowhere {

BinarySet
IndexAnnoy::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto metric_type_length = metric_type_.length();
    std::shared_ptr<uint8_t[]> metric_type(new uint8_t[metric_type_length]);
    memcpy(metric_type.get(), metric_type_.data(), metric_type_.length());

    auto dim = Dim();
    std::shared_ptr<uint8_t[]> dim_data(new uint8_t[sizeof(uint64_t)]);
    memcpy(dim_data.get(), &dim, sizeof(uint64_t));

    size_t index_length = index_->get_index_length();
    std::shared_ptr<uint8_t[]> index_data(new uint8_t[index_length]);
    memcpy(index_data.get(), index_->get_index(), index_length);

    BinarySet res_set;
    res_set.Append("annoy_metric_type", metric_type, metric_type_length);
    res_set.Append("annoy_dim", dim_data, sizeof(uint64_t));
    res_set.Append("annoy_index_data", index_data, index_length);
    return res_set;
}

void
IndexAnnoy::Load(const BinarySet& index_binary) {
    auto metric_type = index_binary.GetByName("annoy_metric_type");
    metric_type_.resize(static_cast<size_t>(metric_type->size));
    memcpy(metric_type_.data(), metric_type->data.get(), static_cast<size_t>(metric_type->size));

    auto dim_data = index_binary.GetByName("annoy_dim");
    uint64_t dim;
    memcpy(&dim, dim_data->data.get(), static_cast<size_t>(dim_data->size));

    if (metric_type_ == metric::L2) {
        index_ = std::make_shared<AnnoyIndex<int64_t, float, ::Euclidean, ::Kiss64Random, ThreadedBuildPolicy>>(dim);
    } else if (metric_type_ == metric::IP) {
        index_ = std::make_shared<AnnoyIndex<int64_t, float, ::DotProduct, ::Kiss64Random, ThreadedBuildPolicy>>(dim);
    } else {
        KNOWHERE_THROW_MSG("metric not supported " + metric_type_);
    }

    auto index_data = index_binary.GetByName("annoy_index_data");
    char* p = nullptr;
    if (!index_->load_index(reinterpret_cast<void*>(index_data->data.get()), index_data->size, &p)) {
        std::string error_msg(p);
        free(p);
        KNOWHERE_THROW_MSG(error_msg);
    }
}

void
IndexAnnoy::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    try {
        GET_TENSOR_DATA_DIM(dataset_ptr)
        metric_type_ = GetMetaMetricType(config);
        if (metric_type_ == metric::L2) {
            index_ =
                std::make_shared<AnnoyIndex<int64_t, float, ::Euclidean, ::Kiss64Random, ThreadedBuildPolicy>>(dim);
        } else if (metric_type_ == metric::IP) {
            index_ =
                std::make_shared<AnnoyIndex<int64_t, float, ::DotProduct, ::Kiss64Random, ThreadedBuildPolicy>>(dim);
        } else {
            KNOWHERE_THROW_MSG("metric not supported " + metric_type_);
        }
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
    is_build_ = false;
}

void
IndexAnnoy::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    // Annoy does not support `add` function, multiple calls will be ignored, same behaviour as before
    if (is_build_) {
        LOG_KNOWHERE_DEBUG_ << "IndexAnnoy::AddWithoutIds: index_ has been built! "
                            << "Annoy not support build item dynamically, please invoke BuildAll interface.";
        return;
    }

    GET_TENSOR_DATA_DIM(dataset_ptr)
    utils::SetBuildOmpThread(config);
    for (int i = 0; i < rows; ++i) {
        index_->add_item(i, static_cast<const float*>(p_data) + dim * i);
    }
    index_->build(GetIndexParamNtrees(config));
    is_build_ = true;
}

DatasetPtr
IndexAnnoy::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_DATA_WITH_IDS(dataset_ptr)
    auto dim = Dim();

    float* p_x = nullptr;
    try {
        p_x = new float[dim * rows];
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < index_->get_n_items(), "invalid id %ld", id);
            index_->get_item(id, p_x + i * dim);
        }
    } catch (std::exception& e) {
        if (p_x != nullptr) {
            delete[] p_x;
        }
        KNOWHERE_THROW_MSG(e.what());
    }
    return GenResultDataset(rows, dim, p_x);
}

DatasetPtr
IndexAnnoy::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetQueryOmpThread(config);
    auto k = GetMetaTopk(config);
    auto search_k = GetIndexParamSearchK(config);
    auto p_id = new int64_t[k * rows];
    auto p_dist = new float[k * rows];

    std::vector<std::future<void>> futures;
    futures.reserve(rows);
    for (unsigned int i = 0; i < rows; ++i) {
        futures.push_back(pool_->push([&, index = i]() {
            std::vector<int64_t> result;
            result.reserve(k);
            std::vector<float> distances;
            distances.reserve(k);
            index_->get_nns_by_vector(static_cast<const float*>(p_data) + index * dim, k, search_k, &result, &distances,
                                    bitset);

            size_t result_num = result.size();
            auto local_p_id = p_id + k * index;
            auto local_p_dist = p_dist + k * index;
            memcpy(local_p_id, result.data(), result_num * sizeof(int64_t));
            memcpy(local_p_dist, distances.data(), result_num * sizeof(float));

            for (; result_num < k; result_num++) {
                local_p_id[result_num] = -1;
                local_p_dist[result_num] = 1.0 / 0.0;
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    return GenResultDataset(p_id, p_dist);
}

int64_t
IndexAnnoy::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->get_n_items();
}

int64_t
IndexAnnoy::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->get_dim();
}

int64_t
IndexAnnoy::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->cal_size();
}

}  // namespace knowhere
