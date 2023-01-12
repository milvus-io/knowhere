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

#include "index/vector_index/IndexHNSW.h"

#include <algorithm>
#include <chrono>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"
#include "index/vector_index/helpers/RangeUtil.h"

namespace knowhere {

BinarySet
IndexHNSW::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        MemoryIOWriter writer;
        index_->saveIndex(writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        res_set.Append("HNSW", data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::Load(const BinarySet& index_binary) {
    try {
        auto binary = index_binary.GetByName("HNSW");

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        hnswlib::SpaceInterface<float>* space = nullptr;
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space);
        index_->loadIndex(reader);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    try {
        GET_TENSOR_DATA_DIM(dataset_ptr)

        hnswlib::SpaceInterface<float>* space;
        std::string metric_type = GetMetaMetricType(config);
        if (metric_type == metric::L2) {
            space = new hnswlib::L2Space(dim);
        } else if (metric_type == metric::IP) {
            space = new hnswlib::InnerProductSpace(dim);
        } else {
            KNOWHERE_THROW_MSG("Metric type not supported: " + metric_type);
        }
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, rows, GetIndexParamHNSWM(config),
                                                                   GetIndexParamEfConstruction(config));
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)
    utils::SetBuildOmpThread(config);
    index_->addPoint(p_data, 0);

#pragma omp parallel for
    for (int i = 1; i < rows; ++i) {
        index_->addPoint((reinterpret_cast<const float*>(p_data) + Dim() * i), i);
    }
}

DatasetPtr
IndexHNSW::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    float* p_x = nullptr;
    try {
        p_x = new float[dim * rows];
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < index_->cur_element_count, "invalid id %ld", id);
            memcpy(p_x + i * dim, index_->getDataByInternalId(id), dim * sizeof(float));
        }
    } catch (std::exception& e) {
        if (p_x != nullptr) {
            delete[] p_x;
        }
        KNOWHERE_THROW_MSG(e.what());
    }
    return GenResultDataset(p_x);
}

DatasetPtr
IndexHNSW::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA_DIM(dataset_ptr)

    auto k = GetMetaTopk(config);
    auto p_id = new int64_t[k * rows];
    auto p_dist = new float[k * rows];

    feder::hnsw::FederResultUniq feder_result;

    QueryImpl(rows, reinterpret_cast<const float*>(p_data), k, p_dist, p_id, feder_result, config, bitset);

    // set visit_info json string into result dataset
    if (feder_result != nullptr) {
        Config json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        return GenResultDataset(p_id, p_dist, json_visit_info.dump(), json_id_set.dump());
    }

    return GenResultDataset(p_id, p_dist);
}

DatasetPtr
IndexHNSW::QueryByRange(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA_DIM(dataset)

    utils::SetQueryOmpThread(config);

    int64_t* p_id = nullptr;
    float* p_dist = nullptr;
    size_t* p_lims = nullptr;

    feder::hnsw::FederResultUniq feder_result;
    QueryByRangeImpl(rows, reinterpret_cast<const float*>(p_data), p_dist, p_id, p_lims, feder_result, config, bitset);

    // set visit_info json string into result dataset
    if (feder_result != nullptr) {
        Config json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        return GenResultDataset(p_id, p_dist, p_lims, json_visit_info.dump(), json_id_set.dump());
    }
    return GenResultDataset(p_id, p_dist, p_lims);
}

DatasetPtr
IndexHNSW::GetIndexMeta(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialized or trained");
    }

    const int64_t default_index_overview_levels = 3;
    auto overview_levels = CheckKeyInConfig(config, indexparam::OVERVIEW_LEVELS) ? GetIndexParamOverviewLevels(config)
                                                                                 : default_index_overview_levels;
    feder::hnsw::HNSWMeta meta(index_->ef_construction_, index_->M_, index_->cur_element_count, index_->maxlevel_,
                               index_->enterpoint_node_, overview_levels);
    std::unordered_set<int64_t> id_set;

    for (int i = 0; i < overview_levels; i++) {
        int64_t level = index_->maxlevel_ - i;
        // do not record level 0
        if (level <= 0)
            break;
        meta.AddLevelLinkGraph(level);
        UpdateLevelLinkList(level, meta, id_set);
    }

    Config json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataset(json_meta.dump(), json_id_set.dump());
}

int64_t
IndexHNSW::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->cur_element_count;
}

int64_t
IndexHNSW::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return (*static_cast<size_t*>(index_->dist_func_param_));
}

int64_t
IndexHNSW::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->cal_size();
}

void
IndexHNSW::QueryImpl(int64_t n, const float* xq, int64_t k, float* distances, int64_t* labels,
                     feder::hnsw::FederResultUniq& feder, const Config& config, const faiss::BitsetView bitset) {
    if (CheckKeyInConfig(config, meta::TRACE_VISIT) && GetMetaTraceVisit(config)) {
        if (n != 1) {
            delete[] labels;
            delete[] distances;
            KNOWHERE_THROW_MSG("NQ must be 1 when Feder tracing");
        }
        feder = std::make_unique<feder::hnsw::FederResult>();
    }

    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};
    bool transform = (index_->metric_type_ == 1);  // InnerProduct: 1

    std::vector<std::future<void>> futures;
    futures.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
        futures.push_back(pool_->push([&, index = i]() {
            auto single_query = xq + index * Dim();
            auto rst = index_->searchKnn(single_query, k, bitset, &param, feder);
            size_t rst_size = rst.size();
            auto p_single_dis = distances + index * k;
            auto p_single_id = labels + index * k;
            size_t idx = rst_size - 1;
            while (!rst.empty()) {
                auto& it = rst.top();
                p_single_dis[idx] = transform ? (1 - it.first) : it.first;
                p_single_id[idx] = it.second;
                rst.pop();
                idx--;
            }
            for (idx = rst_size; idx < k; idx++) {
                p_single_dis[idx] = float(1.0 / 0.0);
                p_single_id[idx] = -1;
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

void
IndexHNSW::QueryByRangeImpl(int64_t n, const float* xq, float*& distances, int64_t*& labels, size_t*& lims,
                            feder::hnsw::FederResultUniq& feder, const Config& config, const faiss::BitsetView bitset) {
    if (CheckKeyInConfig(config, meta::TRACE_VISIT) && GetMetaTraceVisit(config)) {
        KNOWHERE_THROW_IF_NOT_MSG(n == 1, "NQ must be 1 when Feder tracing");
        feder = std::make_unique<feder::hnsw::FederResult>();
    }

    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};

    float radius = GetMetaRadius(config);
    bool range_filter_exist = CheckKeyInConfig(config, meta::RANGE_FILTER);
    float range_filter = range_filter_exist ? GetMetaRangeFilter(config) : (1.0/0.0);
    bool is_ip = (index_->metric_type_ == 1);  // L2: 0, InnerProduct: 1

    std::vector<std::vector<int64_t>> result_id_array(n);
    std::vector<std::vector<float>> result_dist_array(n);
    std::vector<size_t> result_size(n);
    std::vector<size_t> result_lims(n + 1);

    std::vector<std::future<void>> futures;
    futures.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
        futures.push_back(pool_->push([&, index = i]() {
            auto single_query = xq + index * Dim();
            auto rst = index_->searchRange(single_query, radius, bitset, &param, feder);
            auto elem_cnt = rst.size();
            result_dist_array[index].resize(elem_cnt);
            result_id_array[index].resize(elem_cnt);
            for (size_t j = 0; j < elem_cnt; j++) {
                auto& p = rst[j];
                result_dist_array[index][j] = (is_ip ? (1 - p.first) : p.first);
                result_id_array[index][j] = p.second;
            }
            result_size[index] = rst.size();

            // filter range search result
            if (range_filter_exist) {
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                range_filter);
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, n, radius, range_filter, distances, labels, lims);
}

void
IndexHNSW::UpdateLevelLinkList(int32_t level, feder::hnsw::HNSWMeta& meta, std::unordered_set<int64_t>& id_set) {
    KNOWHERE_THROW_IF_NOT_FMT((level > 0 && level <= index_->maxlevel_), "Illegal level %d", level);
    if (index_->cur_element_count == 0) {
        return;
    }

    std::vector<hnswlib::tableint> level_elements;

    // get all elements in current level
    for (size_t i = 0; i < index_->cur_element_count; i++) {
        // elements in high level also exist in low level
        if (index_->element_levels_[i] >= level) {
            level_elements.emplace_back(i);
        }
    }

    // iterate all elements in current level, record their link lists
    for (auto curr_id : level_elements) {
        auto data = index_->get_linklist(curr_id, level);
        auto size = index_->getListCount(data);

        hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
        std::vector<int64_t> neighbors(size);
        for (int i = 0; i < size; i++) {
            hnswlib::tableint cand = datal[i];
            neighbors[i] = cand;
        }
        id_set.insert(curr_id);
        id_set.insert(neighbors.begin(), neighbors.end());
        meta.AddNodeInfo(level, curr_id, std::move(neighbors));
    }
}

}  // namespace knowhere
