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
#include "index/vector_index/IndexHNSW.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"

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
        Disassemble(res_set, config);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::Load(const BinarySet& index_binary) {
    try {
        Assemble(const_cast<BinarySet&>(index_binary));
        auto binary = index_binary.GetByName("HNSW");

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        hnswlib::SpaceInterface<float>* space = nullptr;
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(space);
        index_->stats_enable_ = (STATISTICS_LEVEL >= 3);
        index_->loadIndex(reader);
#if 0
        auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
        if (STATISTICS_LEVEL >= 3) {
            auto lock = hnsw_stats->Lock();
            hnsw_stats->update_level_distribution(index_->maxlevel_, index_->level_stats_);
        }
#endif
        // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Load finished, show statistics:";
        // LOG_KNOWHERE_DEBUG_ << hnsw_stats->ToString();
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
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, rows, GetIndexParamHNSWM(config),
                                                                   GetIndexParamEfConstruction(config));
        index_->stats_enable_ = (STATISTICS_LEVEL >= 3);
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
#if 0
    if (STATISTICS_LEVEL >= 3) {
        auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
        auto lock = hnsw_stats->Lock();
        hnsw_stats->update_level_distribution(index_->maxlevel_, index_->level_stats_);
    }
#endif
    // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Train finished, show statistics:";
    // LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();
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

    utils::SetQueryOmpThread(config);
    auto k = GetMetaTopk(config);
    auto p_id = new int64_t[k * rows];
    auto p_dist = new float[k * rows];
    std::vector<hnswlib::StatisticsInfo> query_stats;
    auto hnsw_stats = std::dynamic_pointer_cast<LibHNSWStatistics>(stats);
    if (STATISTICS_LEVEL >= 3) {
        query_stats.resize(rows);
        for (auto i = 0; i < rows; ++i) {
            query_stats[i].target_level_ = hnsw_stats->target_level;
        }
    }

    feder::hnsw::FederResultUniq feder_result;
    if (CheckKeyInConfig(config, meta::TRACE_VISIT) && GetMetaTraceVisit(config)) {
        KNOWHERE_THROW_IF_NOT_MSG(rows == 1, "NQ must be 1 when Feder tracing");
        feder_result = std::make_unique<feder::hnsw::FederResult>();
    }

    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};
    bool transform = (index_->metric_type_ == 1);  // InnerProduct: 1

    std::chrono::high_resolution_clock::time_point query_start, query_end;
    query_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (unsigned int i = 0; i < rows; ++i) {
        auto single_query = (float*)p_data + i * dim;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> rst;
        if (STATISTICS_LEVEL >= 3) {
            rst = index_->searchKnn(single_query, k, bitset, query_stats[i], &param, feder_result);
        } else {
            auto dummy_stat = hnswlib::StatisticsInfo();
            rst = index_->searchKnn(single_query, k, bitset, dummy_stat, &param, feder_result);
        }
        size_t rst_size = rst.size();

        auto p_single_dis = p_dist + i * k;
        auto p_single_id = p_id + i * k;
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
    }
    query_end = std::chrono::high_resolution_clock::now();

#if 0
    if (STATISTICS_LEVEL) {
        auto lock = hnsw_stats->Lock();
        if (STATISTICS_LEVEL >= 1) {
            hnsw_stats->update_nq(rows);
            hnsw_stats->update_ef_sum(index_->ef_ * rows);
            hnsw_stats->update_total_query_time(
                std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count());
        }
        if (STATISTICS_LEVEL >= 2) {
            hnsw_stats->update_filter_percentage(bitset);
        }
        if (STATISTICS_LEVEL >= 3) {
            for (auto i = 0; i < rows; ++i) {
                for (auto j = 0; j < query_stats[i].accessed_points.size(); ++j) {
                    auto tgt = hnsw_stats->access_cnt_map.find(query_stats[i].accessed_points[j]);
                    if (tgt == hnsw_stats->access_cnt_map.end())
                        hnsw_stats->access_cnt_map[query_stats[i].accessed_points[j]] = 1;
                    else
                        tgt->second += 1;
                }
            }
        }
    }
#endif
    // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Query finished, show statistics:";
    // LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();

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
IndexHNSW::QueryByRange(const DatasetPtr& dataset,
                        const Config& config,
                        const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA_DIM(dataset)

    utils::SetQueryOmpThread(config);
    auto range_k = GetIndexParamHNSWK(config);
    auto radius = GetMetaRadius(config);
    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};
    bool is_IP = (index_->metric_type_ == 1);  // InnerProduct: 1

    if (!is_IP) {
        radius *= radius;
    }

    std::vector<std::vector<int64_t>> result_id_array(rows);
    std::vector<std::vector<float>> result_dist_array(rows);
    std::vector<size_t> result_lims(rows + 1, 0);

//#pragma omp parallel for
    for (unsigned int i = 0; i < rows; ++i) {
        auto single_query = (float*)p_data + i * dim;

        auto dummy_stat = hnswlib::StatisticsInfo();
        auto rst =
            index_->searchRange(single_query, range_k, (is_IP ? 1.0f - radius : radius), bitset, dummy_stat, &param);

        for (auto& p : rst) {
            result_dist_array[i].push_back(is_IP ? (1 - p.first) : p.first);
            result_id_array[i].push_back(p.second);
        }
        result_lims[i+1] = result_lims[i] + rst.size();
    }

    LOG_KNOWHERE_DEBUG_ << "Range search radius: " << radius << ", result num: " << result_lims.back();

    auto p_id = new int64_t[result_lims.back()];
    auto p_dist = new float[result_lims.back()];
    auto p_lims = new size_t[rows + 1];

    for (int64_t i = 0; i < rows; i++) {
        size_t start = result_lims[i];
        size_t size = result_lims[i+1] - result_lims[i];
        memcpy(p_id + start, result_id_array[i].data(), size * sizeof(int64_t));
        memcpy(p_dist + start, result_dist_array[i].data(), size * sizeof(float));
    }
    memcpy(p_lims, result_lims.data(), (rows + 1) * sizeof(size_t));

    return GenResultDataset(p_id, p_dist, p_lims);
}

DatasetPtr
IndexHNSW::GetIndexMeta(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialized or trained");
    }

    const int64_t default_index_overview_levels = 3;
    auto overview_levels =
        CheckKeyInConfig(config, indexparam::OVERVIEW_LEVELS) ? GetIndexParamOverviewLevels(config)
                                                              : default_index_overview_levels;
    feder::hnsw::HNSWMeta meta(index_->ef_construction_, index_->M_, index_->cur_element_count, index_->maxlevel_,
                               index_->enterpoint_node_, overview_levels);
    std::unordered_set<int64_t> id_set;

    for (int i = 0; i < overview_levels; i++) {
        int64_t level = index_->maxlevel_ - i;
        // do not record level 0
        if (level <= 0) break;
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

#if 0
void
IndexHNSW::ClearStatistics() {
    if (!STATISTICS_LEVEL)
        return;
    auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
    auto lock = hnsw_stats->Lock();
    hnsw_stats->clear();
}
#endif

void
IndexHNSW::UpdateLevelLinkList(int32_t level, feder::hnsw::HNSWMeta& meta, std::unordered_set<int64_t>& id_set) {
    KNOWHERE_THROW_IF_NOT_FMT((level > 0 && level <= index_->maxlevel_), "Illegal level %d", level);
    if (index_->cur_element_count == 0) {
        return;
    }

    hnswlib::tableint enter_point = index_->enterpoint_node_;

    std::unordered_set<hnswlib::tableint> visited;
    std::queue<hnswlib::tableint> q;
    q.emplace(enter_point);

    while (!q.empty()) {
        auto curr_id = q.front();
        q.pop();
        visited.insert(curr_id);

        auto data = index_->get_linklist(curr_id, level);
        auto size = index_->getListCount(data);

        hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
        std::vector<int64_t> neighbors(size);
        for (int i = 0; i < size; i++) {
            hnswlib::tableint cand = datal[i];
            neighbors[i] = cand;
            if (visited.find(cand) == visited.end()) {
                q.emplace(cand);
            }
        }
        id_set.insert(curr_id);
        id_set.insert(neighbors.begin(), neighbors.end());
        meta.AddNodeInfo(level, curr_id, std::move(neighbors));
    }
}

#if (TEST_MODE == 1)
    void IndexHNSW::AsyncQuery(const DatasetPtr& dataset, const Config& config, const faiss::BitsetView bitset) {
        std::vector<std::future<DatasetPtr>>& stk = GetFeatureStack();
        stk.emplace_back(std::async(std::launch::async, [this, &dataset, &config, bitset]() {
            std::cout << std::this_thread::get_id() << std::endl;
            return this->Query(dataset, config, bitset);
        }));
    }

    DatasetPtr IndexHNSW::Sync() {
        std::vector<std::future<DatasetPtr>>& stk = GetFeatureStack();
        assert(!stk.empty());
        auto res = stk.back().get();
        stk.pop_back();
        return res;
    }
#endif

}  // namespace knowhere
