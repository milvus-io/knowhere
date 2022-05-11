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
#include <string>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "index/vector_index/IndexRHNSW.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace knowhere {

BinarySet
IndexRHNSW::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        MemoryIOWriter writer;
        writer.name = std::string(this->index_type()) + "_Index";
        faiss::write_index(index_.get(), &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        res_set.Append(writer.name, data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSW::Load(const BinarySet& index_binary) {
    try {
        Assemble(const_cast<BinarySet&>(index_binary));
        MemoryIOReader reader;
        reader.name = std::string(this->index_type()) + "_Index";
        auto binary = index_binary.GetByName(reader.name);

        reader.total = static_cast<size_t>(binary->size);
        reader.data_ = binary->data.get();

        auto idx = faiss::read_index(&reader);
#if 0
        auto hnsw_stats = std::static_pointer_cast<RHNSWStatistics>(stats);
        if (STATISTICS_LEVEL >= 3) {
            auto real_idx = static_cast<faiss::IndexRHNSW*>(idx);
            auto lock = hnsw_stats->Lock();
            hnsw_stats->update_level_distribution(real_idx->hnsw.max_level, real_idx->hnsw.level_stats);
            real_idx->set_target_level(hnsw_stats->target_level);
            //             LOG_KNOWHERE_DEBUG_ << "IndexRHNSW::Load finished, show statistics:";
            //             LOG_KNOWHERE_DEBUG_ << hnsw_stats->ToString();
        }
#endif
        index_.reset(idx);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexRHNSW::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    KNOWHERE_THROW_MSG("IndexRHNSW has no implementation of Train, please use IndexRHNSW(Flat/SQ/PQ) instead!");
}

void
IndexRHNSW::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GET_TENSOR_DATA(dataset_ptr)

    index_->add(rows, reinterpret_cast<const float*>(p_data));
#if 0
    auto hnsw_stats = std::static_pointer_cast<RHNSWStatistics>(stats);
    if (STATISTICS_LEVEL >= 3) {
        auto real_idx = static_cast<faiss::IndexRHNSW*>(index_.get());
        auto lock = hnsw_stats->Lock();
        hnsw_stats->update_level_distribution(real_idx->hnsw.max_level, real_idx->hnsw.level_stats);
        real_idx->set_target_level(hnsw_stats->target_level);
    }
#endif
    //     LOG_KNOWHERE_DEBUG_ << "IndexRHNSW::Load finished, show statistics:";
    //     LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();
}

DatasetPtr
IndexRHNSW::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA(dataset_ptr)
    auto k = GetMetaTopk(config);
    auto result_count = rows * k;

    auto p_id = static_cast<int64_t*>(malloc(result_count * sizeof(int64_t)));
    auto p_dist = static_cast<float*>(malloc(result_count * sizeof(float)));
    for (int64_t i = 0; i < result_count; ++i) {
        p_id[i] = -1;
        p_dist[i] = -1;
    }

    auto real_index = dynamic_cast<faiss::IndexRHNSW*>(index_.get());

    real_index->hnsw.efSearch = GetIndexParamEf(config);

    std::chrono::high_resolution_clock::time_point query_start, query_end;
    query_start = std::chrono::high_resolution_clock::now();
    real_index->search(rows, reinterpret_cast<const float*>(p_data), k, p_dist, p_id, bitset);
    query_end = std::chrono::high_resolution_clock::now();
#if 0
    if (STATISTICS_LEVEL) {
        auto hnsw_stats = std::dynamic_pointer_cast<RHNSWStatistics>(stats);
        auto lock = hnsw_stats->Lock();
        if (STATISTICS_LEVEL >= 1) {
            hnsw_stats->update_nq(rows);
            hnsw_stats->update_ef_sum(real_index->hnsw.efSearch * rows);
            hnsw_stats->update_total_query_time(
                std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count());
        }
        if (STATISTICS_LEVEL >= 2) {
            hnsw_stats->update_filter_percentage(bitset);
        }
    }
#endif
    // LOG_KNOWHERE_DEBUG_ << "IndexRHNSW::Load finished, show statistics:";
    // LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();

    return GenResultDataset(p_id, p_dist);
}

int64_t
IndexRHNSW::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
IndexRHNSW::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

#if 0
StatisticsPtr
IndexRHNSW::GetStatistics() {
    if (!STATISTICS_LEVEL) {
        return stats;
    }
    auto hnsw_stats = std::static_pointer_cast<RHNSWStatistics>(stats);
    auto real_index = static_cast<faiss::IndexRHNSW*>(index_.get());
    auto lock = hnsw_stats->Lock();
    real_index->get_sorted_access_counts(hnsw_stats->access_cnt, hnsw_stats->access_total);
    return hnsw_stats;
}

void
IndexRHNSW::ClearStatistics() {
    if (!STATISTICS_LEVEL) {
        return;
    }
    auto hnsw_stats = std::static_pointer_cast<RHNSWStatistics>(stats);
    auto real_index = static_cast<faiss::IndexRHNSW*>(index_.get());
    real_index->clear_stats();
    auto lock = hnsw_stats->Lock();
    hnsw_stats->clear();
}
#endif

int64_t
IndexRHNSW::Size() {
    KNOWHERE_THROW_MSG("IndexRHNSW has no implementation of Size, please use IndexRHNSW(Flat/SQ/PQ) instead!");
}

/*
BinarySet
IndexRHNSW::SerializeImpl(const milvus::knowhere::IndexType &type) { return BinarySet(); }

void
IndexRHNSW::SealImpl() {}

void
IndexRHNSW::LoadImpl(const milvus::knowhere::BinarySet &, const milvus::knowhere::IndexType &type) {}
*/

}  // namespace knowhere
