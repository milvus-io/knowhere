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

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>

#include <chrono>
#include <string>

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "index/vector_index/IndexBinaryIVF.h"
#include "index/vector_index/adapter/VectorAdapter.h"

namespace knowhere {

using stdclock = std::chrono::high_resolution_clock;

BinarySet
BinaryIVF::Serialize(const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto ret = SerializeImpl(index_type_);
    return ret;
}

void
BinaryIVF::Load(const BinarySet& index_binary) {
    LoadImpl(index_binary, index_type_);
#if 0
    if (STATISTICS_LEVEL >= 3) {
        auto ivf_index = static_cast<faiss::IndexBinaryIVF*>(index_.get());
        ivf_index->nprobe_statistics.resize(ivf_index->nlist, 0);
    }
#endif
}

DatasetPtr
BinaryIVF::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
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
        auto bin_ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
        bin_ivf_index->make_direct_map(true);
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < bin_ivf_index->ntotal, "invalid id %ld", id);
            bin_ivf_index->reconstruct(id, p_x + i * dim / 8);
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
BinaryIVF::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
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
BinaryIVF::QueryByRange(const DatasetPtr& dataset,
                        const Config& config,
                        const faiss::BitsetView bitset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA(dataset)

    utils::SetQueryOmpThread(config);
    auto radius = GetMetaRadius(config);

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
        QueryByRangeImpl(rows, reinterpret_cast<const uint8_t*>(p_data), radius, p_dist, p_id, p_lims, config, bitset);
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
BinaryIVF::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
BinaryIVF::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

int64_t
BinaryIVF::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto bin_ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    auto nb = bin_ivf_index->invlists->compute_ntotal();
    auto nlist = bin_ivf_index->nlist;
    auto code_size = bin_ivf_index->code_size;

    // binary ivf codes, ids and quantizer
    return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
}

#if 0
StatisticsPtr
BinaryIVF::GetStatistics() {
    if (!STATISTICS_LEVEL) {
        return stats;
    }
    auto ivf_stats = std::dynamic_pointer_cast<IVFStatistics>(stats);
    auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    auto lock = ivf_stats->Lock();
    ivf_stats->update_ivf_access_stats(ivf_index->nprobe_statistics);
    return ivf_stats;
}

void
BinaryIVF::ClearStatistics() {
    if (!STATISTICS_LEVEL) {
        return;
    }
    auto ivf_stats = std::dynamic_pointer_cast<IVFStatistics>(stats);
    auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    ivf_index->clear_nprobe_statistics();
    ivf_index->index_ivf_stats.reset();
    auto lock = ivf_stats->Lock();
    ivf_stats->clear();
}
#endif

void
BinaryIVF::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetBuildOmpThread(config);
    int64_t nlist = GetIndexParamNlist(config);
    faiss::MetricType metric_type = GetFaissMetricType(config);
    faiss::IndexBinary* coarse_quantizer = new faiss::IndexBinaryFlat(dim, metric_type);
    auto index = std::make_shared<faiss::IndexBinaryIVF>(coarse_quantizer, dim, nlist, metric_type);
    index->own_fields = true;
    index->train(rows, static_cast<const uint8_t*>(p_data));
    index_ = index;
}

void
BinaryIVF::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const uint8_t*>(p_data));
}

std::shared_ptr<faiss::IVFSearchParameters>
BinaryIVF::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = GetIndexParamNprobe(config);
    // params->max_codes = config["max_code"];
    return params;
}

void
BinaryIVF::QueryImpl(int64_t n,
                     const uint8_t* data,
                     int64_t k,
                     float* distances,
                     int64_t* labels,
                     const Config& config,
                     const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    ivf_index->nprobe = params->nprobe;

    stdclock::time_point before = stdclock::now();
    auto i_distances = reinterpret_cast<int32_t*>(distances);

    index_->search(n, data, k, i_distances, labels, bitset);

#if 0
    stdclock::time_point after = stdclock::now();
    double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
    LOG_KNOWHERE_DEBUG_ << "IVF_NM search cost: " << search_cost
                        << ", quantization cost: " << ivf_index->index_ivf_stats.quantization_time
                        << ", data search cost: " << ivf_index->index_ivf_stats.search_time;

    if (STATISTICS_LEVEL) {
        auto ivf_stats = std::dynamic_pointer_cast<IVFStatistics>(stats);
        auto lock = ivf_stats->Lock();
        if (STATISTICS_LEVEL >= 1) {
            ivf_stats->update_nq(n);
            ivf_stats->count_nprobe(ivf_index->nprobe);
            ivf_stats->update_total_query_time(ivf_index->index_ivf_stats.quantization_time +
                                               ivf_index->index_ivf_stats.search_time);
            ivf_index->index_ivf_stats.quantization_time = 0;
            ivf_index->index_ivf_stats.search_time = 0;
        }
        if (STATISTICS_LEVEL >= 2) {
            ivf_stats->update_filter_percentage(bitset);
        }
    }
#endif

    // convert int32 to float for hamming
    if (ivf_index->metric_type == faiss::METRIC_Hamming) {
        int64_t num = n * k;
        for (int64_t i = 0; i < num; i++) {
            distances[i] = static_cast<float>(i_distances[i]);
        }
    }
}

void
BinaryIVF::QueryByRangeImpl(int64_t n,
                            const uint8_t* data,
                            float radius,
                            float*& distances,
                            int64_t*& labels,
                            size_t*& lims,
                            const Config& config,
                            const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index_.get());
    ivf_index->nprobe = params->nprobe;

    faiss::RangeSearchResult res(n);
    index_->range_search(n, data, radius, &res, bitset);

    distances = res.distances;
    labels = res.labels;
    lims = res.lims;

    LOG_KNOWHERE_DEBUG_ << "Range search radius: " << radius << ", result num: " << lims[n];

    res.distances = nullptr;
    res.labels = nullptr;
    res.lims = nullptr;
}

}  // namespace knowhere
