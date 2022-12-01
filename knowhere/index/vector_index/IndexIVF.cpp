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

#include <faiss/AutoTune.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>
#include <faiss/index_io.h>
#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#endif

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/Exception.h"
#include "common/Log.h"
#include "common/Utils.h"
#include "index/vector_index/IndexIVF.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/IndexParameter.h"
#include "index/vector_index/helpers/RangeUtil.h"
#ifdef KNOWHERE_GPU_VERSION
#include "index/vector_index/gpu/IndexGPUIVF.h"
#include "index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace knowhere {

using stdclock = std::chrono::high_resolution_clock;

BinarySet
IVF::Serialize(const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto ret = SerializeImpl(index_type_);
    return ret;
}

void
IVF::Load(const BinarySet& binary_set) {
    LoadImpl(binary_set, index_type_);

#if 0
    if (IndexMode() == IndexMode::MODE_CPU && STATISTICS_LEVEL >= 3) {
        auto ivf_index = static_cast<faiss::IndexIVFFlat*>(index_.get());
        ivf_index->nprobe_statistics.resize(ivf_index->nlist, 0);
    }
#endif
}

void
IVF::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    utils::SetBuildOmpThread(config);
    auto nlist = GetIndexParamNlist(config);
    faiss::MetricType metric_type = GetFaissMetricType(config);
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim, metric_type);
    auto index = std::make_shared<faiss::IndexIVFFlat>(coarse_quantizer, dim, nlist, metric_type);
    index->own_fields = true;
    index->train(rows, reinterpret_cast<const float*>(p_data));
    index_ = index;
}

void
IVF::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const float*>(p_data));
}

DatasetPtr
IVF::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    float* p_x = nullptr;
    auto release_when_exception = [&]() {
        if (p_x != nullptr) {
            delete[] p_x;
        }
    };

    try {
        p_x = new float[dim * rows];
        auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
        ivf_index->make_direct_map(true);
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < ivf_index->ntotal, "invalid id %ld", id);
            ivf_index->reconstruct(id, p_x + i * dim);
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
IVF::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
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

        QueryImpl(rows, reinterpret_cast<const float*>(p_data), k, p_dist, p_id, config, bitset);

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
IVF::QueryByRange(const DatasetPtr& dataset,
                  const Config& config,
                  const faiss::BitsetView bitset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
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
        QueryByRangeImpl(rows, reinterpret_cast<const float*>(p_data), p_dist, p_id, p_lims, config, bitset);
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
IVF::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
IVF::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

void
IVF::Seal() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    SealImpl();
}

int64_t
IVF::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto ivf_index = static_cast<faiss::IndexIVFFlat*>(index_.get());
    auto nb = ivf_index->invlists->compute_ntotal();
    auto nlist = ivf_index->nlist;
    auto code_size = ivf_index->code_size;
    // ivf codes, ivf ids and quantizer
    return (nb * code_size + nb * sizeof(int64_t) + nlist * code_size);
}

VecIndexPtr
IVF::CopyCpuToGpu(const int64_t device_id, const Config& config) {
#ifdef KNOWHERE_GPU_VERSION
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
        ResScope rs(res, device_id, false);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), device_id, index_.get());

        std::shared_ptr<faiss::Index> device_index;
        device_index.reset(gpu_index);
        return std::make_shared<GPUIVF>(device_index, device_id, res);
    } else {
        KNOWHERE_THROW_MSG("CopyCpuToGpu Error, can't get gpu_resource");
    }

#else
    KNOWHERE_THROW_MSG("Calling IVF::CopyCpuToGpu when we are using CPU version");
#endif
}

void
IVF::GenGraph(const float* data, const int64_t k, GraphType& graph, const Config& config) {
    int64_t K = k + 1;
    auto ntotal = Count();

    auto dim = GetMetaDim(config);
    auto batch_size = 1000;
    auto tail_batch_size = ntotal % batch_size;
    auto batch_search_count = ntotal / batch_size;
    auto total_search_count = tail_batch_size == 0 ? batch_search_count : batch_search_count + 1;

    std::vector<float> res_dis(K * batch_size);
    graph.resize(ntotal);
    GraphType res_vec(total_search_count);
    for (int i = 0; i < total_search_count; ++i) {
        auto b_size = (i == (total_search_count - 1)) && tail_batch_size != 0 ? tail_batch_size : batch_size;

        auto& res = res_vec[i];
        res.resize(K * b_size);

        const float* xq = data + batch_size * dim * i;
        QueryImpl(b_size, xq, K, res_dis.data(), res.data(), config, nullptr);

        for (int j = 0; j < b_size; ++j) {
            auto& node = graph[batch_size * i + j];
            node.resize(k);
            auto start_pos = j * K + 1;
            for (int m = 0, cursor = start_pos; m < k && cursor < start_pos + k; ++m, ++cursor) {
                node[m] = res[cursor];
            }
        }
    }
}

std::shared_ptr<faiss::IVFSearchParameters>
IVF::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = GetIndexParamNprobe(config);
    // params->max_codes = config["max_codes"];
    return params;
}

void
IVF::QueryImpl(int64_t n,
               const float* xq,
               int64_t k,
               float* distances,
               int64_t* labels,
               const Config& config,
               const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());

    stdclock::time_point before = stdclock::now();
    int parallel_mode = -1;
    if (params->nprobe > 1 && n <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;
    auto ivf_stats = std::dynamic_pointer_cast<IVFStatistics>(stats);
    ivf_index->search_thread_safe(n, xq, k, distances, labels, params->nprobe, parallel_mode, max_codes, bitset);
#if 0
    stdclock::time_point after = stdclock::now();
    double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
    if (STATISTICS_LEVEL) {
        auto lock = ivf_stats->Lock();
        if (STATISTICS_LEVEL >= 1) {
            ivf_stats->update_nq(n);
            ivf_stats->count_nprobe(ivf_index->nprobe);

            LOG_KNOWHERE_DEBUG_ << "IVF search cost: " << search_cost
                                << ", quantization cost: " << ivf_index->index_ivf_stats.quantization_time
                                << ", data search cost: " << ivf_index->index_ivf_stats.search_time;
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
    //     LOG_KNOWHERE_DEBUG_ << "IndexIVF::QueryImpl finished, show statistics:";
    //     LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();
}

void
IVF::QueryByRangeImpl(int64_t n,
                      const float* xq,
                      float*& distances,
                      int64_t*& labels,
                      size_t*& lims,
                      const Config& config,
                      const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());

    int parallel_mode = -1;
    if (params->nprobe > 1 && n <= 4) {
        parallel_mode = 1;
    } else {
        parallel_mode = 0;
    }
    size_t max_codes = 0;

    float low_bound = GetMetaRadiusLowBound(config);
    float high_bound = GetMetaRadiusHighBound(config);
    bool is_ip = (ivf_index->metric_type == faiss::METRIC_INNER_PRODUCT);
    float radius = (is_ip ? low_bound : high_bound);

    faiss::RangeSearchResult res(n);
    ivf_index->range_search_thread_safe(n, xq, radius, &res, params->nprobe, parallel_mode, max_codes, bitset);
    GetRangeSearchResult(res, is_ip, n, low_bound, high_bound, distances, labels, lims, bitset);
}

void
IVF::SealImpl() {
#ifdef KNOWHERE_GPU_VERSION
    faiss::Index* index = index_.get();
    auto idx = dynamic_cast<faiss::IndexIVF*>(index);
    if (idx != nullptr) {
        idx->to_readonly();
    }
#endif
}

#if 0
StatisticsPtr
IVF::GetStatistics() {
    if (IndexMode() != IndexMode::MODE_CPU || !STATISTICS_LEVEL) {
        return stats;
    }
    auto ivf_stats = std::static_pointer_cast<IVFStatistics>(stats);
    auto ivf_index = static_cast<faiss::IndexIVF*>(index_.get());
    auto lock = ivf_stats->Lock();
    ivf_stats->update_ivf_access_stats(ivf_index->nprobe_statistics);
    return ivf_stats;
}

void
IVF::ClearStatistics() {
    if (IndexMode() != IndexMode::MODE_CPU || !STATISTICS_LEVEL) {
        return;
    }
    auto ivf_stats = std::static_pointer_cast<IVFStatistics>(stats);
    auto ivf_index = static_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->clear_nprobe_statistics();
    ivf_index->index_ivf_stats.reset();
    auto lock = ivf_stats->Lock();
    ivf_stats->clear();
}
#endif

}  // namespace knowhere
