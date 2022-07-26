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
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/clone_index.h>
#include <faiss/index_io.h>
#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuCloner.h>
#endif

#include <string>
#include <vector>

#include "common/Exception.h"
#include "index/vector_index/IndexIDMAP.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"
#include "index/vector_index/helpers/IndexParameter.h"
#ifdef KNOWHERE_GPU_VERSION
#include "index/vector_index/gpu/IndexGPUIDMAP.h"
#include "index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace knowhere {

BinarySet
IDMAP::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    auto ret = SerializeImpl(index_type_);
    Disassemble(ret, config);
    return ret;
}

void
IDMAP::Load(const BinarySet& binary_set) {
    Assemble(const_cast<BinarySet&>(binary_set));
    LoadImpl(binary_set, index_type_);
}

void
IDMAP::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    faiss::MetricType metric_type = GetFaissMetricType(config);
    auto index = std::make_shared<faiss::IndexFlat>(dim, metric_type);
    index_ = index;
}

void
IDMAP::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const float*>(p_data));
}

DatasetPtr
IDMAP::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    float* p_x = nullptr;
    auto release_when_exception = [&]() {
        if (p_x != nullptr) {
            free(p_x);
        }
    };

    try {
        p_x = (float*)malloc(sizeof(float) * dim * rows);
        auto idmap_index = dynamic_cast<faiss::IndexFlat*>(index_.get());
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < idmap_index->ntotal, "invalid id %ld", id);
            idmap_index->reconstruct(id, p_x + i * dim);
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
IDMAP::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
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
        auto k = GetMetaTopk(config);
        auto elems = rows * k;
        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        p_id = static_cast<int64_t*>(malloc(p_id_size));
        p_dist = static_cast<float*>(malloc(p_dist_size));

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
IDMAP::QueryByRange(const DatasetPtr& dataset,
                    const Config& config,
                    const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GET_TENSOR_DATA(dataset)

    auto radius = GetMetaRadius(config);

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
        QueryByRangeImpl(rows, reinterpret_cast<const float*>(p_data), radius, p_dist, p_id, p_lims, config, bitset);
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
IDMAP::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
IDMAP::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

VecIndexPtr
IDMAP::CopyCpuToGpu(const int64_t device_id, const Config& config) {
#ifdef KNOWHERE_GPU_VERSION
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
        ResScope rs(res, device_id, false);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), device_id, index_.get());

        std::shared_ptr<faiss::Index> device_index;
        device_index.reset(gpu_index);
        return std::make_shared<GPUIDMAP>(device_index, device_id, res);
    } else {
        KNOWHERE_THROW_MSG("CopyCpuToGpu Error, can't get gpu_resource");
    }
#else
    KNOWHERE_THROW_MSG("Calling IDMAP::CopyCpuToGpu when we are using CPU version");
#endif
}

const float*
IDMAP::GetRawVectors() {
    try {
        auto flat_index = dynamic_cast<faiss::IndexFlat*>(index_.get());
        return reinterpret_cast<const float*>(flat_index->codes.data());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IDMAP::QueryImpl(int64_t n,
                 const float* data,
                 int64_t k,
                 float* distances,
                 int64_t* labels,
                 const Config& config,
                 const faiss::BitsetView bitset) {
    index_->search(n, data, k, distances, labels, bitset);
}

void
IDMAP::QueryByRangeImpl(int64_t n,
                        const float* data,
                        float radius,
                        float*& distances,
                        int64_t*& labels,
                        size_t*& lims,
                        const Config& config,
                        const faiss::BitsetView bitset) {
    auto idmap_index = dynamic_cast<faiss::IndexFlat*>(index_.get());

    if (index_->metric_type == faiss::MetricType::METRIC_L2) {
        radius *= radius;
    }

    faiss::RangeSearchResult res(n);
    idmap_index->range_search(n, reinterpret_cast<const float*>(data), radius, &res, bitset);

    distances = res.distances;
    labels = res.labels;
    lims = res.lims;

    LOG_KNOWHERE_DEBUG_ << "Range search radius: " << radius << ", result num: " << lims[n];

    res.distances = nullptr;
    res.labels = nullptr;
    res.lims = nullptr;
}

}  // namespace knowhere
