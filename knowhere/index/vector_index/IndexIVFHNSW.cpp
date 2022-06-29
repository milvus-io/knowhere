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

#include <algorithm>
#include <memory>
#include <string>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/clone_index.h>
#include <faiss/index_io.h>

#include "faiss/IndexRHNSW.h"

#include "common/Exception.h"
#include "index/vector_index/IndexIVFHNSW.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"
#include "index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

BinarySet
IVFHNSW::Serialize(const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        // Serialize IVF index and HNSW data
        auto res_set = SerializeImpl(index_type_);
        auto index = dynamic_cast<faiss::IndexIVFFlat*>(index_.get());
        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index->quantizer);
        if (real_idx == nullptr) {
            KNOWHERE_THROW_MSG("Quantizer index is not a faiss::IndexRHNSWFlat");
        }

        MemoryIOWriter writer;
        faiss::write_index(real_idx->storage, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);
        res_set.Append("HNSW_STORAGE", data, writer.rp);

        Disassemble(res_set, config);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IVFHNSW::Load(const BinarySet& binary_set) {
    try {
        // Load IVF index and HNSW data
        Assemble(const_cast<BinarySet&>(binary_set));
        LoadImpl(binary_set, index_type_);

        auto index = dynamic_cast<faiss::IndexIVFFlat*>(index_.get());
        MemoryIOReader reader;
        auto binary = binary_set.GetByName("HNSW_STORAGE");
        reader.total = static_cast<size_t>(binary->size);
        reader.data_ = binary->data.get();

        auto real_idx = dynamic_cast<faiss::IndexRHNSWFlat*>(index->quantizer);
        real_idx->storage = faiss::read_index(&reader);
        real_idx->init_hnsw();
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IVFHNSW::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    faiss::MetricType metric_type = GetFaissMetricType(config);
    auto coarse_quantizer = new faiss::IndexRHNSWFlat(dim, GetIndexParamHNSWM(config), metric_type);
    coarse_quantizer->hnsw.efConstruction = GetIndexParamEfConstruction(config);
    auto index = std::make_shared<faiss::IndexIVFFlat>(coarse_quantizer, dim, GetIndexParamNlist(config), metric_type);
    index->own_fields = true;
    index->train(rows, reinterpret_cast<const float*>(p_data));
    index_ = index;
}

VecIndexPtr
IVFHNSW::CopyCpuToGpu(const int64_t device_id, const Config& config) {
    KNOWHERE_THROW_MSG("IVFHNSW::CopyCpuToGpu not supported.");
}

int64_t
IVFHNSW::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto ivf_index = static_cast<faiss::IndexIVFFlat*>(index_.get());
    auto nb = ivf_index->invlists->compute_ntotal();
    auto code_size = ivf_index->code_size;
    auto hnsw_quantizer = dynamic_cast<faiss::IndexRHNSWFlat*>(ivf_index->quantizer);
    // ivf codes, ivf ids and hnsw_flat quantizer
    return (nb * code_size + nb * sizeof(int64_t) + hnsw_quantizer->cal_size());
}

void
IVFHNSW::QueryImpl(int64_t n,
                   const float* data,
                   int64_t k,
                   float* distances,
                   int64_t* labels,
                   const Config& config,
                   const faiss::BitsetView bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(params->nprobe, ivf_index->invlists->nlist);
    if (params->nprobe > 1 && n <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }
    // Update HNSW quantizer search param
    auto hnsw_quantizer = dynamic_cast<faiss::IndexRHNSWFlat*>(ivf_index->quantizer);
    hnsw_quantizer->hnsw.efSearch = GetIndexParamEf(config);
    ivf_index->search(n, data, k, distances, labels, bitset);
}

}  // namespace knowhere
