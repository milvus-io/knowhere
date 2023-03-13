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

#include "index/VecIndexFactory.h"

#include <string>

#include "common/Exception.h"
#include "common/Log.h"
#include "index/VecIndexThreadPoolWrapper.h"
#include "index/vector_index/IndexAnnoy.h"
#include "index/vector_index/IndexBinaryIDMAP.h"
#include "index/vector_index/IndexBinaryIVF.h"
#include "index/vector_index/IndexHNSW.h"
#include "index/vector_index/IndexIDMAP.h"
#include "index/vector_index/IndexIVFPQ.h"
#include "index/vector_index/IndexIVFSQ.h"
#include "index/vector_offset_index/IndexIVF_NM.h"

#ifdef KNOWHERE_GPU_VERSION
#include <cuda.h>

#include "knowhere/index/vector_index/gpu/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.h"
#endif

namespace knowhere {

VecIndexPtr
VecIndexFactory::CreateVecIndex(const IndexType& type, const IndexMode mode) {
    switch (mode) {
        case IndexMode::MODE_CPU: {
            if (type == IndexEnum::INDEX_FAISS_BIN_IDMAP) {
                return std::make_shared<VecIndexThreadPoolWrapper>(std::make_unique<BinaryIDMAP>());
            } else if (type == IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
                return std::make_shared<VecIndexThreadPoolWrapper>(std::make_unique<BinaryIVF>());
            } else if (type == IndexEnum::INDEX_FAISS_IDMAP) {
                return std::make_shared<knowhere::IDMAP>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFFLAT) {
                return std::make_shared<knowhere::IVF_NM>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFPQ) {
                return std::make_shared<knowhere::IVFPQ>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFSQ8) {
                return std::make_shared<knowhere::IVFSQ>();
            } else if (type == IndexEnum::INDEX_ANNOY) {
                return std::make_shared<knowhere::IndexAnnoy>();
            } else if (type == IndexEnum::INDEX_HNSW) {
                return std::make_shared<knowhere::IndexHNSW>();
            } else {
                KNOWHERE_THROW_FORMAT("Invalid index type %s", std::string(type).c_str());
            }
        }
#ifdef KNOWHERE_GPU_VERSION
        case IndexMode::MODE_GPU: {
            auto gpu_device = 0;  // TODO: remove hardcode here, get from invoker
            if (type == IndexEnum::INDEX_FAISS_BIN_IDMAP) {
                return std::make_shared<knowhere::BinaryIDMAP>();
            } else if (type == IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
                return std::make_shared<knowhere::BinaryIVF>();
            } else if (type == IndexEnum::INDEX_FAISS_IDMAP) {
                return std::make_shared<knowhere::GPUIDMAP>(gpu_device);
            } else if (type == IndexEnum::INDEX_FAISS_IVFFLAT) {
                return std::make_shared<knowhere::GPUIVF>(gpu_device);
            } else if (type == IndexEnum::INDEX_FAISS_IVFPQ) {
                return std::make_shared<knowhere::GPUIVFPQ>(gpu_device);
            } else if (type == IndexEnum::INDEX_FAISS_IVFSQ8) {
                return std::make_shared<knowhere::GPUIVFSQ>(gpu_device);
            } else {
                KNOWHERE_THROW_FORMAT("Invalid index type %s", std::string(type).c_str());
            }
        }
#endif
        default:
            KNOWHERE_THROW_FORMAT("Invalid index mode %d", (int)mode);
    }
}

}  // namespace knowhere
