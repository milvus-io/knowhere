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

#include <string>

#include "common/Exception.h"
#include "common/Log.h"
#include "index/VecIndexFactory.h"
#include "index/vector_index/IndexAnnoy.h"
#include "index/vector_index/IndexBinaryIDMAP.h"
#include "index/vector_index/IndexBinaryIVF.h"
#include "index/vector_index/IndexHNSW.h"
#include "index/vector_index/IndexIDMAP.h"
#include "index/vector_index/IndexIVFPQ.h"
#include "index/vector_index/IndexIVFSQ.h"
#include "index/vector_index/IndexIVFHNSW.h"
#include "index/vector_index/IndexRHNSWFlat.h"
#include "index/vector_index/IndexRHNSWPQ.h"
#include "index/vector_index/IndexRHNSWSQ.h"
#include "index/vector_offset_index/IndexIVF_NM.h"

#ifdef KNOWHERE_SUPPORT_SPTAG
#include "knowhere/index/vector_index/IndexSPTAG.h"
#endif

#ifdef KNOWHERE_SUPPORT_NGT
#include "IndexNGTONNG.h"
#include "IndexNGTPANNG.h"
#endif

#ifdef KNOWHERE_SUPPORT_NSG
#include "index/vector_offset_index/IndexNSG_NM.h"
#endif

#ifdef KNOWHERE_GPU_VERSION
#include <cuda.h>
#include "knowhere/index/vector_index/gpu/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/gpu/IndexIVFSQHybrid.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.h"
#endif

namespace knowhere {

VecIndexPtr
VecIndexFactory::CreateVecIndex(const IndexType& type, const IndexMode mode) {
    switch (mode) {
        case IndexMode::MODE_CPU: {
            if (type == IndexEnum::INDEX_FAISS_BIN_IDMAP) {
                return std::make_shared<knowhere::BinaryIDMAP>();
            } else if (type == IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
                return std::make_shared<knowhere::BinaryIVF>();
            } else if (type == IndexEnum::INDEX_FAISS_IDMAP) {
                return std::make_shared<knowhere::IDMAP>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFFLAT) {
                return std::make_shared<knowhere::IVF_NM>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFPQ) {
                return std::make_shared<knowhere::IVFPQ>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFSQ8) {
                return std::make_shared<knowhere::IVFSQ>();
            } else if (type == IndexEnum::INDEX_FAISS_IVFHNSW) {
                return std::make_shared<knowhere::IVFHNSW>();
            } else if (type == IndexEnum::INDEX_ANNOY) {
                return std::make_shared<knowhere::IndexAnnoy>();
            } else if (type == IndexEnum::INDEX_HNSW) {
                return std::make_shared<knowhere::IndexHNSW>();
            } else if (type == IndexEnum::INDEX_RHNSWFlat) {
                return std::make_shared<knowhere::IndexRHNSWFlat>();
            } else if (type == IndexEnum::INDEX_RHNSWPQ) {
                return std::make_shared<knowhere::IndexRHNSWPQ>();
            } else if (type == IndexEnum::INDEX_RHNSWSQ) {
                return std::make_shared<knowhere::IndexRHNSWSQ>();
#ifdef KNOWHERE_SUPPORT_NGT
            } else if (type == IndexEnum::INDEX_NGTPANNG) {
                return std::make_shared<knowhere::IndexNGTPANNG>();
            } else if (type == IndexEnum::INDEX_NGTONNG) {
                return std::make_shared<knowhere::IndexNGTONNG>();
#endif
#ifdef KNOWHERE_SUPPORT_NSG
            } else if (type == IndexEnum::INDEX_NSG) {
                return std::make_shared<knowhere::NSG_NM>(-1);
#endif
#ifdef KNOWHERE_SUPPORT_SPTAG
            } else if (type == IndexEnum::INDEX_SPTAG_KDT_RNT) {
                return std::make_shared<knowhere::CPUSPTAGRNG>("KDT");
            } else if (type == IndexEnum::INDEX_SPTAG_BKT_RNT) {
                return std::make_shared<knowhere::CPUSPTAGRNG>("BKT");
#endif
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
