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

#pragma once

#include <string>

namespace knowhere {

using IndexType = std::string;

namespace IndexEnum {

constexpr const char* INVALID = "";

constexpr const char* INDEX_FAISS_BIN_IDMAP = "BIN_FLAT";
constexpr const char* INDEX_FAISS_BIN_IVFFLAT = "BIN_IVF_FLAT";

constexpr const char* INDEX_FAISS_IDMAP = "FLAT";
constexpr const char* INDEX_FAISS_IVFFLAT = "IVF_FLAT";
constexpr const char* INDEX_FAISS_IVFPQ = "IVF_PQ";
constexpr const char* INDEX_FAISS_IVFSQ8 = "IVF_SQ8";
constexpr const char* INDEX_FAISS_IVFSQ8H = "IVF_SQ8_HYBRID";
constexpr const char* INDEX_FAISS_IVFHNSW = "IVF_HNSW";

constexpr const char* INDEX_ANNOY = "ANNOY";
constexpr const char* INDEX_HNSW = "HNSW";
constexpr const char* INDEX_RHNSWFlat = "RHNSW_FLAT";
constexpr const char* INDEX_RHNSWPQ = "RHNSW_PQ";
constexpr const char* INDEX_RHNSWSQ = "RHNSW_SQ";

constexpr const char* INDEX_DISKANN = "DISKANN";

#ifdef KNOWHERE_SUPPORT_NGT
constexpr const char* INDEX_NGTPANNG = "NGT_PANNG";
constexpr const char* INDEX_NGTONNG = "NGT_ONNG";
#endif

#ifdef KNOWHERE_SUPPORT_NSG
constexpr const char* INDEX_NSG = "NSG";
#endif

#ifdef KNOWHERE_SUPPORT_SPTAG
constexpr const char* INDEX_SPTAG_KDT_RNT = "SPTAG_KDT_RNT";
constexpr const char* INDEX_SPTAG_BKT_RNT = "SPTAG_BKT_RNT";
#endif

}  // namespace IndexEnum

enum class IndexMode { MODE_CPU = 0, MODE_GPU = 1 };

}  // namespace knowhere
