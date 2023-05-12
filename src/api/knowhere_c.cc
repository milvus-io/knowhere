// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/knowhere_c.h"

#include <string>

#include "knowhere/binaryset.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/dataset.h"
#include "knowhere/enum.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "knowhere/index.h"
#include "knowhere/log.h"

std::once_flag init_knowhere_once_;

#define KnowhereAssertInfo(expr, info)     \
    do {                                   \
        auto _expr_res = bool(expr);       \
        /* call func only when needed */   \
        if (!_expr_res) {                  \
            LOG_KNOWHERE_WARNING_ << info; \
            return -1;                     \
        }                                  \
    } while (0)

int
knowhere_init(CKnowhereConfig* config) {
    auto init = [&]() {
        knowhere::KnowhereConfig::SetBlasThreshold(16384);
        knowhere::KnowhereConfig::SetEarlyStopThreshold(0);
        knowhere::KnowhereConfig::SetLogHandler();
        knowhere::KnowhereConfig::SetStatisticsLevel(0);
        knowhere::KnowhereConfig::ShowVersion();
    };

    auto value = config->simd_type;
    knowhere::KnowhereConfig::SimdType simd_type;
    if (strcmp(value, "auto") == 0) {
        simd_type = knowhere::KnowhereConfig::SimdType::AUTO;
    } else if (strcmp(value, "avx512") == 0) {
        simd_type = knowhere::KnowhereConfig::SimdType::AVX512;
    } else if (strcmp(value, "avx2") == 0) {
        simd_type = knowhere::KnowhereConfig::SimdType::AVX2;
    } else if (strcmp(value, "avx") == 0 || strcmp(value, "sse4_2") == 0) {
        simd_type = knowhere::KnowhereConfig::SimdType::SSE4_2;
    } else {
        simd_type = knowhere::KnowhereConfig::SimdType::GENERIC;
    }
    try {
        knowhere::KnowhereConfig::SetSimdType(simd_type);
    } catch (std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "simd_type is invalid, which is " << value;
    }
    knowhere::ThreadPool::InitGlobalThreadPool(config->thread_num);
    std::call_once(init_knowhere_once_, init);
    return 0;
}

int
knowhere_destroy_index(CIndexCtx* index) {
    KnowhereAssertInfo(index != nullptr, "index is nullptr");
    KnowhereAssertInfo(index->internal != nullptr, "index->internal is nullptr");
    delete static_cast<knowhere::Index<knowhere::IndexNode>*>(index->internal);
    return 0;
}

int
knowhere_destroy_binary_set(CBinarySet* binary) {
    KnowhereAssertInfo(binary != nullptr, "binary is nullptr");
    KnowhereAssertInfo(binary->internal != nullptr, "binary->internal is nullptr");
    delete static_cast<knowhere::BinarySet*>(binary->internal);
    return 0;
}

int
knowhere_destroy_search_result(CSearchResult* result) {
    KnowhereAssertInfo(result != nullptr, "result is nullptr");
    KnowhereAssertInfo(result->ids != nullptr, "result->ids is nullptr");
    KnowhereAssertInfo(result->distances != nullptr, "result->distances is nullptr");
    delete[] static_cast<const int64_t*>(result->ids);
    delete[] static_cast<const float*>(result->distances);
    return 0;
}

int
knowhere_build_index(const char* name, CBuildParams* build_params, CIndexCtx* index) {
    KnowhereAssertInfo(name != nullptr, "name is nullptr");
    KnowhereAssertInfo(build_params != nullptr, "build params is nullptr");
    auto idx = knowhere::IndexFactory::Instance().Create(std::string(name));
    index->internal = new knowhere::Index<knowhere::IndexNode>(idx);
    auto dataset = knowhere::GenDataSet(build_params->row_nums, build_params->dimension, build_params->binary_vectors);
    knowhere::Json json;
    json[knowhere::meta::METRIC_TYPE] = std::string(build_params->metric_type);
    json[knowhere::meta::DIM] = std::to_string(build_params->dimension);
    for (int i = 0; i < build_params->index_params_size; ++i) {
        json[build_params->index_params[i].key] = std::string(build_params->index_params[i].value);
    }
    auto status = idx.Build(*dataset, json);
    return knowhere::underlying_value(status);
}

int
knowhere_serialize_index(CIndexCtx* index, CBinarySet* binary) {
    KnowhereAssertInfo(index != nullptr, "index is nullptr");
    KnowhereAssertInfo(index->internal != nullptr, "index->internal is nullptr");
    auto idx = static_cast<knowhere::Index<knowhere::IndexNode>*>(index->internal);
    auto ret = new knowhere::BinarySet();
    auto status = idx->Serialize(*ret);
    binary->internal = ret;
    return knowhere::underlying_value(status);
}

int
knowhere_deserialize_index(const char* name, CBinarySet* binary, CIndexCtx* index) {
    KnowhereAssertInfo(binary != nullptr, "binary is nullptr");
    KnowhereAssertInfo(binary->internal != nullptr, "binary->internal is nullptr");
    auto idx = knowhere::IndexFactory::Instance().Create(std::string(name));
    index->internal = new knowhere::Index<knowhere::IndexNode>(idx);
    auto ret = static_cast<knowhere::BinarySet*>(binary->internal);
    auto status = idx.Deserialize(*ret);
    return knowhere::underlying_value(status);
}

int
knowhere_search_index(CIndexCtx* index, CSearchParams* search_params, CSearchResult* out) {
    KnowhereAssertInfo(index != nullptr, "index is nullptr");
    KnowhereAssertInfo(index->internal != nullptr, "index->internal is nullptr");
    KnowhereAssertInfo(search_params != nullptr, "search params is nullptr");
    auto idx = static_cast<knowhere::Index<knowhere::IndexNode>*>(index->internal);
    knowhere::Json json;
    for (int i = 0; i < search_params->search_params_size; ++i) {
        json[search_params->search_params[i].key] = std::string(search_params->search_params[i].value);
    }
    auto dataset =
        knowhere::GenDataSet(search_params->query_nums, search_params->dimension, search_params->binary_vectors);
    knowhere::BitsetView bitset(search_params->bitset_data, search_params->bitset_length);
    auto status = idx->Search(*dataset, json, bitset);
    if (status.has_value()) {
        auto ret = status.value();
        auto total_num = search_params->query_nums * std::stoi(json[knowhere::meta::TOPK].get<std::string>());
        out->row_nums = ret->GetRows();
        out->ids = new int64_t[total_num];
        out->distances = new float[total_num];
        std::copy_n(ret->GetIds(), total_num, out->ids);
        std::copy_n(ret->GetDistance(), total_num, out->distances);
        return knowhere::underlying_value(knowhere::Status::success);
    }
    return knowhere::underlying_value(status.error());
}
