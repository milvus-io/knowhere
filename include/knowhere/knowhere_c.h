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

#pragma once

#include "knowhere/expected.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct CKnowhereConfig {
    const char* simd_type;
    const uint32_t thread_num;
} CKnowhereConfig;

typedef struct KV {
    const char* key;
    const char* value;
} KV;

typedef struct CBuildParams {
    const char* metric_type;
    const KV* index_params;
    const int64_t index_params_size;

    const int64_t row_nums;
    const int64_t dimension;
    const void* binary_vectors;
} CBuildParams;

typedef struct CIndexCtx {
    void* internal;
} CIndexCtx;

typedef struct CBinarySet {
    void* internal;
} CBinarySet;

typedef struct CSearchParams {
    const KV* search_params;
    const int64_t search_params_size;

    const int64_t query_nums;
    const int64_t dimension;
    const void* binary_vectors;

    const uint8_t* bitset_data;
    const int64_t bitset_length;
} CSearchParams;

typedef struct CSearchResult {
    int64_t row_nums;
    int64_t* ids;
    float* distances;
} CSearchResult;

int
knowhere_init(CKnowhereConfig* config);

int
knowhere_build_index(const char* name, CBuildParams* build_params, CIndexCtx* index);

int
knowhere_serialize_index(CIndexCtx* index, CBinarySet* binary);

int
knowhere_deserialize_index(const char* name, CBinarySet* binary, CIndexCtx* index);

int
knowhere_search_index(CIndexCtx* index, CSearchParams* search_params, CSearchResult* out);

int
knowhere_destroy_index(CIndexCtx* index);

int
knowhere_destroy_binary_set(CBinarySet* binary);

int
knowhere_destroy_search_result(CSearchResult* result);

#ifdef __cplusplus
}
#endif
