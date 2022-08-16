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

#pragma once

#include <string>

#include "knowhere/common/Dataset.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

#define DEFINE_DATASET_GETTER(func_name, key, T)    \
inline T func_name(const DatasetPtr& ds_ptr) {      \
    return ds_ptr->Get<T>(key);                     \
}

#define DEFINE_DATASET_SETTER(func_name, key, T)        \
inline void func_name(DatasetPtr& ds_ptr, T value) {    \
    ds_ptr->Set(key, value);                            \
}

///////////////////////////////////////////////////////////////////////////////

DEFINE_DATASET_GETTER(GetDatasetDim, meta::DIM, const int64_t);
DEFINE_DATASET_SETTER(SetDatasetDim, meta::DIM, const int64_t);

DEFINE_DATASET_GETTER(GetDatasetTensor, meta::TENSOR, const void*);
DEFINE_DATASET_SETTER(SetDatasetTensor, meta::TENSOR, const void*);

DEFINE_DATASET_GETTER(GetDatasetRows, meta::ROWS, const int64_t);
DEFINE_DATASET_SETTER(SetDatasetRows, meta::ROWS, const int64_t);

DEFINE_DATASET_GETTER(GetDatasetIDs, meta::IDS, const int64_t*);
DEFINE_DATASET_SETTER(SetDatasetIDs, meta::IDS, const int64_t*);

DEFINE_DATASET_GETTER(GetDatasetDistance, meta::DISTANCE, const float*);
DEFINE_DATASET_SETTER(SetDatasetDistance, meta::DISTANCE, const float*);

DEFINE_DATASET_GETTER(GetDatasetLims, meta::LIMS, const size_t*);
DEFINE_DATASET_SETTER(SetDatasetLims, meta::LIMS, const size_t*);

DEFINE_DATASET_GETTER(GetDatasetInputIDs, meta::INPUT_IDS, const int64_t*);
DEFINE_DATASET_SETTER(SetDatasetInputIDs, meta::INPUT_IDS, const int64_t*);

DEFINE_DATASET_GETTER(GetDatasetOutputTensor, meta::OUTPUT_TENSOR, const void*);
DEFINE_DATASET_SETTER(SetDatasetOutputTensor, meta::OUTPUT_TENSOR, const void*);

DEFINE_DATASET_GETTER(GetDatasetJsonInfo, meta::JSON_INFO, const std::string);
DEFINE_DATASET_SETTER(SetDatasetJsonInfo, meta::JSON_INFO, const std::string);

DEFINE_DATASET_GETTER(GetDatasetJsonIdSet, meta::JSON_ID_SET, const std::string);
DEFINE_DATASET_SETTER(SetDatasetJsonIdSet, meta::JSON_ID_SET, const std::string);

///////////////////////////////////////////////////////////////////////////////

#define GET_DATA_WITH_IDS(ds_ptr)                     \
    auto rows = knowhere::GetDatasetRows(ds_ptr);     \
    auto dim = knowhere::GetDatasetDim(ds_ptr);       \
    auto p_ids = knowhere::GetDatasetInputIDs(ds_ptr);

#define GET_TENSOR_DATA(ds_ptr)                       \
    auto rows = knowhere::GetDatasetRows(ds_ptr);     \
    auto p_data = knowhere::GetDatasetTensor(ds_ptr);

#define GET_TENSOR_DATA_DIM(ds_ptr)             \
    GET_TENSOR_DATA(ds_ptr)                     \
    auto dim = knowhere::GetDatasetDim(ds_ptr);

extern DatasetPtr
GenDataset(const int64_t nb, const int64_t dim, const void* xb);

extern DatasetPtr
GenDatasetWithIds(const int64_t n, const int64_t dim, const int64_t* ids);

extern DatasetPtr
GenResultDataset(const void* tensor);

extern DatasetPtr
GenResultDataset(const int64_t* ids, const float* distance);

extern DatasetPtr
GenResultDataset(const int64_t* ids, const float* distance, const size_t* lims);

extern DatasetPtr
GenResultDataset(const std::string& json_info, const std::string& json_id_set);

extern DatasetPtr
GenResultDataset(const int64_t* ids,
                 const float* distance,
                 const std::string& json_info,
                 const std::string& json_id_set);

}  // namespace knowhere
