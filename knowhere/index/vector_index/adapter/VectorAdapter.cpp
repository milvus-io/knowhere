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

#include <memory>

#include "common/Dataset.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/IndexParameter.h"

namespace knowhere {

DatasetPtr
GenDataset(const int64_t nb, const int64_t dim, const void* xb) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetRows(ret_ds, nb);
    SetDatasetDim(ret_ds, dim);
    SetDatasetTensor(ret_ds, xb);
    return ret_ds;
}

DatasetPtr
GenDatasetWithIds(const int64_t n, const int64_t dim, const int64_t* ids) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetRows(ret_ds, n);
    SetDatasetDim(ret_ds, dim);
    SetDatasetInputIDs(ret_ds, ids);
    return ret_ds;
}

DatasetPtr
GenResultDataset(const void* tensor) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetOutputTensor(ret_ds, tensor);
    return ret_ds;
}

DatasetPtr
GenResultDataset(const int64_t* ids, const float* distance) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetIDs(ret_ds, ids);
    SetDatasetDistance(ret_ds, distance);
    return ret_ds;
}

DatasetPtr
GenResultDataset(const int64_t* ids, const float* distance, const size_t* lims) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetIDs(ret_ds, ids);
    SetDatasetDistance(ret_ds, distance);
    SetDatasetLims(ret_ds, lims);
    return ret_ds;
}

DatasetPtr
GenResultDataset(const std::string& json_info, const std::string& json_id_set) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetJsonInfo(ret_ds, json_info);
    SetDatasetJsonIdSet(ret_ds, json_id_set);
    return ret_ds;
}

DatasetPtr
GenResultDataset(const int64_t* ids,
                 const float* distance,
                 const std::string& json_info,
                 const std::string& json_id_set) {
    auto ret_ds = std::make_shared<Dataset>();
    SetDatasetIDs(ret_ds, ids);
    SetDatasetDistance(ret_ds, distance);
    SetDatasetJsonInfo(ret_ds, json_info);
    SetDatasetJsonIdSet(ret_ds, json_id_set);
    return ret_ds;
}

}  // namespace knowhere
