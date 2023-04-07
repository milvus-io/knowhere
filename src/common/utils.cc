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

#include "knowhere/utils.h"

#include <cmath>
#include <cstdint>

#include "knowhere/log.h"

namespace knowhere {

const float floatDiff = 0.00001;

void
NormalizeVec(float* vector, int32_t dim) {
    double sq_sum = 0.0;
    for (int32_t j = 0; j < dim; j++) {
        sq_sum += (double)vector[j] * (double)vector[j];
    }
    if (std::abs(1.0f - sq_sum) > floatDiff) {
        double inv_sq_sum = 1.0 / std::sqrt(sq_sum);
        for (int32_t j = 0; j < dim; j++) {
            vector[j] = (float)(vector[j] * inv_sq_sum);
        }
    }
}

void
Normalize(const DataSet& dataset) {
    auto rows = dataset.GetRows();
    auto dim = dataset.GetDim();
    float* data = (float*)dataset.GetTensor();

    LOG_KNOWHERE_INFO_ << "vector normalize, rows " << rows << ", dim " << dim;

    for (int32_t i = 0; i < rows; i++) {
        NormalizeVec(data + i * dim, dim);
    }
}

}  // namespace knowhere
