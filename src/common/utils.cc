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
#include "simd/hook.h"

namespace knowhere {

const float FloatAccuracy = 0.00001;

void
NormalizeVec(float* x, int32_t d) {
    float norm_l2_sqr = faiss::fvec_norm_L2sqr(x, d);
    if (norm_l2_sqr > 0 && std::abs(1.0f - norm_l2_sqr) > FloatAccuracy) {
        float norm_l2 = std::sqrt(norm_l2_sqr);
        for (int32_t i = 0; i < d; i++) {
            x[i] = x[i] / norm_l2;
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
