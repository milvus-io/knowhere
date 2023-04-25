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

#include <omp.h>

#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace knowhere::utils {

inline void
SetBuildOmpThread(const Config& conf) {
    int32_t omp_num =
        CheckKeyInConfig(conf, meta::BUILD_INDEX_OMP_NUM) ? GetMetaBuildIndexOmpNum(conf) : omp_get_max_threads();
    omp_set_num_threads(omp_num);
}

inline void
SetQueryOmpThread(const Config& conf) {
    int32_t omp_num = CheckKeyInConfig(conf, meta::QUERY_OMP_NUM) ? GetMetaQueryOmpNum(conf) : omp_get_max_threads();
    omp_set_num_threads(omp_num);
}

inline uint64_t
hash_vec(const float* x, size_t d) {
    uint64_t h = 0;
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + *(uint32_t*)(x + i);
    }
    return h;
}

}  // namespace knowhere::utils
