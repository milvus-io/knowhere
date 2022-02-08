// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** BruteForce provides brute-force search as an option that the data is just
 *  copied to the index without further encoding or organization.
 *
 *  Note that this class does not depend on Faiss or other third-party vector
 *  search libraries (But with some dangling faiss quantifier).
 */

#pragma once

#include <cstdio>
#include "common/Heap.h"
#include "utils/BitsetView.h"
#include "utils/distances_simd.h"

namespace knowhere {

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_L2sqr_sse(
        const float *x,
        const float *y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t *res,
        const faiss::BitsetView bitset = nullptr);

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_inner_product_sse(const float *x,
                           const float *y,
                           size_t d, size_t nx, size_t ny,
                           float_minheap_array_t *res,
                           const faiss::BitsetView bitset = nullptr);

}  // namespace knowhere
