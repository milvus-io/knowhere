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

// TODO: Add back the omp include when issue #15585 is resolved.
// https://github.com/milvus-io/milvus/issues/15585
//#include <omp.h>
#include "index/vector_index/impl/bruteforce/BruteForce.h"

namespace knowhere {

void knn_L2sqr_sse(
        const float *x,
        const float *y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t *res,
        const faiss::BitsetView bitset) {
    size_t k = res->k;

    float *value = res->val;
    int64_t *labels = res->ids;

// TODO: Re-enable parallel run when issue #15585 is resolved.
// https://github.com/milvus-io/milvus/issues/15585
//#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        const float *x_i = x + i * d;
        const float *y_j = y;

        float *__restrict val_ = value + i * k;
        int64_t *__restrict ids_ = labels + i * k;

        for (size_t j = 0; j < k; j++) {
            val_[j] = 1.0 / 0.0;
            ids_[j] = -1;
        }

        for (size_t j = 0; j < ny; j++) {
            if (!bitset || !bitset.test(j)) {
                float disij = faiss::fvec_L2sqr_ref(x_i, y_j, d);
                if (disij < val_[0]) {
                    maxheap_swap_top(k, val_, ids_, disij, j);
                }
            }
            y_j += d;
        }

        maxheap_reorder(k, val_, ids_);
    }
}

void knn_inner_product_sse(const float * x,
                                   const float * y,
                                   size_t d, size_t nx, size_t ny,
                                   float_minheap_array_t * res,
                                   const faiss::BitsetView bitset) {
    size_t k = res->k;
    float * value = res->val;
    int64_t * labels = res->ids;

// TODO: Re-enable parallel run when issue #15585 is resolved.
// https://github.com/milvus-io/milvus/issues/15585
//#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        const float *x_i = x + i * d;
        const float *y_j = y;

        float * __restrict val_ = value  + i * k;
        int64_t * __restrict ids_ = labels  + i * k;

        for (size_t j = 0; j < k; j++) {
            val_[j] = -1.0 / 0.0;
            ids_[j] = -1;
        }

        for (size_t j = 0; j < ny; j++) {
            if (!bitset || !bitset.test(j)) {
                float disij = faiss::fvec_inner_product_ref(x_i, y_j, d);
                if (disij > val_[0]) {
                    minheap_swap_top(k, val_, ids_, disij, j);
                }
            }
            y_j += d;
        }

        minheap_reorder(k, val_, ids_);
    }
}

}  // namespace knowhere
