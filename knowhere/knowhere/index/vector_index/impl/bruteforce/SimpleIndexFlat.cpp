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

#include "knowhere/common/Heap.h"
#include "knowhere/index/vector_index/impl/bruteforce/BruteForce.h"
#include "knowhere/index/vector_index/impl/bruteforce/SimpleIndexFlat.h"

namespace knowhere {

SimpleIndexFlat::SimpleIndexFlat(idx_t d, MetricType metric) {
    this->d = d;
    this->metric_type = metric;
}

void SimpleIndexFlat::add(idx_t n, const float *x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}

void SimpleIndexFlat::search(idx_t n, const float *x, idx_t k,
                        float *distances, idx_t *labels,
                        const faiss::BitsetView bitset) const {
    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
        knn_inner_product_sse(x, xb.data(), d, n, ntotal, &res, bitset);
    } else {
        // metric_type == METRIC_L2
        float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
        knn_L2sqr_sse(x, xb.data(), d, n, ntotal, &res, bitset);
    }
}

void SimpleIndexFlat::train(idx_t n, const float* x) {
    // Do nothing.
}

} // namespace knowhere
