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
 */
#pragma once

#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/utils/BitsetView.h"

namespace knowhere {

void
BruteForceSearch(const knowhere::MetricType& metric_type,
                 const void* xb,
                 const void* xq,
                 const int64_t dim,
                 const int64_t nb,
                 const int64_t nq,
                 const int64_t topk,
                 int64_t* labels,
                 float* distances,
                 const faiss::BitsetView bitset);

}  // namespace knowhere
