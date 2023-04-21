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

#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H
#include "knowhere/bitsetview.h"
#include "knowhere/dataset.h"
#include "knowhere/factory.h"

namespace knowhere {

class BruteForce {
 public:
    static expected<DataSetPtr, Status>
    Search(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config, const BitsetView& bitset);

    static Status
    SearchWithBuf(const DataSetPtr base_dataset, const DataSetPtr query_dataset, int64_t* ids, float* dis,
                  const Json& config, const BitsetView& bitset);

    static expected<DataSetPtr, Status>
    RangeSearch(const DataSetPtr base_dataset, const DataSetPtr query_dataset, const Json& config,
                const BitsetView& bitset);
};

}  // namespace knowhere

#endif /* BRUTE_FORCE_H */
