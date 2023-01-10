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

#ifndef HNSW_CONFIG_H
#define HNSW_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {
class HnswConfig : public BaseConfig {
 public:
    int M;
    int efConstruction;
    int ef;
    int overview_levels;
    KNOHWERE_DECLARE_CONFIG(HnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M)
            .description("hnsw M")
            .set_default(16)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction)
            .description("hnsw efConstruction")
            .set_default(200)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("hnsw ef")
            .set_default(32)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(overview_levels)
            .description("hnsw overview levels for feder")
            .set_default(3)
            .for_feder();
    }
};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
