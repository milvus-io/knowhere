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

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"

// This is not a valid EF value for HNSW
// This value is used to tell if HnswConfig.ef is coming from user or not
const int64_t kDefaultHnswEfPlaceholder = -1;

namespace knowhere {
class HnswConfig : public BaseConfig {
 public:
    int M;
    int efConstruction;
    int ef;
    int overview_levels;
    KNOHWERE_DECLARE_CONFIG(HnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M).description("hnsw M").set_default(30).set_range(1, 2048).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction)
            .description("hnsw efConstruction")
            .set_default(360)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("hnsw ef")
            .set_default(kDefaultHnswEfPlaceholder)
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(overview_levels)
            .description("hnsw overview levels for feder")
            .set_default(3)
            .set_range(1, 5)
            .for_feder();
    }

    Status
    CheckAndAdjustConfig() override {
        auto& hnsw_cfg = static_cast<HnswConfig&>(*this);
        auto maxef = std::max(65536, hnsw_cfg.k * 2);
        if (hnsw_cfg.ef > maxef) {
            LOG_KNOWHERE_ERROR_ << "ef should be in range: [topk, max(65536, topk * 2)]";
            return Status::out_of_range_in_json;
        }
        if (hnsw_cfg.ef < hnsw_cfg.k) {
            if (hnsw_cfg.ef == kDefaultHnswEfPlaceholder) {
                // ef is set by default value, set ef to k
                hnsw_cfg.ef = hnsw_cfg.k;
            } else {
                // ef is set by user
                LOG_KNOWHERE_ERROR_ << "ef should be in range: [topk, max(65536, topk * 2)]";
                return Status::out_of_range_in_json;
            }
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
