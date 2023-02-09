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

#ifndef ANNOY_CONFIG_H
#define ANNOY_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {
class AnnoyConfig : public BaseConfig {
 public:
    int n_trees;
    int search_k;
    KNOHWERE_DECLARE_CONFIG(AnnoyConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(n_trees)
            .description("annoy n_trees.")
            .set_default(8)
            .set_range(1, 2048)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_k)
            .description("annoy search k.")
            .set_default(100)
            .set_range(-1, 65536)
            .for_search();
    }
};

}  // namespace knowhere

#endif /* ANNOY_CONFIG_H */
