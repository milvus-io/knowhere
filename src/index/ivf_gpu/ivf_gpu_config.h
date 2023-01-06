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

#include "index/ivf/ivf_config.h"

namespace knowhere {

class IvfGpuFlatConfig : public IvfFlatConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

class IvfGpuPqConfig : public IvfPqConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

class IvfGpuSqConfig : public IvfSqConfig {
 public:
    CFG_LIST gpu_ids;
    KNOHWERE_DECLARE_CONFIG(IvfGpuSqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("the gpu id, which device use")
            .set_default({
                0,
            })
            .for_train();
    }
};

}  // namespace knowhere
