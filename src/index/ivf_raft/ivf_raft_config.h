/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef IVF_RAFT_CONFIG_H
#define IVF_RAFT_CONFIG_H

#include "index/ivf/ivf_config.h"

namespace knowhere {

class RaftIvfFlatConfig : public IvfFlatConfig {
    public:
        CFG_LIST gpu_ids;
        KNOHWERE_DECLARE_CONFIG(RaftIvfFlatConfig) {
            KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
                .description("gpu device ids")
                .set_default({
                        0,
                        })
            .for_train();
        }
};

class RaftIvfPqConfig : public IvfPqConfig {
    public:
        CFG_LIST gpu_ids;
        KNOHWERE_DECLARE_CONFIG(RaftIvfPqConfig) {
            KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
                .description("gpu device ids")
                .set_default({
                        0,
                        })
            .for_train();
        }
};

}  // namespace knowhere
#endif /* IVF_CONFIG_H */
