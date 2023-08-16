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
#include "cagra.cuh"
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"

static constexpr uint32_t cuda_concurrent_size = 16;

namespace knowhere {

static std::shared_ptr<ThreadPool>
GlobalThreadPoolRaft() {
    static std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(cuda_concurrent_size);
    return pool;
}

KNOWHERE_REGISTER_GLOBAL(GPU_RAFT_CAGRA, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<CagraIndexNode>(object), GlobalThreadPoolRaft());
});
}  // namespace knowhere
