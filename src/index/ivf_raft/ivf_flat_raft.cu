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
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"
#include "ivf_raft.cuh"

namespace knowhere {
KNOWHERE_REGISTER_GLOBAL(RAFTIVFFLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<RaftIvfIndexNode<detail::raft_ivf_flat_index>>(object));
});
}  // namespace knowhere
