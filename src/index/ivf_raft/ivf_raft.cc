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

#include <cstddef>
#include "common/metric.h"
#include "raft/core/device_resources.hpp"
#include "index/ivf_gpu/ivf_gpu_config.h"
#include "io/FaissIO.h"
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"

namespace knowhere {

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<raft::neighbors::ivf_flat::index> {
    typedef RaftIvfFlatConfig Type;
};
template <>
struct KnowhereConfigType<raft::neighbors::ivf_pq::index> {
    typedef RaftIvfPqConfig Type;
};

template <typename T>
class RaftIvfIndexNode : public IndexNode {
 public:
    RaftIvfIndexNode(const Object& object) : devs_{}, res_{}, gpu_index_(nullptr) {}

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        // TODO(wphicks)
        for (std::size_t i = 0; i < ivf_gpu_cfg.gpu_ids.size(); ++i) {
            // QUESTION(wphicks): It looks like we are just grabbing as many
            // devices as the length of the given IDs as opposed to grabbing
            // those specific devices. Is this correct and desired?
            this->devs_.push_back(i);
        }
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        // TODO(wphicks)
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        // TODO(wphicks)
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return unexpected(Status::not_implemented);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        // TODO(wphicks)
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        // TODO(wphicks)
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    virtual int64_t
    Dim() const override {
        return gpu_index_.dim();
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        return gpu_index_.size();
    }

    virtual std::string
    Type() const override {
        if constexpr (std::is_same<raft::neighbors::ivf_flat::index, T>::value) {
            return "RAFTIVFFLAT";
        }
        if constexpr (std::is_same<raft::neighbors::ivf_pq::index, T>::value) {
            return "RAFTIVFPQ";
        }
    }

 private:
    std::vector<int32_t> devs_;
    raft::device_resources res_;
    T gpu_index_;
};

KNOWHERE_REGISTER_GLOBAL(RAFTIVFFLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<RaftIvfIndexNode<raft::neighbors::ivf_flat::index>>(object));
});
KNOWHERE_REGISTER_GLOBAL(RAFTIVFPQ, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<RaftIvfIndexNode<raft::neighbors::ivf_pq::index>>(object));
});
}  // namespace knowhere
