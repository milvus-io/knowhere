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
#include <cstdint>
#include <optional>
#include "common/metric.h"
#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_flat.cuh"
#include "raft/neighbors/ivf_flat_types.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"
#include "index/ivf_raft/ivf_raft_config.h"
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"

namespace knowhere {

namespace detail {
using raft_ivf_flat_index = raft::neighbors::ivf_flat::index<float, std::uint32_t>;
using raft_ivf_pq_index = raft::neighbors::ivf_pq::index<std::uint32_t>;

// TODO(wphicks): Replace this with version from RAFT once merged
struct device_setter {
  device_setter(int new_device) : prev_device_{[]() {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDevice(&result));
    return result;
  }()} {
    RAFT_CUDA_TRY(cudaSetDevice(new_device));
  }

  ~device_setter() {
    RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
  }
 private:
  int prev_device_;
};

}

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<detail::raft_ivf_flat_index> {
    typedef RaftIvfFlatConfig Type;
};
template <>
struct KnowhereConfigType<detail::raft_ivf_pq_index> {
    typedef RaftIvfPqConfig Type;
};

template <typename T>
class RaftIvfIndexNode : public IndexNode {
 public:
    RaftIvfIndexNode(const Object& object) : devs_{}, res_{}, gpu_index_{} {}

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        auto result = Status::success;
        if (gpu_index_) {
          LOG_KNOWHERE_WARNING_ << "index is already trained";
          result = Status::index_already_trained;
        } else if (ivf_raft_cfg.gpu_ids.size() == 1) {
          auto scoped_device = detail::device_setter{ivf_raft_cfg.gpu_ids[0]};
          res_ = raft::device_resources{};
          auto rows = dataset.GetRows();
          auto dim = dataset.GetDim();
          auto* data = reinterpret_cast<float*>(dataset.GetTensor());

          auto stream = res_.get_stream();
          auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
          RAFT_CUDA_TRY(
            cudaMemcpyAsync(
              data_gpu.data(),
              data,
              data_gpu.size() * sizeof(float),
              cudaMemcpyDefault,
              stream.value()
            )
          );
          if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            auto build_params = raft::neighbors::ivf_flat::index_params{
              ivf_raft_cfg.nlist,
              20,  // TODO(wphicks): Confirm just default
              0.5,  // TODO(wphicks): Confirm just default
              false  //TODO(wphicks): Allow configuration?
            };
            gpu_index_ = raft::neighbors::ivf_flat::build(
                res_,
                build_params,
                data_gpu.data(),
                rows,
                dim
            );
          } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            // TODO(wphicks): Specify m?
            auto build_params = raft::neighbors::ivf_pq::index_params{
              ivf_raft_cfg.nlist,
              20,  // TODO(wphicks): Confirm just default
              0.5,  // TODO(wphicks): Confirm just default
              ivf_raft_cfg.nbits
            };
            gpu_index_ = raft::neighbors::ivf_pq::build(
                res_,
                build_params,
                data_gpu.data(),
                rows,
                dim
            );
          } else {
            static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
          }
        } else {
          LOG_KNOWHERE_WARNING_ << "RAFT IVF implementation is single-GPU only";
          result = Status::raft_inner_error;
        }
        return result;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        auto result = Status::success;
        if (!gpu_index_) {
            result = Status::empty_index;
        } else {
          auto rows = dataset.GetRows();
          auto* data = reinterpret_cast<float*>(dataset.GetTensor());

          auto stream = res_.get_stream();
          auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
          RAFT_CUDA_TRY(
            cudaMemcpyAsync(
              data_gpu.data(),
              data,
              data_gpu.size() * sizeof(float),
              cudaMemcpyDefault,
              stream.value()
            )
          );

          if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            raft::neighbors::ivf_flat::add(
                res_,
                &(*gpu_index_),
                data_gpu.data(),
                nullptr, // TODO(wphicks): indices for non-empty
                rows
            );
          } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            raft::neighbors::ivf_pq::add(
                res_,
                &(*gpu_index_),
                data_gpu.data(),
                nullptr, // TODO(wphicks): indices for non-empty
                rows
            );
          } else {
            static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
          }
        }

        return result;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);

        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float*>(dataset.GetTensor());

        auto stream = res_.get_stream();
        auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(
            data_gpu.data(),
            data,
            data_gpu.size() * sizeof(float),
            cudaMemcpyDefault,
            stream.value()
          )
        );

        auto output_size = rows * ivf_raft_cfg.k;
        auto ids = std::unique_ptr<std::int32_t[]>(new std::int32_t[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);

        auto stream = res_.get_stream();
        auto ids_gpu = rmm::device_uvector<std::int32_t>(output_size, stream);
        auto dis_gpu = rmm::device_uvector<float>(output_size, stream);

        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
          auto search_params = raft::neighbors::ivf_flat::search_params{
            ivf_raft_cfg.nprobe
          };
          raft::neighbors::ivf_flat::search(
              search_params,
              *gpu_index_,
              data_gpu.data(),
              rows,
              ivf_raft_cfg.k,
              ids_gpu.data(),
              dis_gpu.data()
          );
        } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
          auto search_params = raft::neighbors::ivf_pq::search_params{
            ivf_raft_cfg.nprobe
          };
          raft::neighbors::ivf_pq::search(
              search_params,
              *gpu_index_,
              data_gpu.data(),
              rows,
              ivf_raft_cfg.k,
              ids_gpu.data(),
              dis_gpu.data()
          );
        } else {
          static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
        }
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(
            ids.get(),
            ids_gpu.data(),
            ids_gpu.size() * sizeof(std::int32_t),
            cudaMemcpyDefault,
            stream.value()
          )
        );
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(
            dis.get(),
            dis_gpu.data(),
            dis_gpu.size() * sizeof(float),
            cudaMemcpyDefault,
            stream.value()
          )
        );
        stream.synchronize();

        return GenResultDataSet(rows, ivf_gpu_cfg.k, ids.release(), dis.release());
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
        return Status::not_implemented;
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        return Status::not_implemented;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    virtual int64_t
    Dim() const override {
        auto result = std::int64_t{};
        if (gpu_index_) {
          result = gpu_index_->dim();
        }
        return result;
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        auto result = std::int64_t{};
        if (gpu_index_) {
          result = gpu_index_->size();
        }
        return result;
    }

    virtual std::string
    Type() const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return "RAFTIVFFLAT";
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return "RAFTIVFPQ";
        }
    }

 private:
    std::vector<int32_t> devs_;
    raft::device_resources res_;
    std::optional<T> gpu_index_;
};

KNOWHERE_REGISTER_GLOBAL(RAFTIVFFLAT, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<RaftIvfIndexNode<detail::raft_ivf_flat_index>>(object));
});
KNOWHERE_REGISTER_GLOBAL(RAFTIVFPQ, [](const Object& object) {
    return Index<IndexNodeThreadPoolWrapper>::Create(std::make_unique<RaftIvfIndexNode<detail::raft_ivf_pq_index>>(object));
});
}  // namespace knowhere
