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

#ifndef IVF_RAFT_CUH
#define IVF_RAFT_CUH

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "common/raft_metric.h"
#include "index/ivf_raft/ivf_raft_config.h"
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"
#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_flat.cuh"
#include "raft/neighbors/ivf_flat_types.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"
#include "thrust/execution_policy.h"
#include "thrust/sequence.h"

namespace knowhere {

namespace raft_res_pool {

struct context {
    context()
        : resources_(
              []() {
                  return new rmm::cuda_stream();  // Avoid program exit datart
                                                  // unload error
              }()
                  ->view(),
              nullptr, rmm::mr::get_current_device_resource()) {
    }
    ~context() = default;
    context(context&&) = delete;
    context(context const&) = delete;
    context&
    operator=(context&&) = delete;
    context&
    operator=(context const&) = delete;
    raft::device_resources resources_;
};

context&
get_context() {
    thread_local context ctx;
    return ctx;
};

class resource {
 public:
    static resource&
    instance() {
        static resource res;
        return res;
    }
    void
    init(rmm::cuda_device_id device_id) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(device_id.value());
        if (it == map_.end()) {
            auto mr_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(&up_mr_);
            rmm::mr::set_per_device_resource(device_id, mr_.get());
            map_[device_id.value()] = std::move(mr_);
        }
    }

 private:
    resource(){};
    ~resource(){};
    resource(resource&&) = delete;
    resource(resource const&) = delete;
    resource&
    operator=(resource&&) = delete;
    resource&
    operator=(context const&) = delete;
    rmm::mr::cuda_memory_resource up_mr_;
    std::map<rmm::cuda_device_id::value_type,
             std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>>
        map_;
    mutable std::mutex mtx_;
};

};  // namespace raft_res_pool

namespace detail {
using raft_ivf_flat_index = raft::neighbors::ivf_flat::index<float, std::int64_t>;
using raft_ivf_pq_index = raft::neighbors::ivf_pq::index<std::int64_t>;

// TODO(wphicks): Replace this with version from RAFT once merged
struct device_setter {
    device_setter(int new_device)
        : prev_device_{[]() {
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

namespace codebook {
auto static constexpr const PER_SUBSPACE = "PER_SUBSPACE";
auto static constexpr const PER_CLUSTER = "PER_CLUSTER";
}  // namespace codebook

inline expected<raft::neighbors::ivf_pq::codebook_gen, Status>
str_to_codebook_gen(std::string const& str) {
    static const std::unordered_map<std::string, raft::neighbors::ivf_pq::codebook_gen> name_map = {
        {codebook::PER_SUBSPACE, raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE},
        {codebook::PER_CLUSTER, raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER},
    };

    auto it = name_map.find(str);
    if (it == name_map.end())
        return unexpected(Status::invalid_args);
    return it->second;
}

namespace cuda_type {
auto static constexpr const CUDA_R_16F = "CUDA_R_16F";
auto static constexpr const CUDA_C_16F = "CUDA_C_16F";
auto static constexpr const CUDA_R_16BF = "CUDA_R_16BF";
auto static constexpr const CUDA_C_16BF = "CUDA_C_16BF";
auto static constexpr const CUDA_R_32F = "CUDA_R_32F";
auto static constexpr const CUDA_C_32F = "CUDA_C_32F";
auto static constexpr const CUDA_R_64F = "CUDA_R_64F";
auto static constexpr const CUDA_C_64F = "CUDA_C_64F";
auto static constexpr const CUDA_R_8I = "CUDA_R_8I";
auto static constexpr const CUDA_C_8I = "CUDA_C_8I";
auto static constexpr const CUDA_R_8U = "CUDA_R_8U";
auto static constexpr const CUDA_C_8U = "CUDA_C_8U";
auto static constexpr const CUDA_R_32I = "CUDA_R_32I";
auto static constexpr const CUDA_C_32I = "CUDA_C_32I";
auto static constexpr const CUDA_R_8F_E4M3 = "CUDA_R_8F_E4M3";
auto static constexpr const CUDA_R_8F_E5M2 = "CUDA_R_8F_E5M2";
}  // namespace cuda_type

inline expected<cudaDataType_t, Status>
str_to_cuda_dtype(std::string const& str) {
    static const std::unordered_map<std::string, cudaDataType_t> name_map = {
        {cuda_type::CUDA_R_16F, CUDA_R_16F},   {cuda_type::CUDA_C_16F, CUDA_C_16F},
        {cuda_type::CUDA_R_16BF, CUDA_R_16BF}, {cuda_type::CUDA_C_16BF, CUDA_C_16BF},
        {cuda_type::CUDA_R_32F, CUDA_R_32F},   {cuda_type::CUDA_C_32F, CUDA_C_32F},
        {cuda_type::CUDA_R_64F, CUDA_R_64F},   {cuda_type::CUDA_C_64F, CUDA_C_64F},
        {cuda_type::CUDA_R_8I, CUDA_R_8I},     {cuda_type::CUDA_C_8I, CUDA_C_8I},
        {cuda_type::CUDA_R_8U, CUDA_R_8U},     {cuda_type::CUDA_C_8U, CUDA_C_8U},
        {cuda_type::CUDA_R_32I, CUDA_R_32I},   {cuda_type::CUDA_C_32I, CUDA_C_32I},
        // not support, when we use cuda 11.6
        //{cuda_type::CUDA_R_8F_E4M3, CUDA_R_8F_E4M3}, {cuda_type::CUDA_R_8F_E5M2, CUDA_R_8F_E5M2},

    };

    auto it = name_map.find(str);
    if (it == name_map.end())
        return unexpected(Status::invalid_args);
    return it->second;
}

}  // namespace detail

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
    RaftIvfIndexNode(const Object& object) : devs_{}, gpu_index_{} {
    }

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
        if (gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        } else if (ivf_raft_cfg.gpu_ids.size() == 1) {
            try {
                auto metric = Str2RaftMetricType(ivf_raft_cfg.metric_type);
                if (!metric.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check metric value: " << ivf_raft_cfg.metric_type;
                    return metric.error();
                }
                if (metric.value() != raft::distance::DistanceType::L2Expanded &&
                    metric.value() != raft::distance::DistanceType::InnerProduct) {
                    LOG_KNOWHERE_WARNING_ << "selected metric not supported in RAFT IVF indexes: "
                                          << ivf_raft_cfg.metric_type;
                    return Status::invalid_metric_type;
                }
                devs_.insert(devs_.begin(), ivf_raft_cfg.gpu_ids.begin(), ivf_raft_cfg.gpu_ids.end());
                auto scoped_device = detail::device_setter{*ivf_raft_cfg.gpu_ids.begin()};
                raft_res_pool::resource::instance().init(rmm::cuda_device_id(devs_[0]));
                auto* res_ = &raft_res_pool::get_context().resources_;
                auto rows = dataset.GetRows();
                auto dim = dataset.GetDim();
                auto* data = reinterpret_cast<float const*>(dataset.GetTensor());

                auto stream = res_->get_stream();
                auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
                RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data(), data, data_gpu.size() * sizeof(float), cudaMemcpyDefault,
                                              stream.value()));
                if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                    auto build_params = raft::neighbors::ivf_flat::index_params{};
                    build_params.metric = metric.value();
                    build_params.n_lists = ivf_raft_cfg.nlist;
                    build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters;
                    build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction;
                    build_params.adaptive_centers = ivf_raft_cfg.adaptive_centers;
                    gpu_index_ = raft::neighbors::ivf_flat::build<float, std::int64_t>(*res_, build_params,
                                                                                       data_gpu.data(), rows, dim);
                } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                    auto build_params = raft::neighbors::ivf_pq::index_params{};
                    build_params.metric = metric.value();
                    build_params.n_lists = ivf_raft_cfg.nlist;
                    build_params.pq_bits = ivf_raft_cfg.nbits;
                    build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters;
                    build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction;
                    build_params.pq_dim = ivf_raft_cfg.pq_dim;
                    auto codebook_kind = detail::str_to_codebook_gen(ivf_raft_cfg.codebook_kind);
                    if (!codebook_kind.has_value()) {
                        LOG_KNOWHERE_WARNING_ << "please check codebook kind: " << ivf_raft_cfg.codebook_kind;
                        return codebook_kind.error();
                    }
                    build_params.codebook_kind = codebook_kind.value();
                    build_params.force_random_rotation = ivf_raft_cfg.force_random_rotation;
                    gpu_index_ = raft::neighbors::ivf_pq::build<float, std::int64_t>(*res_, build_params,
                                                                                     data_gpu.data(), rows, dim);
                } else {
                    static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
                }
                dim_ = dim;
                counts_ = rows;
                stream.synchronize();

            } catch (std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
                return Status::raft_inner_error;
            }
        } else {
            LOG_KNOWHERE_WARNING_ << "RAFT IVF implementation is single-GPU only";
            return Status::raft_inner_error;
        }
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        auto result = Status::success;
        if (!gpu_index_) {
            result = Status::index_not_trained;
        } else {
            try {
                auto rows = dataset.GetRows();
                auto dim = dataset.GetDim();
                auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
                auto scoped_device = detail::device_setter{devs_[0]};
                raft_res_pool::resource::instance().init(rmm::cuda_device_id(devs_[0]));
                auto* res_ = &raft_res_pool::get_context().resources_;

                auto stream = res_->get_stream();
                // TODO(wphicks): Clean up transfer with raft
                // buffer objects when available
                auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
                RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data(), data, data_gpu.size() * sizeof(float), cudaMemcpyDefault,
                                              stream.value()));

                auto indices = rmm::device_uvector<std::int64_t>(rows, stream);
                thrust::sequence(thrust::device, indices.begin(), indices.end(), gpu_index_->size());

                if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                    raft::neighbors::ivf_flat::extend<float, std::int64_t>(*res_, *gpu_index_, data_gpu.data(),
                                                                           indices.data(), rows);
                } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                    raft::neighbors::ivf_pq::extend<float, std::int64_t>(*res_, *gpu_index_, data_gpu.data(),
                                                                         indices.data(), rows);
                } else {
                    static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
                }
                dim_ = dim;
                counts_ = rows;
            } catch (std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
                result = Status::raft_inner_error;
            }
        }

        return result;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto output_size = rows * ivf_raft_cfg.k;
        auto ids = std::unique_ptr<std::int64_t[]>(new std::int64_t[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);

        try {
            auto scoped_device = detail::device_setter{devs_[0]};
            auto* res_ = &raft_res_pool::get_context().resources_;

            auto stream = res_->get_stream();
            // TODO(wphicks): Clean up transfer with raft
            // buffer objects when available
            auto data_gpu = rmm::device_uvector<float>(rows * dim, stream);
            RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data(), data, data_gpu.size() * sizeof(float), cudaMemcpyDefault,
                                          stream.value()));

            auto ids_gpu = rmm::device_uvector<std::int64_t>(output_size, stream);
            auto dis_gpu = rmm::device_uvector<float>(output_size, stream);

            if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                auto search_params = raft::neighbors::ivf_flat::search_params{};
                search_params.n_probes = ivf_raft_cfg.nprobe;
                raft::neighbors::ivf_flat::search<float, std::int64_t>(*res_, search_params, *gpu_index_,
                                                                       data_gpu.data(), rows, ivf_raft_cfg.k,
                                                                       ids_gpu.data(), dis_gpu.data());
            } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                auto search_params = raft::neighbors::ivf_pq::search_params{};
                search_params.n_probes = ivf_raft_cfg.nprobe;
                auto lut_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.lut_dtype);
                if (!lut_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check lookup dtype: " << ivf_raft_cfg.lut_dtype;
                    return unexpected(lut_dtype.error());
                }
                if (lut_dtype.value() != CUDA_R_32F && lut_dtype.value() != CUDA_R_16F &&
                    lut_dtype.value() != CUDA_R_8U) {
                    LOG_KNOWHERE_WARNING_ << "selected lookup dtype not supported: " << ivf_raft_cfg.lut_dtype;
                    return unexpected(Status::invalid_args);
                }
                search_params.lut_dtype = lut_dtype.value();
                auto internal_distance_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.internal_distance_dtype);
                if (!internal_distance_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check internal distance dtype: "
                                          << ivf_raft_cfg.internal_distance_dtype;
                    return unexpected(internal_distance_dtype.error());
                }
                if (internal_distance_dtype.value() != CUDA_R_32F && internal_distance_dtype.value() != CUDA_R_16F) {
                    LOG_KNOWHERE_WARNING_ << "selected internal distance dtype not supported: "
                                          << ivf_raft_cfg.internal_distance_dtype;
                    return unexpected(Status::invalid_args);
                }
                search_params.internal_distance_dtype = internal_distance_dtype.value();
                search_params.preferred_shmem_carveout = search_params.preferred_shmem_carveout;
                raft::neighbors::ivf_pq::search<float, std::int64_t>(*res_, search_params, *gpu_index_, data_gpu.data(),
                                                                     rows, ivf_raft_cfg.k, ids_gpu.data(),
                                                                     dis_gpu.data());
            } else {
                static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
            }
            RAFT_CUDA_TRY(cudaMemcpyAsync(ids.get(), ids_gpu.data(), ids_gpu.size() * sizeof(std::int64_t),
                                          cudaMemcpyDefault, stream.value()));
            RAFT_CUDA_TRY(cudaMemcpyAsync(dis.get(), dis_gpu.data(), dis_gpu.size() * sizeof(float), cudaMemcpyDefault,
                                          stream.value()));
            stream.synchronize();
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return unexpected(Status::raft_inner_error);
        }

        return GenResultDataSet(rows, ivf_raft_cfg.k, ids.release(), dis.release());
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
        if (!gpu_index_.has_value())
            return Status::empty_index;
        std::stringbuf buf;

        std::ostream os(&buf);

        os.write((char*)(&this->dim_), sizeof(this->dim_));
        os.write((char*)(&this->counts_), sizeof(this->counts_));
        os.write((char*)(&this->devs_[0]), sizeof(this->devs_[0]));

        auto scoped_device = detail::device_setter{devs_[0]};
        raft_res_pool::resource::instance().init(rmm::cuda_device_id(devs_[0]));
        auto* res_ = &raft_res_pool::get_context().resources_;

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            raft::serialize_scalar(*res_, os, gpu_index_->size());
            raft::serialize_scalar(*res_, os, gpu_index_->dim());
            raft::serialize_scalar(*res_, os, gpu_index_->n_lists());
            raft::serialize_scalar(*res_, os, gpu_index_->metric());
            raft::serialize_scalar(*res_, os, gpu_index_->veclen());
            raft::serialize_scalar(*res_, os, gpu_index_->adaptive_centers());
            raft::serialize_mdspan(*res_, os, gpu_index_->data());
            raft::serialize_mdspan(*res_, os, gpu_index_->indices());
            raft::serialize_mdspan(*res_, os, gpu_index_->list_sizes());
            raft::serialize_mdspan(*res_, os, gpu_index_->list_offsets());
            raft::serialize_mdspan(*res_, os, gpu_index_->centers());
            if (gpu_index_->center_norms()) {
                bool has_norms = true;
                serialize_scalar(*res_, os, has_norms);
                serialize_mdspan(*res_, os, *gpu_index_->center_norms());
            } else {
                bool has_norms = false;
                serialize_scalar(*res_, os, has_norms);
            }
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            raft::serialize_scalar(*res_, os, gpu_index_->size());
            raft::serialize_scalar(*res_, os, gpu_index_->dim());
            raft::serialize_scalar(*res_, os, gpu_index_->pq_bits());
            raft::serialize_scalar(*res_, os, gpu_index_->pq_dim());

            raft::serialize_scalar(*res_, os, gpu_index_->metric());
            raft::serialize_scalar(*res_, os, gpu_index_->codebook_kind());
            raft::serialize_scalar(*res_, os, gpu_index_->n_lists());
            raft::serialize_scalar(*res_, os, gpu_index_->n_nonempty_lists());

            raft::serialize_mdspan(*res_, os, gpu_index_->pq_centers());
            raft::serialize_mdspan(*res_, os, gpu_index_->pq_dataset());
            raft::serialize_mdspan(*res_, os, gpu_index_->indices());
            raft::serialize_mdspan(*res_, os, gpu_index_->rotation_matrix());
            raft::serialize_mdspan(*res_, os, gpu_index_->list_offsets());
            raft::serialize_mdspan(*res_, os, gpu_index_->list_sizes());
            raft::serialize_mdspan(*res_, os, gpu_index_->centers());
            raft::serialize_mdspan(*res_, os, gpu_index_->centers_rot());
        }

        os.flush();
        std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);

        memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
        binset.Append(this->Type(), index_binary, buf.str().size());
        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset) override {
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        buf.sputn((char*)binary->data.get(), binary->size);
        std::istream is(&buf);

        is.read((char*)(&this->dim_), sizeof(this->dim_));
        is.read((char*)(&this->counts_), sizeof(this->counts_));
        this->devs_.resize(1);
        is.read((char*)(&this->devs_[0]), sizeof(this->devs_[0]));
        auto scoped_device = detail::device_setter{devs_[0]};

        raft_res_pool::resource::instance().init(rmm::cuda_device_id(devs_[0]));
        auto* res_ = &raft_res_pool::get_context().resources_;

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            auto n_rows = raft::deserialize_scalar<std::int64_t>(*res_, is);
            auto dim = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            auto n_lists = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            auto metric = raft::deserialize_scalar<raft::distance::DistanceType>(*res_, is);
            auto veclen = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            bool adaptive_centers = raft::deserialize_scalar<bool>(*res_, is);

            T index_ = T(*res_, metric, n_lists, adaptive_centers, dim);

            index_.allocate(*res_, n_rows);
            raft::deserialize_mdspan(*res_, is, index_.data());
            raft::deserialize_mdspan(*res_, is, index_.indices());
            raft::deserialize_mdspan(*res_, is, index_.list_sizes());
            raft::deserialize_mdspan(*res_, is, index_.list_offsets());
            raft::deserialize_mdspan(*res_, is, index_.centers());
            bool has_norms = raft::deserialize_scalar<bool>(*res_, is);
            if (has_norms) {
                if (!index_.center_norms()) {
                    RAFT_FAIL("Error inconsistent center norms");
                } else {
                    auto center_norms = *index_.center_norms();
                    raft::deserialize_mdspan(*res_, is, center_norms);
                }
            }
            res_->sync_stream();
            is.sync();
            gpu_index_ = T(std::move(index_));
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            auto n_rows = raft::deserialize_scalar<std::int64_t>(*res_, is);
            auto dim = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            auto pq_bits = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            auto pq_dim = raft::deserialize_scalar<std::uint32_t>(*res_, is);

            auto metric = raft::deserialize_scalar<raft::distance::DistanceType>(*res_, is);
            auto codebook_kind = raft::deserialize_scalar<raft::neighbors::ivf_pq::codebook_gen>(*res_, is);
            auto n_lists = raft::deserialize_scalar<std::uint32_t>(*res_, is);
            auto n_nonempty_lists = raft::deserialize_scalar<std::uint32_t>(*res_, is);

            T index_ = T(*res_, metric, codebook_kind, n_lists, dim, pq_bits, pq_dim, n_nonempty_lists);
            index_.allocate(*res_, n_rows);

            raft::deserialize_mdspan(*res_, is, index_.pq_centers());
            raft::deserialize_mdspan(*res_, is, index_.pq_dataset());
            raft::deserialize_mdspan(*res_, is, index_.indices());
            raft::deserialize_mdspan(*res_, is, index_.rotation_matrix());
            raft::deserialize_mdspan(*res_, is, index_.list_offsets());
            raft::deserialize_mdspan(*res_, is, index_.list_sizes());
            raft::deserialize_mdspan(*res_, is, index_.centers());
            raft::deserialize_mdspan(*res_, is, index_.centers_rot());
            res_->sync_stream();
            is.sync();
            gpu_index_ = T(std::move(index_));
        }
        // TODO(yusheng.ma):support no raw data mode
        /*
#define RAW_DATA "RAW_DATA"
    auto data = binset.GetByName(RAW_DATA);
    raft_gpu::raw_data_copy(*this->index_, data->data.get(), data->size);
    */
        is.sync();

        return Status::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    virtual int64_t
    Dim() const override {
        return dim_;
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        return counts_;
    }

    virtual std::string
    Type() const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return "RAFT_IVF_FLAT";
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return "RAFT_IVF_PQ";
        }
    }

 private:
    std::vector<int32_t> devs_;
    int64_t dim_ = 0;
    int64_t counts_ = 0;
    std::optional<T> gpu_index_;
};
}  // namespace knowhere
#endif /* IVF_RAFT_CUH */
