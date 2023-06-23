#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "knowhere/log.h"
#include "raft/core/device_resources.hpp"
#include "rmm/cuda_stream_pool.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include "thrust/optional.h"

namespace raft_utils {

auto
get_current_device() {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDevice(&result));
    return result;
}

inline auto thread_counter = std::atomic<std::size_t>{};
inline auto
get_thread_id() {
    thread_local std::size_t id = ++thread_counter;
    return id;
}

// TODO(wphicks): Replace this with version from RAFT once merged
struct device_setter {
    device_setter(int new_device) : prev_device_{[]() { return get_current_device(); }()} {
        RAFT_CUDA_TRY(cudaSetDevice(new_device));
    }

    ~device_setter() {
        RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
    }

 private:
    int prev_device_;
};

inline auto raft_mutex = std::mutex{};

struct gpu_resources {
    gpu_resources(std::size_t streams_per_device = std::size_t{1})
        : streams_per_device_{streams_per_device}, stream_pools_{}, memory_resources_{}, upstream_mr_{} {
    }
    ~gpu_resources() {
        memory_resources_.clear();
    }

    void
    init(int device_id = get_current_device()) {
        auto lock = std::lock_guard{raft_mutex};
        auto stream_iter = stream_pools_.find(device_id);

        if (stream_iter == stream_pools_.end()) {
            auto scoped_device = device_setter{device_id};
            stream_pools_[device_id] = std::make_shared<rmm::cuda_stream_pool>(streams_per_device_);

            // Set up device memory pool for this device
            auto init_pool_size = thrust::optional<std::size_t>{};
            auto max_pool_size = thrust::optional<std::size_t>{};
            auto* env_str = getenv("KNOWHERE_GPU_MEM_POOL_SIZE");
            if (env_str != NULL) {
                auto init_pool_size_tmp = std::size_t{};
                auto max_pool_size_tmp = std::size_t{};
                auto stat = sscanf(env_str, "%zu;%zu", &init_pool_size_tmp, &max_pool_size_tmp);
                if (stat == 2) {
                    LOG_KNOWHERE_INFO_ << "Get Gpu Pool Size From env, init size: " << init_pool_size_tmp
                                       << " MB, max size: " << max_pool_size_tmp << " MB";
                    init_pool_size = init_pool_size_tmp << 20;
                    max_pool_size = max_pool_size_tmp << 20;
                } else {
                    LOG_KNOWHERE_WARNING_ << "please check env format";
                }
            }
            memory_resources_[device_id] =
                std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                    &upstream_mr_, init_pool_size, max_pool_size);
            rmm::mr::set_current_device_resource(memory_resources_[device_id].get());
        }
    }

    auto
    get_streams_per_device() const {
        return streams_per_device_;
    }

    auto
    get_stream_view(int device_id = get_current_device(), std::size_t thread_id = get_thread_id()) {
        return stream_pools_[device_id]->get_stream(thread_id % streams_per_device_);
    }

 private:
    std::size_t streams_per_device_;
    std::map<int, std::shared_ptr<rmm::cuda_stream_pool>> stream_pools_;
    std::map<int, std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>> memory_resources_;
    rmm::mr::cuda_memory_resource upstream_mr_;
};

inline auto&
get_gpu_resources(std::size_t streams_per_device = std::size_t{1}) {
    static auto resources = gpu_resources{streams_per_device};
    return resources;
}

inline void
init_gpu_resources(std::size_t streams_per_device = std::size_t{1}, int device_id = get_current_device()) {
    get_gpu_resources(streams_per_device).init(device_id);
}

inline auto&
get_raft_resources(int device_id = get_current_device()) {
    thread_local auto all_resources = std::map<int, std::unique_ptr<raft::device_resources>>{};

    auto iter = all_resources.find(device_id);
    if (iter == all_resources.end()) {
        auto scoped_device = device_setter{device_id};
        all_resources[device_id] = std::make_unique<raft::device_resources>(
            get_gpu_resources().get_stream_view(), nullptr, rmm::mr::get_current_device_resource());
    }
    return *all_resources[device_id];
}

};  // namespace raft_utils
