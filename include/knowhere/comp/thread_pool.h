// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <omp.h>

#include <memory>
#include <utility>

#include "ctpl/ctpl-std.h"
#include "knowhere/log.h"

namespace knowhere {

class ThreadPool {
 public:
    explicit ThreadPool(uint32_t num_threads) {
        pool_ = std::make_unique<ctpl::thread_pool>(num_threads);
    }

    ThreadPool(const ThreadPool&) = delete;

    ThreadPool&
    operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) noexcept = delete;

    ThreadPool&
    operator=(ThreadPool&&) noexcept = delete;

    template <typename Func, typename... Args>
    auto
    push(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        return pool_->push([func = std::forward<Func>(func), &args...](int /* unused */) mutable {
            return func(std::forward<Args>(args)...);
        });
    }

    uint32_t
    size() const noexcept {
        return pool_->size();
    }

    /**
     * @brief Set the threads number to the global thread pool of knowhere
     *
     * @param num_threads
     */
    static void
    InitGlobalThreadPool(uint32_t num_threads) {
        if (num_threads <= 0) {
            LOG_KNOWHERE_ERROR_ << "num_threads should be bigger than 0";
            return;
        }

        if (global_thread_pool_size_ == 0) {
            std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
            if (global_thread_pool_size_ == 0) {
                global_thread_pool_size_ = num_threads;
                return;
            }
        }
        LOG_KNOWHERE_WARNING_ << "Global ThreadPool has already been initialized with threads num: "
                              << global_thread_pool_size_;
    }

    /**
     * @brief Get the global thread pool of knowhere.
     *
     * @return ThreadPool&
     */
    static std::shared_ptr<ThreadPool>
    GetGlobalThreadPool() {
        if (global_thread_pool_size_ == 0) {
            std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
            if (global_thread_pool_size_ == 0) {
                global_thread_pool_size_ = std::thread::hardware_concurrency();
                LOG_KNOWHERE_WARNING_ << "Global ThreadPool has not been initialized yet, init it with threads num: "
                                      << global_thread_pool_size_;
            }
        }
        static auto pool = std::make_shared<ThreadPool>(global_thread_pool_size_);
        return pool;
    }

    class ScopedOmpSetter {
        int omp_before;

     public:
        explicit ScopedOmpSetter(int num_threads = 1) : omp_before(omp_get_num_threads()) {
            omp_set_num_threads(num_threads);
        }
        ~ScopedOmpSetter() {
            omp_set_num_threads(omp_before);
        }
    };

 private:
    std::unique_ptr<ctpl::thread_pool> pool_;
    inline static uint32_t global_thread_pool_size_ = 0;
    inline static std::mutex global_thread_pool_mutex_;
};
}  // namespace knowhere
