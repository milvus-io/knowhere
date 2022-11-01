// Copyright (C) 2019-2020 Zilliz. All rights reserved.
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

#include <memory>
#include <mutex>

#include "ctpl/ctpl-std.h"
namespace knowhere {

class ThreadPoolHolder {
 public:
    ThreadPoolHolder(const ThreadPoolHolder&) = delete;

    ~ThreadPoolHolder();

    ThreadPoolHolder&
    operator=(const ThreadPoolHolder&) = delete;

    ThreadPoolHolder(ThreadPoolHolder&&) noexcept = delete;

    ThreadPoolHolder&
    operator=(ThreadPoolHolder&&) noexcept = delete;

    void
    Init(uint32_t num_threads);

    std::shared_ptr<ctpl::thread_pool>
    GetThreadPool();

    static std::shared_ptr<ctpl::thread_pool>
    GetInstance();

 private:
    ThreadPoolHolder() = default;

    std::shared_ptr<ctpl::thread_pool> pool_;
    std::mutex mutex_;
};
}  // namespace knowhere
