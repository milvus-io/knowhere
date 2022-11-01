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

#include "knowhere/common/ThreadPoolHolder.h"

#include "knowhere/common/Log.h"

namespace knowhere {

void
ThreadPoolHolder::Init(uint32_t num_threads) {
    if (!pool_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_) {
            pool_ = std::make_shared<ctpl::thread_pool>(num_threads);
        }
    }
    LOG_KNOWHERE_WARNING_ << "ThreadPoolHolder has already been initilaized with threads num: " << pool_->size();
}

std::shared_ptr<ctpl::thread_pool>
ThreadPoolHolder::GetThreadPool() {
    if (!pool_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_) {
            LOG_KNOWHERE_WARNING_ << "ThreadPoolHolder is not inialized, init it with threads num: "
                                  << std::thread::hardware_concurrency();
            pool_ = std::make_shared<ctpl::thread_pool>(std::thread::hardware_concurrency());
        }
    }
    return pool_;
}

ThreadPoolHolder&
ThreadPoolHolder::GetInstance() {
    static ThreadPoolHolder holder;
    return holder;
}
}  // namespace knowhere