// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>

#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/AsyncIndex.h"
#include "unittest/Helper.h"
#include "unittest/ThreadChecker.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {
constexpr int32_t kQuerySum = 10;
}

class AsyncIndexTest : public DataGen, public TestWithParam<knowhere::IndexType> {
 protected:
    void
    SetUp() override {
        knowhere::IndexType index_type_ = GetParam();
        if (index_type_ == knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP ||
            index_type_ == knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
            Init_with_default(true);
        } else {
            Init_with_default();
        }
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        index_ = std::make_unique<knowhere::AsyncIndex>(index_type_);
    }

    void
    TearDown() override {
    }

 protected:
    knowhere::Config conf_;
    std::unique_ptr<knowhere::AsyncIndex> index_;
};

INSTANTIATE_TEST_CASE_P(AsyncIndexParameters, AsyncIndexTest,
                        Values(knowhere::IndexEnum::INDEX_ANNOY, knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP,
                               knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, knowhere::IndexEnum::INDEX_FAISS_IDMAP,
                               knowhere::IndexEnum::INDEX_HNSW, knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
                               knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::IndexEnum::INDEX_FAISS_IVFSQ8));

TEST_P(AsyncIndexTest, async_query_thread_num) {
    int pid = getpid();
    int32_t num_threads_before_build = knowhere::threadchecker::GetThreadNum(pid);
    index_->BuildAll(base_dataset, conf_);
    int32_t num_threads_after_build = knowhere::threadchecker::GetThreadNum(pid);
    EXPECT_GE(knowhere::threadchecker::GetBuildOmpThread(conf_),
              num_threads_after_build - num_threads_before_build + 1);
    for (int i = 0; i < kQuerySum; i++) {
        index_->QueryAsync(query_dataset, conf_, nullptr);
    }
    int32_t num_threads_after_query = knowhere::threadchecker::GetThreadNum(pid);
    int32_t expected_num_threads =
        knowhere::threadchecker::GetQueryOmpThread(conf_) * kQuerySum + num_threads_after_build;
    EXPECT_GE(expected_num_threads, num_threads_after_query);
    for (int i = 0; i < kQuerySum; i++) {
        auto result = index_->Sync();
    }
    int32_t num_threads_end_query = knowhere::threadchecker::GetThreadNum(pid);
    expected_num_threads -= kQuerySum;
    EXPECT_GE(expected_num_threads, num_threads_end_query);
}
