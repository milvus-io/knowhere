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
#include <thread>

#include "knowhere/common/Exception.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#include "unittest/Helper.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class IVFHNSWTest : public DataGen,
                    public TestWithParam<::std::tuple<knowhere::IndexType, knowhere::IndexMode>> {
 protected:
    void
    SetUp() override {
        Init_with_default();
        std::tie(index_type_, index_mode_) = GetParam();
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

 protected:
    knowhere::IndexType index_type_;
    knowhere::IndexMode index_mode_;
    knowhere::Config conf_;
    knowhere::VecIndexPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    IVFParameters,
    IVFHNSWTest,
    Values(std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFHNSW, knowhere::IndexMode::MODE_CPU)));

TEST_P(IVFHNSWTest, ivfhnsw_basic_cpu) {
    assert(!xb.empty());

    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }

    // null faiss index
    ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);
}

TEST_P(IVFHNSWTest, ivfhnsw_slice) {
    knowhere::SetMetaSliceSize(conf_, knowhere::index_file_slice_size);
    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);
    // load index
    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
}
