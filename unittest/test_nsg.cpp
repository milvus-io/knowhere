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
#include <memory>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/index/vector_offset_index/IndexNSG_NM.h"
#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

#include "knowhere/common/Timer.h"
#include "knowhere/index/vector_index/impl/nsg/NSGIO.h"

#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr int64_t DEVICE_GPU0 = 0;

class NSGInterfaceTest : public DataGen, public ::testing::Test {
 protected:
    void
    SetUp() override {
#ifdef KNOWHERE_GPU_VERSION
        int64_t MB = 1024 * 1024;
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICE_GPU0, MB * 200, MB * 600, 1);
#endif
        int nsg_dim = 256;
        Generate(nsg_dim, nb, nq);
        index_ = std::make_shared<knowhere::NSG_NM>();

        train_conf = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::meta::DIM, 256},
            {knowhere::indexparam::NLIST, 163},
            {knowhere::indexparam::NPROBE, 8},
            {knowhere::indexparam::KNNG, 20},
            {knowhere::indexparam::SEARCH_LENGTH, 40},
            {knowhere::indexparam::OUT_DEGREE, 30},
            {knowhere::indexparam::CANDIDATE, 100},
        };

        search_conf = knowhere::Config{
            {knowhere::meta::SLICE_SIZE, knowhere::index_file_slice_size},
            {knowhere::meta::TOPK, k},
            {knowhere::IndexParams::search_length, 30},
        };
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

 protected:
    std::shared_ptr<knowhere::NSG_NM> index_;
    knowhere::Config train_conf;
    knowhere::Config search_conf;
};

TEST_F(NSGInterfaceTest, basic_test) {
    assert(!xb.empty());
    // untrained index
    {
        ASSERT_ANY_THROW(index_->Serialize(search_conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, search_conf, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, search_conf));
    }

    knowhere::SetMetaDeviceID(train_conf, -1);
    index_->BuildAll(base_dataset, train_conf);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(search_conf);

    int64_t dim = knowhere::GetDatasetDim(base_dataset);
    int64_t rows = knowhere::GetDatasetRows(base_dataset);
    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result_1 = index_->Query(query_dataset, search_conf, nullptr);
    AssertAnns(result_1, nq, k);

    /* test NSG GPU train */
    auto new_index = std::make_shared<knowhere::NSG_NM>(DEVICE_GPU0);
    knowhere::SetMetaDeviceID(train_conf, DEVICE_GPU0);
    new_index->BuildAll(base_dataset, train_conf);

    // Serialize and Load before Query
    bs = new_index->Serialize(search_conf);

    dim = knowhere::GetDatasetDim(base_dataset);
    rows = knowhere::GetDatasetRows(base_dataset);
    raw_data = knowhere::GetDatasetTensor(base_dataset);
    bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    new_index->Load(bs);

    auto result_2 = new_index->Query(query_dataset, search_conf, nullptr);
    AssertAnns(result_2, nq, k);

    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
}

TEST_F(NSGInterfaceTest, compare_test) {
    knowhere::impl::DistanceL2 distanceL2;
    knowhere::impl::DistanceIP distanceIP;

    knowhere::TimeRecorder tc("Compare");
    for (int i = 0; i < 1000; ++i) {
        distanceL2.Compare(xb.data(), xq.data(), 256);
    }
    tc.RecordSection("L2");
    for (int i = 0; i < 1000; ++i) {
        distanceIP.Compare(xb.data(), xq.data(), 256);
    }
    tc.RecordSection("IP");
}

TEST_F(NSGInterfaceTest, delete_test) {
    assert(!xb.empty());

    knowhere::SetMetaDeviceID(train_conf, DEVICE_GPU0);
    index_->BuildAll(base_dataset, train_conf);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(search_conf);

    int64_t dim = knowhere::GetDatasetDim(base_dataset);
    int64_t rows = knowhere::GetDatasetRows(base_dataset);
    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result = index_->Query(query_dataset, search_conf, nullptr);
    AssertAnns(result, nq, k);
    auto I_before = GetDatasetIDs(result);

    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    bs = index_->Serialize(search_conf);

    dim = knowhere::GetDatasetDim(base_dataset);
    rows = knowhere::GetDatasetRows(base_dataset);
    raw_data = knowhere::GetDatasetTensor(base_dataset);
    bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    // search xq with delete
    auto result_after = index_->Query(query_dataset, search_conf, *bitset);

    AssertAnns(result_after, nq, k, CheckMode::CHECK_NOT_EQUAL);
    auto I_after = GetDatasetIDs(result_after);

    // First vector deleted
    for (int i = 0; i < nq; i++) {
        ASSERT_NE(I_before[i * k], I_after[i * k]);
    }
}

TEST_F(NSGInterfaceTest, slice_test) {
    assert(!xb.empty());
    // untrained index
    {
        ASSERT_ANY_THROW(index_->Serialize(search_conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, search_conf, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, search_conf));
    }

    knowhere::SetMetaDeviceID(train_conf, -1);
    index_->BuildAll(base_dataset, train_conf);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(search_conf);

    int64_t dim = knowhere::GetDatasetDim(base_dataset);
    int64_t rows = knowhere::GetDatasetRows(base_dataset);
    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result = index_->Query(query_dataset, search_conf, nullptr);
    AssertAnns(result, nq, k);

    /* test NSG GPU train */
    auto new_index_1 = std::make_shared<knowhere::NSG_NM>(DEVICE_GPU0);
    knowhere::SetMetaDeviceID(train_conf, DEVICE_GPU0);
    new_index_1->BuildAll(base_dataset, train_conf);

    // Serialize and Load before Query
    bs = new_index_1->Serialize(search_conf);

    dim = knowhere::GetDatasetDim(base_dataset);
    rows = knowhere::GetDatasetRows(base_dataset);
    raw_data = knowhere::GetDatasetTensor(base_dataset);
    bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    new_index_1->Load(bs);

    auto new_result_1 = new_index_1->Query(query_dataset, search_conf, nullptr);
    AssertAnns(new_result_1, nq, k);

    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
}
