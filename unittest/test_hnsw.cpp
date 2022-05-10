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
#include <iostream>
#include <random>

#include "knowhere/common/Config.h"
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Generate(64, 10000, 10);  // dim = 64, nb = 10000, nq = 10
        index_ = std::make_shared<knowhere::IndexHNSW>();
        conf = knowhere::Config{
            {knowhere::meta::DIM, 64},          {knowhere::meta::TOPK, 10},
            {knowhere::indexparam::HNSW_M, 16}, {knowhere::indexparam::EFCONSTRUCTION, 200},
            {knowhere::indexparam::EF, 200},    {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        };
    }

 protected:
    knowhere::Config conf;
    knowhere::IndexMode index_mode_ = knowhere::IndexMode::MODE_CPU;
    knowhere::IndexType index_type_ = knowhere::IndexEnum::INDEX_HNSW;
    std::shared_ptr<knowhere::IndexHNSW> index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, HNSWTest, Values("HNSW"));

TEST_P(HNSWTest, HNSW_basic) {
    assert(!xb.empty());

    // null faiss index
    /*
    {
        ASSERT_ANY_THROW(index_->Serialize());
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf));
        ASSERT_ANY_THROW(index_->Count());
        ASSERT_ANY_THROW(index_->Dim());
    }
    */

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = knowhere::GetDatasetDim(base_dataset);
    int64_t rows = knowhere::GetDatasetRows(base_dataset);
    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf, index_type_, index_mode_));

    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);

    // case: k > nb
    const int64_t new_rows = 6;
    knowhere::SetDatasetRows(base_dataset, new_rows);
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    auto result2 = index_->Query(query_dataset, conf, nullptr);
    auto res_ids = knowhere::GetDatasetIDs(result2);
    for (int64_t i = 0; i < nq; i++) {
        for (int64_t j = new_rows; j < k; j++) {
            ASSERT_EQ(res_ids[i * k + j], -1);
        }
    }
}

TEST_P(HNSWTest, HNSW_delete) {
    assert(!xb.empty());

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = knowhere::GetDatasetDim(base_dataset);
    int64_t rows = knowhere::GetDatasetRows(base_dataset);
    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf, *bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

/*
TEST_P(HNSWTest, HNSW_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    {
        index_->Train(base_dataset, conf);
        index_->AddWithoutIds(base_dataset, conf);
        auto binaryset = index_->Serialize();
        auto bin = binaryset.GetByName("HNSW");

        std::string filename = temp_path("/tmp/HNSW_test_serialize.bin");
        auto load_data = new uint8_t[bin->size];
        serialize(filename, bin, load_data);

        binaryset.clear();
        std::shared_ptr<uint8_t[]> data(load_data);
        binaryset.Append("HNSW", data, bin->size);

        index_->Load(binaryset);
        EXPECT_EQ(index_->Count(), nb);
        EXPECT_EQ(index_->Dim(), dim);
        auto result = index_->Query(query_dataset, conf);
        AssertAnns(result, nq, conf[knowhere::meta::TOPK]);
    }
}*/
