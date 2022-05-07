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
#include <knowhere/index/vector_index/IndexRHNSWSQ.h>
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include <iostream>
#include <random>
#include "knowhere/common/Exception.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class RHNSWSQ8Test : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        IndexType = GetParam();
        std::cout << "IndexType from GetParam() is: " << IndexType << std::endl;
        Generate(64, 10000, 10);  // dim = 64, nb = 10000, nq = 10
                                  //        Generate(2, 10, 2);  // dim = 64, nb = 10000, nq = 10
        index_ = std::make_shared<knowhere::IndexRHNSWSQ>();
        conf = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::meta::DIM, 64},
            {knowhere::meta::TOPK, 10},
            {knowhere::indexparam::M, 16},
            {knowhere::indexparam::EFCONSTRUCTION, 200},
            {knowhere::indexparam::EF, 200},
            {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
        };
    }

 protected:
    knowhere::Config conf;
    std::shared_ptr<knowhere::IndexRHNSWSQ> index_ = nullptr;
    std::string IndexType;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, RHNSWSQ8Test, Values("RHNSWSQ8"));

TEST_P(RHNSWSQ8Test, HNSW_basic) {
    assert(!xb.empty());

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf);
    auto result1 = index_->Query(query_dataset, conf, nullptr);
    //    AssertAnns(result1, nq, k);

    auto tmp_index = std::make_shared<knowhere::IndexRHNSWSQ>();

    tmp_index->Load(bs);

    auto result2 = tmp_index->Query(query_dataset, conf, nullptr);
    //    AssertAnns(result2, nq, k);
}

TEST_P(RHNSWSQ8Test, HNSW_delete) {
    assert(!xb.empty());

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    //    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf, *bitset);
    //    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);

    /*
     * delete result checked by eyes
    auto ids1 = result1->Get<int64_t*>(knowhere::meta::IDS);
    auto ids2 = result2->Get<int64_t*>(knowhere::meta::IDS);
    std::cout << std::endl;
    for (int i = 0; i < nq; ++ i) {
        std::cout << "ids1: ";
        for (int j = 0; j < k; ++ j) {
            std::cout << *(ids1 + i * k + j) << " ";
        }
        std::cout << "ids2: ";
        for (int j = 0; j < k; ++ j) {
            std::cout << *(ids2 + i * k + j) << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < std::min(5, k>>1); ++ j) {
            ASSERT_EQ(*(ids1 + i * k + j + 1), *(ids2 + i * k + j));
        }
    }
    */
}

TEST_P(RHNSWSQ8Test, HNSW_serialize) {
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
        auto binaryset = index_->Serialize(conf);
        auto bin_idx = binaryset.GetByName(std::string(index_->index_type()) + "_Index");
        auto bin_dat = binaryset.GetByName(QUANTIZATION_DATA);

        std::string filename_idx = temp_path("/tmp/RHNSWSQ_test_serialize_idx.bin");
        std::string filename_dat = temp_path("/tmp/RHNSWSQ_test_serialize_dat.bin");
        auto load_idx = new uint8_t[bin_idx->size];
        auto load_dat = new uint8_t[bin_dat->size];
        serialize(filename_idx, bin_idx, load_idx);
        serialize(filename_dat, bin_dat, load_dat);

        binaryset.clear();
        auto new_idx = std::make_shared<knowhere::IndexRHNSWSQ>();
        std::shared_ptr<uint8_t[]> dat(load_dat);
        std::shared_ptr<uint8_t[]> idx(load_idx);
        binaryset.Append(std::string(new_idx->index_type()) + "_Index", idx, bin_idx->size);
        binaryset.Append(QUANTIZATION_DATA, dat, bin_dat->size);

        new_idx->Load(binaryset);
        EXPECT_EQ(new_idx->Count(), nb);
        EXPECT_EQ(new_idx->Dim(), dim);
        auto result = new_idx->Query(query_dataset, conf, nullptr);
        // AssertAnns(result, nq, conf[knowhere::meta::TOPK]);
    }
}

TEST_P(RHNSWSQ8Test, HNSW_slice) {
    {
        index_->Train(base_dataset, conf);
        index_->AddWithoutIds(base_dataset, conf);
        auto binaryset = index_->Serialize(conf);
        auto new_idx = std::make_shared<knowhere::IndexRHNSWSQ>();
        new_idx->Load(binaryset);
        EXPECT_EQ(new_idx->Count(), nb);
        EXPECT_EQ(new_idx->Dim(), dim);
        auto result = new_idx->Query(query_dataset, conf, nullptr);
        // AssertAnns(result, nq, conf[knowhere::meta::TOPK]);
    }
}
