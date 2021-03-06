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

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexAnnoy.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class AnnoyTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Generate(128, 10000, 10);
        index_ = std::make_shared<knowhere::IndexAnnoy>();
        conf = knowhere::Config{
            {knowhere::meta::SLICE_SIZE, knowhere::index_file_slice_size},
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, 10},
            {knowhere::indexparam::N_TREES, 4},
            {knowhere::indexparam::SEARCH_K, 100},
        };
    }

 protected:
    knowhere::Config conf;
    knowhere::IndexMode index_mode_ = knowhere::IndexMode::MODE_CPU;
    knowhere::IndexType index_type_ = knowhere::IndexEnum::INDEX_ANNOY;
    std::shared_ptr<knowhere::IndexAnnoy> index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(AnnoyParameters, AnnoyTest, Values("Annoy"));

TEST_P(AnnoyTest, annoy_basic) {
    assert(!xb.empty());

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Train(base_dataset, conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf, nullptr));
        ASSERT_ANY_THROW(index_->Serialize(conf));
        ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf));
        ASSERT_ANY_THROW(index_->Count());
        ASSERT_ANY_THROW(index_->Dim());
    }

    index_->BuildAll(base_dataset, conf);  // Train + AddWithoutIds
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    auto result = index_->GetVectorById(id_dataset, conf);
    AssertVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);
}

TEST_P(AnnoyTest, annoy_delete) {
    assert(!xb.empty());

    index_->BuildAll(base_dataset, conf);  // Train + AddWithoutIds
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf, *bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(AnnoyTest, annoy_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            // write and flush
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    // serialize index
    index_->BuildAll(base_dataset, conf);
    auto binaryset = index_->Serialize(knowhere::Config());

    auto bin_data = binaryset.GetByName("annoy_index_data");
    std::string filename1 = temp_path("/tmp/annoy_test_data_serialize.bin");
    auto load_data1 = new uint8_t[bin_data->size];
    serialize(filename1, bin_data, load_data1);

    auto bin_metric_type = binaryset.GetByName("annoy_metric_type");
    std::string filename2 = temp_path("/tmp/annoy_test_metric_type_serialize.bin");
    auto load_data2 = new uint8_t[bin_metric_type->size];
    serialize(filename2, bin_metric_type, load_data2);

    auto bin_dim = binaryset.GetByName("annoy_dim");
    std::string filename3 = temp_path("/tmp/annoy_test_dim_serialize.bin");
    auto load_data3 = new uint8_t[bin_dim->size];
    serialize(filename3, bin_dim, load_data3);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> index_data(load_data1);
    binaryset.Append("annoy_index_data", index_data, bin_data->size);

    std::shared_ptr<uint8_t[]> metric_data(load_data2);
    binaryset.Append("annoy_metric_type", metric_data, bin_metric_type->size);

    std::shared_ptr<uint8_t[]> dim_data(load_data3);
    binaryset.Append("annoy_dim", dim_data, bin_dim->size);

    index_->Load(binaryset);
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf));
}

TEST_P(AnnoyTest, annoy_slice) {
    // serialize index
    index_->BuildAll(base_dataset, conf);
    auto binaryset = index_->Serialize(knowhere::Config());
    index_->Load(binaryset);
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf));
}
