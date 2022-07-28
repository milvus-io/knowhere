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
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "unittest/Helper.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class RHNSWTest : public DataGen,
                  public TestWithParam<std::tuple<knowhere::IndexType, std::string>> {
 protected:
    void
    SetUp() override {
        Init_with_default();
        std::tie(index_type_, meta_key_) = GetParam();
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
    }

 protected:
    knowhere::Config conf_;
    knowhere::IndexMode index_mode_ = knowhere::IndexMode::MODE_CPU;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
    std::string meta_key_;
};

INSTANTIATE_TEST_CASE_P(
    HNSWParameters,
    RHNSWTest,
    Values(
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWFlat, "META"),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWPQ, "QUANTIZATION_DATA"),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWSQ, "QUANTIZATION_DATA")
        ));

TEST_P(RHNSWTest, RHNSW_basic) {
    assert(!xb.empty());

    // null faiss index
    ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    if (index_type_ != knowhere::IndexEnum::INDEX_RHNSWPQ) {
        AssertAnns(result, nq, k);
    }

    auto result_bs = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(RHNSWTest, RHNSW_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);

    std::string idx_name = std::string(index_type_) + "_Index";
    std::string meta_name = meta_key_;
    auto bin_idx = binaryset.GetByName(idx_name);
    auto bin_meta = binaryset.GetByName(meta_name);

    std::string filename_idx = temp_path("/tmp/RHNSW_test_serialize_idx.bin");
    std::string filename_meta = temp_path("/tmp/RHNSW_test_serialize_meta.bin");
    auto load_idx_data = new uint8_t[bin_idx->size];
    auto load_meta_data = new uint8_t[bin_meta->size];
    serialize(filename_idx, bin_idx, load_idx_data);
    serialize(filename_meta, bin_meta, load_meta_data);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> meta_data(load_meta_data);
    std::shared_ptr<uint8_t[]> idx_data(load_idx_data);
    binaryset.Append(idx_name, idx_data, bin_idx->size);
    binaryset.Append(meta_name, meta_data, bin_meta->size);

    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * nb * sizeof(float);
    binaryset.Append(RAW_DATA, bptr);

    auto new_index = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_);
    new_index->Load(binaryset);
    EXPECT_EQ(new_index->Count(), nb);
    EXPECT_EQ(new_index->Dim(), dim);
    auto result = new_index->Query(query_dataset, conf_, nullptr);
    if (index_type_ != knowhere::IndexEnum::INDEX_RHNSWPQ) {
        AssertAnns(result, nq, k);
    }
}

TEST_P(RHNSWTest, RHNSW_slice) {
    knowhere::SetMetaSliceSize(conf_, knowhere::index_file_slice_size);
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);

    auto raw_data = knowhere::GetDatasetTensor(base_dataset);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * nb * sizeof(float);
    binaryset.Append(RAW_DATA, bptr);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    if (index_type_ != knowhere::IndexEnum::INDEX_RHNSWPQ) {
        AssertAnns(result, nq, k);
    }
}
