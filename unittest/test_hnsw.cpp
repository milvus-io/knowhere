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
#include <random>

#include "knowhere/common/Config.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Init_with_default();
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        index_ = std::make_shared<knowhere::IndexHNSW>();
    }

 protected:
    knowhere::Config conf_;
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
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf_));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
        ASSERT_ANY_THROW(index_->Count());
        ASSERT_ANY_THROW(index_->Dim());
    }
    */

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    GET_TENSOR_DATA(base_dataset)

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf_);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);

    // case: k > nb
    const int64_t new_rows = 6;
    knowhere::SetDatasetRows(base_dataset, new_rows);
    index_->BuildAll(base_dataset, conf_);
    auto result3 = index_->Query(query_dataset, conf_, nullptr);
    auto res_ids = knowhere::GetDatasetIDs(result3);
    for (int64_t i = 0; i < nq; i++) {
        for (int64_t j = new_rows; j < k; j++) {
            ASSERT_EQ(res_ids[i * k + j], -1);
        }
    }
}

TEST_P(HNSWTest, HNSW_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }
        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);
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
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
}

TEST_P(HNSWTest, hnsw_slice) {
    knowhere::SetMetaSliceSize(conf_, knowhere::index_file_slice_size);
    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(knowhere::Config());
    index_->Load(binaryset);
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
}

TEST_P(HNSWTest, hnsw_range_search_l2) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::L2);
    knowhere::SetIndexParamHNSWK(conf_, 20);

    index_->BuildAll(base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::L2,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius * radius, golden_labels.data(), golden_lims.data(), false);
    };

    for (float radius: {4.1f, 4.2f, 4.3f}) {
        knowhere::SetMetaRadius(conf_, radius);
        test_range_search_l2(radius, nullptr);
        test_range_search_l2(radius, *bitset);
    }
}

TEST_P(HNSWTest, hnsw_range_search_ip) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::IP);
    knowhere::SetIndexParamHNSWK(conf_, 20);

    index_->BuildAll(base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF<CMax<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::IP,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMax<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), false);
    };

    for (float radius: {42.0f, 43.0f, 44.0f}) {
        knowhere::SetMetaRadius(conf_, radius);
        test_range_search_ip(radius, nullptr);
        test_range_search_ip(radius, *bitset);
    }
}
