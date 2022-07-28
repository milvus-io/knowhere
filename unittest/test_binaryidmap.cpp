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
#include "knowhere/index/vector_index/IndexBinaryIDMAP.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

typedef float (*binary_float_func_ptr)(const uint8_t*, const uint8_t*, const size_t);

class BinaryIDMAPTest : public DataGen,
                        public TestWithParam<knowhere::IndexMode> {
 protected:
    void
    SetUp() override {
        Init_with_default(true);
        index_mode_ = GetParam();
        index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
    }

    void
    TearDown() override{};

 protected:
    knowhere::Config conf_;
    knowhere::IndexMode index_mode_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    BinaryIDMAPParameters,
    BinaryIDMAPTest,
    Values(
#ifdef KNOWHERE_GPU_VERSION
        knowhere::IndexMode::MODE_GPU,
#endif
        knowhere::IndexMode::MODE_CPU));

TEST_P(BinaryIDMAPTest, binaryidmap_basic) {
    ASSERT_TRUE(!xb_bin.empty());

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf_));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
    }

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);
    ASSERT_TRUE(std::static_pointer_cast<knowhere::BinaryIDMAP>(index_)->GetRawVectors() != nullptr);

    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertBinVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result, nq, k);

    auto binaryset = index_->Serialize(conf_);
    auto new_index = std::make_shared<knowhere::BinaryIDMAP>();
    new_index->Load(binaryset);
    auto result2 = new_index->Query(query_dataset, conf_, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(re_result, nq, k);

    auto result_bs_1 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);

    // auto result4 = index_->SearchById(id_dataset, conf);
    // AssertAneq(result4, nq, k);
}

TEST_P(BinaryIDMAPTest, binaryidmap_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }
        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result1, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf_);
    auto bin = binaryset.GetByName("BinaryIVF");

    std::string filename = temp_path("/tmp/bianryidmap_test_serialize.bin");
    auto load_data = new uint8_t[bin->size];
    serialize(filename, bin, load_data);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> data(load_data);
    binaryset.Append("BinaryIVF", data, bin->size);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result2 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(result2, nq, k);

    // query with bitset
    auto result_bs_1 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(BinaryIDMAPTest, binaryidmap_slice) {
    knowhere::SetMetaSliceSize(conf_, knowhere::index_file_slice_size);
    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result1, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf_);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result2 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(result2, nq, k);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_hamming) {
    int hamming_radius = 50;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::HAMMING);
    knowhere::SetMetaRadius(conf_, hamming_radius);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_hamming = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::HAMMING,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_hamming(hamming_radius, nullptr);
    test_range_search_hamming(hamming_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_jaccard) {
    float jaccard_radius = 0.5;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::JACCARD);
    knowhere::SetMetaRadius(conf_, jaccard_radius);

    // serialize index
    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_jaccard = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::JACCARD,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_jaccard(jaccard_radius, nullptr);
    test_range_search_jaccard(jaccard_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_tanimoto) {
    float tanimoto_radius = 1.0;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::TANIMOTO);
    knowhere::SetMetaRadius(conf_, tanimoto_radius);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_tanimoto = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::TANIMOTO,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_tanimoto(tanimoto_radius, nullptr);
    test_range_search_tanimoto(tanimoto_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_superstructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUPERSTRUCTURE);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf_, nullptr));
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_substructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUBSTRUCTURE);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf_, nullptr));
}
