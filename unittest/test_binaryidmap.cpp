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

#include <faiss/utils/BinaryDistance.h>
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexBinaryIDMAP.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
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
        index_ = std::make_shared<knowhere::BinaryIDMAP>();
    }

    void
    TearDown() override{};

    template <class C>
    void RunRangeSearchBF(
        std::vector<int64_t>& golden_labels,
        float radius,
        binary_float_func_ptr func,
        const faiss::BitsetView bitset) {
        for (auto i = 0; i < nq; ++i) {
            const uint8_t* pq = reinterpret_cast<uint8_t*>(xq_bin.data()) + i * dim / 8;
            for (auto j = 0; j < nb; ++j) {
                if (bitset.empty() || !bitset.test(j)) {
                    const uint8_t* pb = reinterpret_cast<uint8_t*>(xb_bin.data()) + j * dim / 8;
                    auto dist = func(pq, pb, dim/8);
                    if (C::cmp(dist, radius)) {
                        golden_labels.push_back(j);
                    }
                }
            }
        }
    }

    template <class C>
    void CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const float radius,
        const std::vector<int64_t>& golden_labels) {
        auto lims = result->Get<size_t*>(knowhere::meta::LIMS);
        auto ids = result->Get<int64_t*>(knowhere::meta::IDS);
        auto distances = result->Get<float*>(knowhere::meta::DISTANCE);
        ASSERT_EQ(lims[nq], golden_labels.size());
        for (auto i = 0; i < lims[nq]; ++i) {
            ASSERT_EQ(golden_labels[i], ids[i]);
            ASSERT_TRUE(C::cmp(distances[i], radius));
        }
    }

 protected:
    knowhere::BinaryIDMAPPtr index_ = nullptr;
    knowhere::IndexMode index_mode_;
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

    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::HAMMING},
    };

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf));
    }

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_TRUE(index_->GetRawVectors() != nullptr);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

    auto binaryset = index_->Serialize(conf);
    auto new_index = std::make_shared<knowhere::BinaryIDMAP>();
    new_index->Load(binaryset);
    auto result2 = new_index->Query(query_dataset, conf, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(re_result, nq, k);

    std::shared_ptr<uint8_t[]> data(new uint8_t[nb/8]);
    for (int64_t i = 0; i < nq; ++i) {
        set_bit(data.get(), i);
    }
    auto bitset = faiss::BitsetView(data.get(), nb);

    auto result_bs_1 = index_->Query(query_dataset, conf, bitset);
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

    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::HAMMING},
    };

    // serialize index
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result1, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf);
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
    auto result2 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(result2, nq, k);

    // query with bitset
    std::shared_ptr<uint8_t[]> bs_data(new uint8_t[nb/8]);
    for (int64_t i = 0; i < nq; ++i) {
        set_bit(bs_data.get(), i);
    }
    auto bitset = faiss::BitsetView(bs_data.get(), nb);
    auto result_bs_1 = index_->Query(query_dataset, conf, bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(BinaryIDMAPTest, binaryidmap_slice) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::HAMMING},
        {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
    };

    // serialize index
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result1, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result2 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(result2, nq, k);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_hamming) {
    int hamming_radius = 20;
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::RADIUS, hamming_radius},
        {knowhere::Metric::TYPE, knowhere::Metric::HAMMING},
    };

    // hamming
    auto hamming_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        return faiss::xor_popcnt(pa, pb, code_size);
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    std::shared_ptr<uint8_t[]> data(new uint8_t[nb / 8]);
    for (int64_t i = 0; i < nb; i += 2) {
        set_bit(data.get(), i);
    }
    auto bitset = faiss::BitsetView(data.get(), nb);

    // test without bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, hamming_radius, hamming_dis, nullptr);

        auto result = index_->QueryByRange(qd, conf, nullptr);
        CheckRangeSearchResult<CMin<float>>(result, hamming_radius, golden_labels);
    }

    // test with bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, hamming_radius, hamming_dis, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult<CMin<float>>(result, hamming_radius, golden_labels);
    }
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_jaccard) {
    float jaccard_radius = 0.5;
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::RADIUS, jaccard_radius},
        {knowhere::Metric::TYPE, knowhere::Metric::JACCARD},
    };

    auto jaccard_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        return 1.0 - (double)and_value / or_value;
    };

    // serialize index
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    std::shared_ptr<uint8_t[]> data(new uint8_t[nb / 8]);
    for (int64_t i = 0; i < nb; i += 2) {
        set_bit(data.get(), i);
    }
    auto bitset = faiss::BitsetView(data.get(), nb);

    // test without bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, jaccard_radius, jaccard_dis, nullptr);

        auto result = index_->QueryByRange(qd, conf, nullptr);
        CheckRangeSearchResult<CMin<float>>(result, jaccard_radius, golden_labels);
    }

    // test with bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, jaccard_radius, jaccard_dis, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult<CMin<float>>(result, jaccard_radius, golden_labels);
    }
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_tanimoto) {
    float tanimoto_radius = 1.0;
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::RADIUS, tanimoto_radius},
        {knowhere::Metric::TYPE, knowhere::Metric::TANIMOTO},
    };

    auto tanimoto_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        auto v = 1.0 - (double)and_value / or_value;
        return (-log2(1 - v));
    };


    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    std::shared_ptr<uint8_t[]> data(new uint8_t[nb / 8]);
    for (int64_t i = 0; i < nb; i += 2) {
        set_bit(data.get(), i);
    }
    auto bitset = faiss::BitsetView(data.get(), nb);

    // test without bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, tanimoto_radius, tanimoto_dis, nullptr);

        auto result = index_->QueryByRange(qd, conf, nullptr);
        CheckRangeSearchResult<CMin<float>>(result, tanimoto_radius, golden_labels);
    }

    // test with bitset
    {
        std::vector<int64_t> golden_labels;
        RunRangeSearchBF<CMin<float>>(golden_labels, tanimoto_radius, tanimoto_dis, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult<CMin<float>>(result, tanimoto_radius, golden_labels);
    }
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_superstructure) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::RADIUS, radius},
        {knowhere::Metric::TYPE, knowhere::Metric::SUPERSTRUCTURE},
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf, nullptr));
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_substructure) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::RADIUS, radius},
        {knowhere::Metric::TYPE, knowhere::Metric::SUBSTRUCTURE},
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf, nullptr));
}
