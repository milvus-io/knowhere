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

        conf_ = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, knowhere::metric::HAMMING},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, k},
            {knowhere::meta::RADIUS, radius},
            {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
        };
        index_mode_ = GetParam();
        index_ = std::make_shared<knowhere::BinaryIDMAP>();
    }

    void
    TearDown() override{};

    template <class C>
    void RunRangeSearchBF(
        std::vector<int64_t>& golden_labels,
        std::vector<float>& golden_distances,
        std::vector<size_t>& golden_lims,
        float radius,
        binary_float_func_ptr func,
        const faiss::BitsetView bitset) {

        golden_lims.push_back(0);
        for (auto i = 0; i < nq; ++i) {
            const uint8_t* pq = reinterpret_cast<uint8_t*>(xq_bin.data()) + i * dim / 8;
            for (auto j = 0; j < nb; ++j) {
                if (bitset.empty() || !bitset.test(j)) {
                    const uint8_t* pb = reinterpret_cast<uint8_t*>(xb_bin.data()) + j * dim / 8;
                    auto dist = func(pq, pb, dim/8);
                    if (C::cmp(dist, radius)) {
                        golden_labels.push_back(j);
                        golden_distances.push_back(dist);
                    }
                }
            }
            golden_lims.push_back(golden_labels.size());
        }
    }

    template <class C>
    void CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const float radius,
        const std::vector<int64_t>& golden_labels,
        const std::vector<float>& golden_distances,
        const std::vector<size_t>& golden_lims) {

        auto lims = result->Get<size_t*>(knowhere::meta::LIMS);
        auto ids = result->Get<int64_t*>(knowhere::meta::IDS);
        auto distances = result->Get<float*>(knowhere::meta::DISTANCE);

        for (int64_t i = 0; i < nq; i++) {
            ASSERT_EQ(golden_lims[i+1], lims[i+1]);
            for (auto j = lims[i]; j < lims[i+1]; j++) {
                ASSERT_EQ(golden_labels[j], ids[j]);
                ASSERT_TRUE(C::cmp(distances[j], radius));
            }
        }
    }

 protected:
    knowhere::Config conf_;
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

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf_));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
    }

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_TRUE(index_->GetRawVectors() != nullptr);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
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
    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
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
    // serialize index
    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
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

    // hamming
    auto hamming_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        return faiss::xor_popcnt(pa, pb, code_size);
    };

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_hamming = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, radius, hamming_dis, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius, golden_labels, golden_distances, golden_lims);
    };

    test_range_search_hamming(hamming_radius, nullptr);
    test_range_search_hamming(hamming_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_jaccard) {
    float jaccard_radius = 0.5;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::JACCARD);
    knowhere::SetMetaRadius(conf_, jaccard_radius);

    auto jaccard_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        return 1.0 - (double)and_value / or_value;
    };

    // serialize index
    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_jaccard = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, radius, jaccard_dis, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius, golden_labels, golden_distances, golden_lims);
    };

    test_range_search_jaccard(jaccard_radius, nullptr);
    test_range_search_jaccard(jaccard_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_tanimoto) {
    float tanimoto_radius = 1.0;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::TANIMOTO);
    knowhere::SetMetaRadius(conf_, tanimoto_radius);

    auto tanimoto_dis = [](const uint8_t* pa, const uint8_t* pb, const size_t code_size) -> float {
        auto and_value = faiss::and_popcnt(pa, pb, code_size);
        auto or_value = faiss::or_popcnt(pa, pb, code_size);
        auto v = 1.0 - (double)and_value / or_value;
        return (-log2(1 - v));
    };

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_tanimoto = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, radius, tanimoto_dis, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius, golden_labels, golden_distances, golden_lims);
    };

    test_range_search_tanimoto(tanimoto_radius, nullptr);
    test_range_search_tanimoto(tanimoto_radius, *bitset);
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_superstructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUPERSTRUCTURE);

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf_, nullptr));
}

TEST_P(BinaryIDMAPTest, binaryidmap_range_search_substructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUBSTRUCTURE);

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());
    ASSERT_ANY_THROW(index_->QueryByRange(qd, conf_, nullptr));
}
