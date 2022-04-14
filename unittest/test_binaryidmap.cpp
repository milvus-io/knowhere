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
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexBinaryIDMAP.h"

#include "Helper.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

typedef int (*binary_int_func_ptr)(const int64_t*, const int64_t*);
typedef float (*binary_float_func_ptr)(const int64_t*, const int64_t*);
typedef bool (*binary_bool_func_ptr)(const int64_t*, const int64_t*);

class BinaryIDMAPTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        Init_with_default(true);
        index_ = std::make_shared<knowhere::BinaryIDMAP>();
    }

    void
    TearDown() override{};

    // for hamming
    void
    RunRangeSearchBF(
        std::vector<std::vector<bool>>& golden_result,
        std::vector<size_t>& golden_cnt,
        binary_int_func_ptr func,
        float radius) {
        for (auto i = 0; i < nq; ++i) {
            const int64_t* pq = reinterpret_cast<int64_t*>(xq_bin.data()) + i;
            const int64_t* pb = reinterpret_cast<int64_t*>(xb_bin.data());
            for (auto j = 0; j < nb; ++j) {
                auto dist = func(pq, pb + j);
                if (dist < radius) {
                    golden_result[i][j] = true;
                    golden_cnt[i]++;
                }
            }
        }
    };

    // for jaccard & tanimoto
    void
    RunRangeSearchBF(
        std::vector<std::vector<bool>>& golden_result,
        std::vector<size_t>& golden_cnt,
        binary_float_func_ptr func,
        float radius) {
        for (auto i = 0; i < nq; ++i) {
            const int64_t* pq = reinterpret_cast<int64_t*>(xq_bin.data()) + i;
            const int64_t* pb = reinterpret_cast<int64_t*>(xb_bin.data());
            for (auto j = 0; j < nb; ++j) {
                auto dist = func(pq, pb + j);
                if (dist < radius) {
                    golden_result[i][j] = true;
                    golden_cnt[i]++;
                }
            }
        }
    };

    // for superstructure & substructure
    void
    RunRangeSearchBF(
        std::vector<std::vector<bool>>& golden_result,
        std::vector<size_t>& golden_cnt,
        binary_bool_func_ptr func,
        float radius) {
        for (auto i = 0; i < nq; ++i) {
            const int64_t* pq = reinterpret_cast<int64_t*>(xq_bin.data()) + i;
            const int64_t* pb = reinterpret_cast<int64_t*>(xb_bin.data());
            for (auto j = 0; j < nb; ++j) {
                auto dist = func(pq, pb + j);
                if (dist > radius) {
                    golden_result[i][j] = true;
                    golden_cnt[i]++;
                }
            }
        }
    };

    void
    CheckRangeSearchResult(
        std::vector<std::vector<bool>>& golden_result,
        std::vector<size_t>& golden_cnt,
        std::vector<knowhere::DynamicResultSegment>& results) {
        for (auto i = 0; i < nq; ++i) {
            int correct_cnt = 0;
            for (auto& res_space : results[i]) {
                auto qnr = res_space->buffer_size * res_space->buffers.size() - res_space->buffer_size + res_space->wp;
                for (auto j = 0; j < qnr; ++j) {
                    auto bno = j / res_space->buffer_size;
                    auto pos = j % res_space->buffer_size;
                    auto idx = res_space->buffers[bno].ids[pos];
                    ASSERT_EQ(golden_result[i][idx], true);
                    if (golden_result[i][idx]) {
                        correct_cnt++;
                    }
                }
            }
            ASSERT_EQ(correct_cnt, golden_cnt[i]);
        }
    }

 protected:
    knowhere::BinaryIDMAPPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(METRICParameters,
                        BinaryIDMAPTest,
                        Values(std::string("HAMMING"),
                               std::string("JACCARD"),
                               std::string("TANIMOTO"),
                               std::string("SUPERSTRUCTURE"),
                               std::string("SUBSTRUCTURE")));

TEST_P(BinaryIDMAPTest, binaryidmap_basic) {
    ASSERT_TRUE(!xb_bin.empty());

    std::string MetricType = GetParam();
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, MetricType},
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

    std::string MetricType = GetParam();
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, MetricType},
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
}

TEST_P(BinaryIDMAPTest, binaryidmap_slice) {
    std::string MetricType = GetParam();
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, MetricType},
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

TEST_P(BinaryIDMAPTest, binaryidmap_range_search) {
    std::string MetricType = GetParam();
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::IndexParams::range_search_radius, radius},
        {knowhere::IndexParams::range_search_buffer_size, buffer_size},
        {knowhere::Metric::TYPE, MetricType},
    };

    std::vector<std::vector<bool>> golden_result(nq, std::vector<bool>(nb, false));
    std::vector<size_t> golden_cnt(nq, 0);

    // hamming
    int hamming_radius = 10;
    auto hamming_dis = [](const int64_t* pa, const int64_t* pb) -> int { return __builtin_popcountl((*pa) ^ (*pb)); };

    // jaccard
    float jaccard_radius = 0.4;
    auto jaccard_dis = [](const int64_t* pa, const int64_t* pb) -> float {
        auto intersection = __builtin_popcountl((*pa) & (*pb));
        auto Union = __builtin_popcountl((*pa) | (*pb));
        return 1.0 - (double)intersection / (double)Union;
    };

    // tanimoto
    float tanimoto_radius = 0.2;
    auto tanimoto_dis = [](const int64_t* pa, const int64_t* pb) -> float {
        auto intersection = __builtin_popcountl((*pa) & (*pb));
        auto Union = __builtin_popcountl((*pa) | (*pb));
        auto jcd = 1.0 - (double)intersection / (double)Union;
        return (-log2(1 - jcd));
    };

    // superstructure
    bool superstructure_radius = false;
    auto superstructure_dis = [](const int64_t* pa, const int64_t* pb) -> bool { return ((*pa) & (*pb)) == (*pb); };

    // substructure
    int substructure_radius = false;
    auto substructure_dis = [](const int64_t* pa, const int64_t* pb) -> bool { return ((*pa) & (*pb)) == (*pa); };

    auto metric_type = conf[knowhere::Metric::TYPE].get<std::string>();
    float curr_radius = radius;
    if (metric_type == "HAMMING") {
        curr_radius = hamming_radius;
        RunRangeSearchBF(golden_result, golden_cnt, hamming_dis, hamming_radius);
    } else if (metric_type == "JACCARD") {
        curr_radius = jaccard_radius;
        RunRangeSearchBF(golden_result, golden_cnt, jaccard_dis, jaccard_radius);
    } else if (metric_type == "TANIMOTO") {
        curr_radius = tanimoto_radius;
        RunRangeSearchBF(golden_result, golden_cnt, tanimoto_dis, tanimoto_radius);
    } else if (metric_type == "SUPERSTRUCTURE") {
        curr_radius = superstructure_radius;
        RunRangeSearchBF(golden_result, golden_cnt, superstructure_dis, superstructure_radius);
    } else if (metric_type == "SUBSTRUCTURE") {
        curr_radius = substructure_radius;
        RunRangeSearchBF(golden_result, golden_cnt, substructure_dis, substructure_radius);
    } else {
        std::cout << "unsupport type of metric type" << std::endl;
    }

    conf[knowhere::IndexParams::range_search_radius] = curr_radius;

    // serialize index
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    std::vector<knowhere::DynamicResultSegment> results1;
    for (auto i = 0; i < nq; ++i) {
        auto qd = knowhere::GenDataset(1, dim, xq_bin.data() + i * dim / 8);
        results1.push_back(index_->QueryByDistance(qd, conf, nullptr));
    }
    CheckRangeSearchResult(golden_result, golden_cnt, results1);

    auto binaryset = index_->Serialize(conf);
    index_->Load(binaryset);

    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    std::vector<knowhere::DynamicResultSegment> results2;
    for (auto i = 0; i < nq; ++i) {
        auto qd = knowhere::GenDataset(1, dim, xq_bin.data() + i * dim / 8);
        results2.push_back(index_->QueryByDistance(qd, conf, nullptr));
    }
    CheckRangeSearchResult(golden_result, golden_cnt, results1);
}
