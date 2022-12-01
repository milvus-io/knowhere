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

#include "knowhere/archive/BruteForce.h"
#include "knowhere/index/IndexType.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class BruteForceTest : public DataGen, public TestWithParam<knowhere::IndexMode> {
 protected:
    void
    SetUp() override {
    }

    void
    TearDown() override {
    }

 protected:
    knowhere::IndexMode index_mode_;
};

INSTANTIATE_TEST_CASE_P(
    BruteForceParameters,
    BruteForceTest,
    Values(knowhere::IndexMode::MODE_CPU)
    );

TEST_P(BruteForceTest, float_basic) {
    Init_with_default();
    std::vector<std::string> array = {"l2", "ip"};
    for (std::string& metric_type : array) {
        auto faiss_metric_type = knowhere::GetFaissMetricType(metric_type);
        if (faiss_metric_type == faiss::METRIC_INNER_PRODUCT) {
            normalize(xb.data(), nb, dim);
            normalize(xq.data(), nq, dim);
        }
        auto base_dataset = knowhere::GenDataset(nb, dim, xb.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, k},
        };
        auto result1 = knowhere::BruteForce::Search(base_dataset, query_dataset, config, nullptr);
        auto labels1 = knowhere::GetDatasetIDs(result1);
        for (int i = 0; i < nq; i++) {
            ASSERT_TRUE(labels1[i * k] == i);
        }

        // query with bitset
        auto result2 = knowhere::BruteForce::Search(base_dataset, query_dataset, config, *bitset);
        auto labels2 = knowhere::GetDatasetIDs(result2);
        for (int i = 0; i < nq; i++) {
            ASSERT_FALSE(labels2[i * k] == i);
        }
    }
}

TEST_P(BruteForceTest, binary_basic) {
    Init_with_default(true);
    std::vector<std::string> array = {"hamming", "jaccard", "tanimoto", "superstructure", "substructure"};
    for (std::string& metric_type : array) {
        auto base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, k},
        };
        auto result1 = knowhere::BruteForce::Search(base_dataset, query_dataset, config, nullptr);
        auto labels1 = knowhere::GetDatasetIDs(result1);
        for (int i = 0; i < nq; i++) {
            ASSERT_TRUE(labels1[i * k] == i);
        }

        // query with bitset
        auto result2 = knowhere::BruteForce::Search(base_dataset, query_dataset, config, *bitset);
        auto labels2 = knowhere::GetDatasetIDs(result2);
        for (int i = 0; i < nq; i++) {
            ASSERT_FALSE(labels2[i * k] == i);
        }
    }
}

TEST_P(BruteForceTest, float_range_search_l2) {
    Init_with_default();
    auto metric_type = knowhere::metric::L2;

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;

        auto base_dataset = knowhere::GenDataset(nb, dim, xb.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS_LOW_BOUND, low_bound},
            {knowhere::meta::RADIUS_HIGH_BOUND, high_bound},
        };

        RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, metric_type,
                              xb.data(), nb, xq.data(), nq, dim, low_bound, high_bound, bitset);

        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (std::pair<float, float> range: {
             std::make_pair<float, float>(0, 16.81f),
             std::make_pair<float, float>(16.81f, 17.64f),
             std::make_pair<float, float>(17.64f, 18.49f)}) {
            test_range_search_l2(range.first, range.second, nullptr);
            test_range_search_l2(range.first, range.second, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(BruteForceTest, float_range_search_ip) {
    Init_with_default();
    auto metric_type = knowhere::metric::IP;

    normalize(xb.data(), nb, dim);
    normalize(xq.data(), nq, dim);

    auto test_range_search_ip = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;

        auto base_dataset = knowhere::GenDataset(nb, dim, xb.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS_LOW_BOUND, low_bound},
            {knowhere::meta::RADIUS_HIGH_BOUND, high_bound},
        };

        RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, metric_type,
                              xb.data(), nb, xq.data(), nq, dim, low_bound, high_bound, bitset);

        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.70f, 0.75f),
        std::make_pair<float, float>(0.75f, 0.80f),
        std::make_pair<float, float>(0.80f, 1.01f)}) {
            test_range_search_ip(range.first, range.second, nullptr);
            test_range_search_ip(range.first, range.second, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(BruteForceTest, binary_range_search_hamming) {
    Init_with_default(true);
    auto metric_type = knowhere::metric::HAMMING;

    auto test_range_search_hamming = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;

        auto base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS_LOW_BOUND, low_bound},
            {knowhere::meta::RADIUS_HIGH_BOUND, high_bound},
        };

        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, metric_type,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, low_bound, high_bound, bitset);

        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 45.0f),
        std::make_pair<float, float>(45.0f, 48.0f),
        std::make_pair<float, float>(48.0f, 50.0f)}) {
        test_range_search_hamming(range.first, range.second, nullptr);
        test_range_search_hamming(range.first, range.second, *bitset);
    }
}

TEST_P(BruteForceTest, binary_range_search_jaccard) {
    Init_with_default(true);
    auto metric_type = knowhere::metric::JACCARD;

    auto test_range_search_jaccard = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;

        auto base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS_LOW_BOUND, low_bound},
            {knowhere::meta::RADIUS_HIGH_BOUND, high_bound},
        };
        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::JACCARD,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, low_bound, high_bound, bitset);

        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 0.5f),
        std::make_pair<float, float>(0.5f, 0.55f),
        std::make_pair<float, float>(0.55f, 0.6f)}) {
        test_range_search_jaccard(range.first, range.second, nullptr);
        test_range_search_jaccard(range.first, range.second, *bitset);
    }
}

TEST_P(BruteForceTest, binary_range_search_tanimoto) {
    Init_with_default(true);
    auto metric_type = knowhere::metric::TANIMOTO;

    auto test_range_search_tanimoto = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;

        auto base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        auto query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
        auto config = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, metric_type},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::RADIUS_LOW_BOUND, low_bound},
            {knowhere::meta::RADIUS_HIGH_BOUND, high_bound},
        };
        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, metric_type,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, low_bound, high_bound, bitset);

        auto result = knowhere::BruteForce::RangeSearch(base_dataset, query_dataset, config, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 1.0f),
        std::make_pair<float, float>(1.0f, 1.2f),
        std::make_pair<float, float>(1.2f, 1.5f)}) {
        test_range_search_tanimoto(range.first, range.second, nullptr);
        test_range_search_tanimoto(range.first, range.second, *bitset);
    }
}
