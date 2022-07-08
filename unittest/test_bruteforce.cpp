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
    std::vector<int64_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    std::vector<std::string> array = {"l2", "ip"};
    for (std::string& metric_type : array) {
        auto faiss_metric_type = knowhere::GetFaissMetricType(metric_type);
        if (faiss_metric_type == faiss::METRIC_INNER_PRODUCT) {
            normalize(xb.data(), nb, dim);
            normalize(xq.data(), nq, dim);
        }
        knowhere::BruteForceSearch(metric_type, xb.data(), xq.data(), dim, nb, nq, k, labels.data(), distances.data(),
                                   nullptr);
        for (int i = 0; i < nq; i++) {
            ASSERT_TRUE(labels[i * k] == i);
        }

        // query with bitset
        knowhere::BruteForceSearch(metric_type, xb.data(), xq.data(), dim, nb, nq, k, labels.data(), distances.data(),
                                   *bitset);
        for (int i = 0; i < nq; i++) {
            ASSERT_FALSE(labels[i * k] == i);
        }
    }
}

TEST_P(BruteForceTest, binary_basic) {
    Init_with_default(true);
    std::vector<int64_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    std::vector<std::string> array = {"hamming", "jaccard", "tanimoto", "superstructure", "substructure"};
    for (std::string& metric_type : array) {
        knowhere::BruteForceSearch(metric_type, xb_bin.data(), xq_bin.data(), dim, nb, nq, k, labels.data(),
                                   distances.data(), nullptr);
        for (int i = 0; i < nq; i++) {
            ASSERT_TRUE(labels[i * k] == i);
        }

        // query with bitset
        knowhere::BruteForceSearch(metric_type, xb_bin.data(), xq_bin.data(), dim, nb, nq, k, labels.data(),
                                   distances.data(), *bitset);
        for (int i = 0; i < nq; i++) {
            ASSERT_FALSE(labels[i * k] == i);
        }
    }
}

TEST_P(BruteForceTest, float_range_search_l2) {
    Init_with_default();
    auto metric_type = knowhere::metric::L2;

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        int64_t* labels = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        RunFloatRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, metric_type,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        knowhere::BruteForceRangeSearch(metric_type, xb.data(), xq.data(), dim, nb, nq, radius, labels, distances,
                                        lims, bitset);
        auto result = knowhere::GenResultDataset(labels, distances, lims);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius * radius, golden_labels.data(), golden_lims.data(), true);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (float radius: {4.1f, 4.2f, 4.3f}) {
            test_range_search_l2(radius, nullptr);
            test_range_search_l2(radius, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(BruteForceTest, float_range_search_ip) {
    Init_with_default();
    auto metric_type = knowhere::metric::IP;

    normalize(xb.data(), nb, dim);
    normalize(xq.data(), nq, dim);

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        int64_t* labels = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        RunFloatRangeSearchBF<CMax<float>>(golden_labels, golden_distances, golden_lims, metric_type,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        knowhere::BruteForceRangeSearch(metric_type, xb.data(), xq.data(), dim, nb, nq, radius, labels, distances,
                                        lims, bitset);
        auto result = knowhere::GenResultDataset(labels, distances, lims);
        CheckRangeSearchResult<CMax<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        //for (float radius: {42.0f, 43.0f, 44.0f}) {
        for (float radius: {0.75f, 0.78f, 0.81f}) {
            test_range_search_ip(radius, nullptr);
            test_range_search_ip(radius, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(BruteForceTest, binary_range_search_hamming) {
    Init_with_default(true);
    int hamming_radius = 50;
    auto metric_type = knowhere::metric::HAMMING;

    auto test_range_search_hamming = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        int64_t* labels = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, metric_type,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        knowhere::BruteForceRangeSearch(metric_type, xb_bin.data(), xq_bin.data(), dim, nb, nq, radius, labels,
                                        distances, lims, bitset);
        auto result = knowhere::GenResultDataset(labels, distances, lims);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_hamming(hamming_radius, nullptr);
    test_range_search_hamming(hamming_radius, *bitset);
}

TEST_P(BruteForceTest, binary_range_search_jaccard) {
    Init_with_default(true);
    float jaccard_radius = 0.5;
    auto metric_type = knowhere::metric::JACCARD;

    auto test_range_search_jaccard = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        int64_t* labels = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::JACCARD,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        knowhere::BruteForceRangeSearch(metric_type, xb_bin.data(), xq_bin.data(), dim, nb, nq, radius, labels,
                                        distances, lims, bitset);
        auto result = knowhere::GenResultDataset(labels, distances, lims);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_jaccard(jaccard_radius, nullptr);
    test_range_search_jaccard(jaccard_radius, *bitset);
}

TEST_P(BruteForceTest, binary_range_search_tanimoto) {
    Init_with_default(true);
    float tanimoto_radius = 1.0;
    auto metric_type = knowhere::metric::TANIMOTO;

    auto test_range_search_tanimoto = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        int64_t* labels = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;
        RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, metric_type,
                                            xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, bitset);

        knowhere::BruteForceRangeSearch(metric_type, xb_bin.data(), xq_bin.data(), dim, nb, nq, radius, labels,
                                        distances, lims, bitset);
        auto result = knowhere::GenResultDataset(labels, distances, lims);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), true);
    };

    test_range_search_tanimoto(tanimoto_radius, nullptr);
    test_range_search_tanimoto(tanimoto_radius, *bitset);
}
