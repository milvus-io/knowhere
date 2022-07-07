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
    std::vector<knowhere::MetricType> array = {
        knowhere::metric::L2,
        knowhere::metric::IP,
    };
    for (std::string& metric_type : array) {
        if (metric_type == knowhere::metric::IP) {
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
    std::vector<knowhere::MetricType> array = {
        knowhere::metric::HAMMING,
        knowhere::metric::JACCARD,
        knowhere::metric::TANIMOTO,
        knowhere::metric::SUPERSTRUCTURE,
        knowhere::metric::SUBSTRUCTURE,
    };
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