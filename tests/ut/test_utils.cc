// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace {
const std::vector<size_t> kBitsetSizes{4, 8, 10, 64, 100, 500, 1024};
}

TEST_CASE("Test Vector Normalization", "[normalize]") {
    using Catch::Approx;

    const float floatDiff = 0.00001;
    uint64_t nb = 1000000;
    uint64_t dim = 128;
    int64_t seed = 42;

    SECTION("Test normalize") {
        auto train_ds = GenDataSet(nb, dim, seed);
        auto data = (float*)train_ds->GetTensor();

        knowhere::Normalize(*train_ds);

        for (size_t i = 0; i < nb; ++i) {
            float sum = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                auto val = data[i * dim + j];
                sum += val * val;
            }
            CHECK(std::abs(1.0f - sum) <= floatDiff);
        }
    }
}

TEST_CASE("Test Bitset Generation", "[utils]") {
    SECTION("Sequential") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithFirstTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                for (size_t j = 0; j < i; ++j) {
                    REQUIRE(bitset.test(j));
                }
                for (size_t j = i; j < size; ++j) {
                    REQUIRE(!bitset.test(j));
                }
            }
        }
    }

    SECTION("Random") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithRandomTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                size_t cnt = 0;
                for (size_t j = 0; j < size; ++j) {
                    cnt += bitset.test(j);
                }
                REQUIRE(cnt == i);
            }
        }
    }
}
