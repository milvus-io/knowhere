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

#include <gtest/gtest.h>
#include "unittest/utils.h"
#include "knowhere/utils/BitsetView.h"

class UtilsBitsetTest : public ::testing::Test {
 protected:
    const std::vector<size_t> kBitsetSizes{4, 8, 10, 64, 100, 500, 1024};
};

TEST_F(UtilsBitsetTest, FirstTBits) {
    for (const auto size : kBitsetSizes) {
        for (size_t i = 0; i <= size; ++i) {
            auto bitset_data = GenerateBitsetWithFirstTbitsSet(size, i);
            faiss::BitsetView bitset(bitset_data.data(), size);
            for (size_t j = 0; j < i; ++j) {
                ASSERT_TRUE(bitset.test(j));
            }
            for (size_t j = i; j < size; ++j) {
                ASSERT_FALSE(bitset.test(j));
            }
        }
    }
}

TEST_F(UtilsBitsetTest, RandomTBits) {
    for (const auto size : kBitsetSizes) {
        for (size_t i = 0; i <= size; ++i) {
            auto bitset_data = GenerateBitsetWithRandomTbitsSet(size, i);
            faiss::BitsetView bitset(bitset_data.data(), size);
            size_t cnt = 0;
            for (size_t j = 0; j < size; ++j) {
                cnt += bitset.test(j);
            }
            ASSERT_EQ(cnt, i);
        }
    }
}
