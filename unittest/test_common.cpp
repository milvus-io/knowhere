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

#include <thread>
#include <gtest/gtest.h>

#include "knowhere/common/Dataset.h"
#include "knowhere/common/Timer.h"
#include "knowhere/common/Exception.h"
#include "knowhere/utils/BitsetView.h"

/*Some unittest for knowhere/common, mainly for improve code coverage.*/

TEST(COMMON_TEST, dataset_test) {
    knowhere::Dataset set;
    int64_t v1 = 111;

    set.Set("key1", v1);
    auto get_v1 = set.Get<int64_t>("key1");
    ASSERT_EQ(get_v1, v1);

    ASSERT_ANY_THROW(set.Get<int8_t>("key1"));
    ASSERT_ANY_THROW(set.Get<int64_t>("dummy"));
}

TEST(COMMON_TEST, knowhere_exception) {
    const std::string msg = "test";
    knowhere::KnowhereException ex(msg);
    ASSERT_EQ(ex.what(), msg);
}

TEST(COMMON_TEST, time_recoder) {
    knowhere::TimeRecorder recoder("COMMTEST", 0);
    std::this_thread::sleep_for(std::chrono::seconds{1});
    double span = recoder.ElapseFromBegin("get time");
    ASSERT_GE(span, 1.0);
}

TEST(COMMON_TEST, BitsetView) {
    int N = 120;
    std::shared_ptr<uint8_t[]> data(new uint8_t[N/8]);
    auto bitset = faiss::BitsetView(data.get(), N);

    std::vector<uint8_t> init_array = {0x0, 0x1, 0x3, 0x7, 0xf, 0xf1, 0xf3, 0xf7, 0xff};
    for (size_t i = 0; i < init_array.size(); i++) {
        memset(data.get(), init_array[i], N / 8);
        ASSERT_EQ(bitset.count(), N / 8 * i);
        std::cout << bitset.to_string(0, N) << std::endl;
    }
}
