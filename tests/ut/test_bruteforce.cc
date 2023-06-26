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

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "utils.h"

TEST_CASE("Test Brute Force", "[float vector]") {
    using Catch::Approx;

    int64_t nb = 1000;
    int64_t dim = 128;
    int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds);

    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
        {knowhere::meta::RADIUS, knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99},
    };

    SECTION("Test Search") {
        auto res = knowhere::BruteForce::Search(train_ds, query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        for (int i = 0; i < nb; i++) {
            REQUIRE(ids[i * k] == i);
            if (metric == knowhere::metric::L2) {
                REQUIRE(dist[i * k] == 0);
            } else {
                REQUIRE(std::abs(dist[i * k]- 1.0) < 0.00001);
            }
        }
    }

    SECTION("Test Search With Buf") {
        auto ids = new int64_t[nb * k];
        auto dist = new float[nb * k];
        auto res = knowhere::BruteForce::SearchWithBuf(train_ds, query_ds, ids, dist, conf, nullptr);
        REQUIRE(res == knowhere::Status::success);
        for (int i = 0; i < nb; i++) {
            REQUIRE(ids[i * k] == i);
            if (metric == knowhere::metric::L2) {
                REQUIRE(dist[i * k] == 0);
            } else {
                REQUIRE(std::abs(dist[i * k]- 1.0) < 0.00001);
            }
        }
    }

    SECTION("Test Range Search") {
        auto res = knowhere::BruteForce::RangeSearch(train_ds, query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        auto lims = res.value()->GetLims();
        for (int i = 0; i < nb; i++) {
            REQUIRE(lims[i] == i);
            REQUIRE(ids[i] == i);
            if (metric == knowhere::metric::L2) {
                REQUIRE(dist[i] == 0);
            } else {
                REQUIRE(std::abs(dist[i]- 1.0) < 0.00001);
            }
        }
    }
}

TEST_CASE("Test Brute Force", "[binary vector]") {
    using Catch::Approx;

    int64_t nb = 1000;
    int64_t dim = 1024;
    int64_t k = 5;

    auto metric =
        GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD, knowhere::metric::TANIMOTO, knowhere::metric::SUPERSTRUCTURE, knowhere::metric::SUBSTRUCTURE);

    const auto train_ds = GenBinDataSet(nb, dim);
    const auto query_ds = CopyBinDataSet(train_ds);

    std::unordered_map<std::string, float> radius_map = {
        {knowhere::metric::HAMMING, 1.0},
        {knowhere::metric::JACCARD, 0.1},
        {knowhere::metric::TANIMOTO, 0.1},
    };
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };

    SECTION("Test Search") {
        auto res = knowhere::BruteForce::Search(train_ds, query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        for (int i = 0; i < nb; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
    }

    SECTION("Test Search With Buf") {
        auto ids = new int64_t[nb * k];
        auto dist = new float[nb * k];
        auto res = knowhere::BruteForce::SearchWithBuf(train_ds, query_ds, ids, dist, conf, nullptr);
        REQUIRE(res == knowhere::Status::success);
        for (int i = 0; i < nb; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
    }

    SECTION("Test Range Search") {
        if (metric == knowhere::metric::SUPERSTRUCTURE || metric == knowhere::metric::SUBSTRUCTURE) {
            return;
        }

        // set radius for different metric type
        auto cfg = conf;
        cfg[knowhere::meta::RADIUS] = radius_map[metric];

        auto res = knowhere::BruteForce::RangeSearch(train_ds, query_ds, cfg, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        auto lims = res.value()->GetLims();
        for (int i = 0; i < nb; i++) {
            REQUIRE(lims[i] == i);
            REQUIRE(ids[i] == i);
            REQUIRE(dist[i] == 0);
        }
    }
}
