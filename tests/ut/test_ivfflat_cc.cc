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

#include <future>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/invlists/InvertedLists.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/factory.h"
#include "utils.h"

TEST_CASE("Test Build Search Concurrency", "[Concurrency]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;
    int64_t times = 5;
    int64_t top_k = 100;
    int64_t build_task_num = 1;
    int64_t search_task_num = 10;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        json[knowhere::meta::TOPK] = top_k;
        json[knowhere::meta::RADIUS] = 10.0;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 128;
        json[knowhere::indexparam::NPROBE] = 16;
        return json;
    };

    auto ivfflatcc_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::meta::NUM_BUILD_THREAD] = 1;
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    SECTION("Test Concurrent Invlists ") {
        size_t nlist = 128;
        size_t code_size = 512;
        size_t segment_size = 1024;

        auto invList = std::make_unique<faiss::ConcurrentArrayInvertedLists>(nlist, code_size, segment_size);

        for (size_t i = 0; i < nlist; i++) {
            REQUIRE(invList->list_size(i) == 0);
        }

        std::vector<size_t> list_size_count(nlist, 0);
        for (int cnt = 0; cnt < times; cnt++) {
            {
                // small batch append
                std::uniform_int_distribution<int> distribution(0, segment_size);
                for (size_t i = 0; i < nlist; i++) {
                    std::mt19937_64 rng(i);
                    int64_t add_size = distribution(rng);
                    std::vector<faiss::Index::idx_t> ids(add_size, i);
                    std::vector<uint8_t> codes(add_size * code_size, (uint8_t)(i % 256));
                    invList->add_entries(i, add_size, ids.data(), codes.data());
                    list_size_count[i] += add_size;
                    CHECK(invList->list_size(i) == list_size_count[i]);
                }
            }
            {
                // large batch append
                std::uniform_int_distribution<int> distribution(1, 5 * segment_size);
                for (size_t i = 0; i < nlist; i++) {
                    std::mt19937_64 rng(i * i);
                    int64_t add_size = distribution(rng);
                    std::vector<faiss::Index::idx_t> ids(add_size, i);
                    std::vector<uint8_t> codes(add_size * code_size, (uint8_t)(i % 256));
                    invList->add_entries(i, add_size, ids.data(), codes.data());
                    list_size_count[i] += add_size;
                    CHECK(invList->list_size(i) == list_size_count[i]);
                }
            }
        }
        {
            for (size_t i = 0; i < nlist; i++) {
                auto list_size = list_size_count[i];
                CHECK(invList->get_segment_num(i) == ((list_size / segment_size) + (list_size % segment_size != 0)));
                CHECK(invList->get_segment_size(i, invList->get_segment_num(i) - 1) ==
                      (list_size % segment_size == 0 ? segment_size : list_size % segment_size));
                CHECK(invList->get_segment_offset(i, 0) == 0);

                for (size_t j = 0; j < list_size; j++) {
                    CHECK(*(invList->get_ids(i, j)) == static_cast<int64_t>(i));
                }

                for (size_t j = 0; j < list_size; j++) {
                    for (size_t k = 0; k < code_size; k++) {
                        CHECK(*(invList->get_codes(i, j) + k) == static_cast<int64_t>(i));
                    }
                }
            }
        }
    }

    SECTION("Test Add & Search & RangeSearch Serialized ") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == name);

        auto& build_ds = train_ds;
        auto query_ds = GenDataSet(nq, dim, seed);

        for (int i = 1; i <= times; i++) {
            idx.Add(*build_ds, json);
            {
                auto results = idx.Search(*query_ds, json, nullptr);
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k <= i; k++) {
                        CHECK(ids[j * top_k + k] % nb == j);
                    }
                }
            }
            {
                auto results = idx.RangeSearch(*query_ds, json, nullptr);
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                auto lims = results.value()->GetLims();
                for (int j = 0; j < nq; ++j) {
                    for (int k = 0; k <= i; k++) {
                        CHECK(ids[lims[j] + k] % nb == j);
                    }
                }
            }
        }
    }

    SECTION("Test Build & Search Correctness") {
        using std::make_tuple;

        auto ivf_flat = knowhere::IndexFactory::Instance().Create(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT);
        auto ivf_flat_cc = knowhere::IndexFactory::Instance().Create(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC);

        knowhere::Json ivf_flat_json = knowhere::Json::parse(ivfflat_gen().dump());
        knowhere::Json ivf_flat_cc_json = knowhere::Json::parse(ivfflatcc_gen().dump());

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);

        auto flat_res = ivf_flat.Build(*train_ds, ivf_flat_json);
        REQUIRE(flat_res == knowhere::Status::success);
        auto cc_res = ivf_flat_cc.Build(*train_ds, ivf_flat_json);
        REQUIRE(cc_res == knowhere::Status::success);

        // test search
        {
            auto flat_results = ivf_flat.Search(*query_ds, ivf_flat_json, nullptr);
            REQUIRE(flat_results.has_value());

            auto cc_results = ivf_flat_cc.Search(*query_ds, ivf_flat_json, nullptr);
            REQUIRE(cc_results.has_value());

            auto flat_ids = flat_results.value()->GetIds();
            auto cc_ids = cc_results.value()->GetIds();
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < top_k; j++) {
                    auto id = i * top_k + j;
                    CHECK(flat_ids[id] == cc_ids[id]);
                }
            }
        }
        // test range_search
        {
            auto flat_results = ivf_flat.RangeSearch(*query_ds, ivf_flat_json, nullptr);
            REQUIRE(flat_results.has_value());

            auto cc_results = ivf_flat_cc.RangeSearch(*query_ds, ivf_flat_json, nullptr);
            REQUIRE(cc_results.has_value());

            auto flat_ids = flat_results.value()->GetIds();
            auto flat_limits = flat_results.value()->GetLims();
            auto cc_ids = cc_results.value()->GetIds();
            auto cc_limits = cc_results.value()->GetLims();
            for (int i = 0; i < nq; i++) {
                CHECK(flat_limits[i] == cc_limits[i]);
                CHECK(flat_limits[i + 1] == cc_limits[i + 1]);
                for (size_t offset = flat_limits[i]; offset < flat_limits[i + 1]; offset++) {
                    CHECK(flat_ids[offset] == cc_ids[offset]);
                }
            }
        }
    }

    SECTION("Test Add & Search & RangeSearch ConCurrent") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == name);

        auto& build_ds = train_ds;
        auto query_ds = GenDataSet(nq, dim, seed);

        for (int i = 1; i <= times; i++) {
            std::vector<std::future<knowhere::Status>> add_task_list;
            std::vector<std::future<knowhere::expected<knowhere::DataSetPtr, knowhere::Status>>> search_task_list;
            std::vector<std::future<knowhere::expected<knowhere::DataSetPtr, knowhere::Status>>> range_search_task_list;
            for (int j = 0; j < build_task_num; j++) {
                add_task_list.push_back(
                    std::async(std::launch::async, [&idx, &build_ds, &json] { return idx.Add(*build_ds, json); }));
            }
            for (int j = 0; j < search_task_num; j++) {
                search_task_list.push_back(std::async(
                    std::launch::async, [&idx, &query_ds, &json] { return idx.Search(*query_ds, json, nullptr); }));
            }
            for (int j = 0; j < search_task_num; j++) {
                range_search_task_list.push_back(std::async(std::launch::async, [&idx, &query_ds, &json] {
                    return idx.RangeSearch(*query_ds, json, nullptr);
                }));
            }
            for (auto& task : add_task_list) {
                REQUIRE(task.get() == knowhere::Status::success);
            }

            for (auto& task : search_task_list) {
                auto results = task.get();
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k < i; k++) {
                        CHECK(ids[j * top_k + k] % nb == j);
                    }
                }
            }
            for (auto& task : range_search_task_list) {
                auto results = task.get();
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                auto lims = results.value()->GetLims();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k < i; k++) {
                        CHECK(ids[lims[j] + k] % nb == j);
                    }
                }
            }
        }
    }
}
