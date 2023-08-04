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

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/diskann/diskann_config.h"
#include "index/flat/flat_config.h"
#include "index/hnsw/hnsw_config.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/config.h"

TEST_CASE("Test config json parse", "[config]") {
    knowhere::Status s;
    std::string err_msg;
    SECTION("check invalid json keys") {
        auto invalid_json_str = GENERATE(as<std::string>{},
                                         R"({
                "metric_type": "L2",
                "invalid_key": 100
            })",
                                         R"({
                "collection_id": 100,
                "segments_id": 101
            })",
                                         R"({
                "": 0
            })",
                                         R"({
                " ": 0
            })",
                                         R"({
                "topk": 100.1
            })",
                                         R"({
                "metric": "L2"
            })",
                                         R"({
                "12-s": 19878
            })",
                                         R"({
                "k": "100.12"
            })");
        knowhere::BaseConfig test_config;
        knowhere::Json test_json = knowhere::Json::parse(invalid_json_str);
        s = knowhere::Config::FormatAndCheck(test_config, test_json);
        CHECK(s != knowhere::Status::success);
    }

    SECTION("Check the json for the specific index") {
        knowhere::Json large_build_json = knowhere::Json::parse(R"({
            "beamwidth_ratio":"4.000000",
            "build_dram_budget_gb":4.38,
            "collection_id":"438538303581716485",
            "data_path":"temp",
            "dim":128,
            "disk_pq_dims":0,
            "field_id":"102",
            "index_build_id":"438538303582116508",
            "index_id":"0",
            "index_prefix":"temp",
            "index_type":"DISKANN",
            "index_version":"1",
            "max_degree":56,
            "metric_type":"L2",
            "num_build_thread":2,
            "num_build_thread_ratio":"1.000000",
            "num_load_thread":8,
            "num_load_thread_ratio":"8.000000",
            "partition_id":"438538303581716486",
            "pq_code_budget_gb":0.011920999735593796,
            "pq_code_budget_gb_ratio":"0.125000",
            "search_cache_budget_gb_ratio":"0.100000",
            "search_list_size":100,
            "segment_id":"438538303581916493"
        })");
        knowhere::HnswConfig hnsw_config;
        s = knowhere::Config::FormatAndCheck(hnsw_config, large_build_json);
        CHECK(s == knowhere::Status::invalid_param_in_json);
        knowhere::DiskANNConfig diskann_config;
        s = knowhere::Config::FormatAndCheck(diskann_config, large_build_json);
        CHECK(s == knowhere::Status::success);
    }

    SECTION("check flat index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100
        })");
        knowhere::FlatConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type.value() == "L2");

        knowhere::FlatConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type.value() == "L2");
        CHECK(search_cfg.k.value() == 100);
    }

    SECTION("check ivf index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "nlist": 128,
            "nprobe": 16,
            "radius": 1000.0,
            "range_filter": 1.0,
            "trace_visit": true
        })");
        knowhere::IvfFlatConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type.value() == "L2");
        CHECK(train_cfg.nlist.value() == 128);

        knowhere::IvfFlatConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type.value() == "L2");
        CHECK(search_cfg.k.value() == 100);
        CHECK(search_cfg.nprobe.value() == 16);

        knowhere::IvfFlatConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type.value() == "L2");
        CHECK(range_cfg.radius.value() == 1000.0);
        CHECK(range_cfg.range_filter.value() == 1.0);

        knowhere::IvfFlatConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit.value() == true);
    }

    SECTION("check hnsw index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "M": 32,
            "efConstruction": 100,
            "ef": 116,
            "range_filter": 1.0,
            "radius": 1000.0,
            "trace_visit": true
        })");

        // invalid value check
        {
            knowhere::HnswConfig wrong_cfg;
            auto invalid_value_json = json;
            invalid_value_json["efConstruction"] = 100.10;
            s = knowhere::Config::Load(wrong_cfg, invalid_value_json, knowhere::TRAIN);
            CHECK(s == knowhere::Status::type_conflict_in_json);

            invalid_value_json = json;
            invalid_value_json["ef"] = -1;
            s = knowhere::Config::Load(wrong_cfg, invalid_value_json, knowhere::SEARCH);
            CHECK(s == knowhere::Status::out_of_range_in_json);

            invalid_value_json = json;
            invalid_value_json["ef"] = nlohmann::json::array({20, 30, 40});
            s = knowhere::Config::Load(wrong_cfg, invalid_value_json, knowhere::SEARCH);
            CHECK(s == knowhere::Status::type_conflict_in_json);

            invalid_value_json = json;
            invalid_value_json["ef"] = 99;
            s = knowhere::Config::Load(wrong_cfg, invalid_value_json, knowhere::SEARCH);
            s = wrong_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::out_of_range_in_json);
        }

        knowhere::HnswConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type.value() == "L2");
        CHECK(train_cfg.M.value() == 32);
        CHECK(train_cfg.efConstruction.value() == 100);

        {
            knowhere::HnswConfig search_cfg;
            s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::success);
        }

        {
            knowhere::HnswConfig search_cfg;
            auto search_json = json;
            search_json.erase("ef");
            s = knowhere::Config::Load(search_cfg, search_json, knowhere::SEARCH);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::success);
            CHECK_EQ(100, search_cfg.ef.value());
        }

        {
            knowhere::HnswConfig search_cfg;
            auto search_json = json;
            search_json.erase("ef");
            search_json["k"] = 10;
            s = knowhere::Config::Load(search_cfg, search_json, knowhere::SEARCH);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::success);
            CHECK_EQ(16, search_cfg.ef.value());
        }

        knowhere::HnswConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type.value() == "L2");
        CHECK(range_cfg.radius.value() == 1000);
        CHECK(range_cfg.range_filter.value() == 1.0);

        knowhere::HnswConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit.value() == true);
        CHECK(range_cfg.overview_levels.value() == 3);
    }

    SECTION("check diskann index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "index_prefix": "tmp",
            "data_path": "/tmp",
            "pq_code_budget_gb": 1.0,
            "build_dram_budget_gb": 1.0,
            "radius": 1000.0 ,
            "range_filter": 1.0,
            "trace_visit": true
        })");
        {
            knowhere::DiskANNConfig train_cfg;
            s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
            CHECK(s == knowhere::Status::success);
            s = train_cfg.CheckAndAdjustForBuild();
            CHECK(s == knowhere::Status::success);
            CHECK_EQ(128, train_cfg.search_list_size.value());
            CHECK_EQ("L2", train_cfg.metric_type.value());
        }

        {
            knowhere::DiskANNConfig search_cfg;
            s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
            CHECK(s == knowhere::Status::success);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::success);
            CHECK_EQ("L2", search_cfg.metric_type.value());
            CHECK_EQ(100, search_cfg.k.value());
            CHECK_EQ(100, search_cfg.search_list_size.value());
        }

        {
            knowhere::DiskANNConfig search_cfg;
            auto search_json = json;
            search_json["k"] = 2;
            s = knowhere::Config::Load(search_cfg, search_json, knowhere::SEARCH);
            CHECK(s == knowhere::Status::success);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::success);
            CHECK_EQ(16, search_cfg.search_list_size.value());
        }

        {
            knowhere::DiskANNConfig search_cfg;
            auto search_json = json;
            search_json["search_list_size"] = 99;
            s = knowhere::Config::Load(search_cfg, search_json, knowhere::SEARCH);
            s = search_cfg.CheckAndAdjustForSearch(&err_msg);
            CHECK(s == knowhere::Status::out_of_range_in_json);
        }

        knowhere::DiskANNConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type.value() == "L2");
        CHECK(range_cfg.radius.value() == 1000.0);
        CHECK(range_cfg.range_filter.value() == 1.0);

        knowhere::DiskANNConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit.value() == true);
    }
}
