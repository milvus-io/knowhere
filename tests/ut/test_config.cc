#include "catch2/catch_test_macros.hpp"
#include "index/diskann/diskann_config.h"
#include "index/flat/flat_config.h"
#include "index/hnsw/hnsw_config.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/config.h"

TEST_CASE("Test config json parse", "[config]") {
    knowhere::Status s;
    SECTION("check flat index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100
        })");
        knowhere::FlatConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type == "L2");

        knowhere::FlatConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type == "L2");
        CHECK(search_cfg.k == 100);
    }

    SECTION("check ivf index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "nlist": 128,
            "nprobe": 16,
            "radius_low_bound": -1.0,
            "radius_high_bound": 1000.0,
            "trace_visit": true
        })");
        knowhere::IvfFlatConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type == "L2");
        CHECK(train_cfg.nlist == 128);

        knowhere::IvfFlatConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type == "L2");
        CHECK(search_cfg.k == 100);
        CHECK(search_cfg.nprobe == 16);

        knowhere::IvfFlatConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type == "L2");
        CHECK(range_cfg.radius_low_bound == -1.0);
        CHECK(range_cfg.radius_high_bound == 1000);

        knowhere::IvfFlatConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit == true);
    }

    SECTION("check hnsw index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "M": 32,
            "efConstruction": 100,
            "ef": 16,
            "radius_low_bound": -1.0,
            "radius_high_bound": 1000.0,
            "trace_visit": true
        })");
        knowhere::HnswConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type == "L2");
        CHECK(train_cfg.M == 32);
        CHECK(train_cfg.efConstruction == 100);

        knowhere::HnswConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type == "L2");
        CHECK(search_cfg.k == 100);
        CHECK(search_cfg.ef == 16);

        knowhere::HnswConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type == "L2");
        CHECK(range_cfg.radius_low_bound == -1.0);
        CHECK(range_cfg.radius_high_bound == 1000);

        knowhere::HnswConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit == true);
        CHECK(range_cfg.overview_levels == 3);
    }

    SECTION("check diskann index config") {
        knowhere::Json json = knowhere::Json::parse(R"({
            "metric_type": "L2",
            "k": 100,
            "index_prefix": "tmp",
            "data_path": "/tmp",
            "pq_code_budget_gb": 1.0,
            "build_dram_budget_gb": 1.0,
            "radius_low_bound": -10.0,
            "radius_high_bound": 1000.0 ,
            "trace_visit": true
        })");
        knowhere::DiskANNConfig train_cfg;
        s = knowhere::Config::Load(train_cfg, json, knowhere::TRAIN);
        CHECK(s == knowhere::Status::success);
        CHECK(train_cfg.metric_type == "L2");

        knowhere::DiskANNConfig search_cfg;
        s = knowhere::Config::Load(search_cfg, json, knowhere::SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(search_cfg.metric_type == "L2");
        CHECK(search_cfg.k == 100);

        knowhere::DiskANNConfig range_cfg;
        s = knowhere::Config::Load(range_cfg, json, knowhere::RANGE_SEARCH);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.metric_type == "L2");
        CHECK(range_cfg.radius_low_bound == -10.0);
        CHECK(range_cfg.radius_high_bound == 1000);

        knowhere::DiskANNConfig feder_cfg;
        s = knowhere::Config::Load(feder_cfg, json, knowhere::FEDER);
        CHECK(s == knowhere::Status::success);
        CHECK(range_cfg.trace_visit == true);
    }
}
