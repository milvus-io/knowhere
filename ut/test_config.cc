#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "index/flat/flat_config.h"
#include "knowhere/config.h"
TEST_CASE("Test config json parse", "[config]") {
    knowhere::Json json = knowhere::Json::parse(
        R"({"dim": 128, "metric_type": "L2", "k": 10, "radius": 100.0 , "nlist": 100, "nprobe": 80, "gpu_id": 0, "m": 4, "nbits": 8 })");
    SECTION("check flat index config") {
        knowhere::FlatConfig cfg;
        knowhere::Config::Load(cfg, json, knowhere::TRAIN);
        CHECK(cfg.dim == 128);
        CHECK(cfg.metric_type == "L2");
        CHECK_FALSE(cfg.radius == 100.0);
    }
}
