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
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "utils.h"

#ifdef USE_CUDA
TEST_CASE("Test All GPU Index", "[search]") {
    using Catch::Approx;

    knowhere::KnowhereConfig::InitGPUResource(0);

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        json[knowhere::meta::TOPK] = 1;
        json[knowhere::meta::RADIUS] = 10.0;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 4;
        return json;
    };

    auto ivfsq_gen = ivfflat_gen;

    auto ivfpq_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 4;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto gpu_flat_gen = [&base_gen]() {
        auto json = base_gen();
        return json;
    };

    SECTION("Test Gpu Index Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            // GPU_FLAT cannot run this test is because its Train() and Add() actually run in CPU,
            // "res_" in gpu_index_ is not set correctly
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IDMAP, gpu_flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8, ivfsq_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 0; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IDMAP, gpu_flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8, ivfsq_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_ = knowhere::IndexFactory::Instance().Create(name);
        idx_.Deserialize(bs);
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 0; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    knowhere::KnowhereConfig::FreeGPUResource();
}
#endif
