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
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.8f;
constexpr float kBruteForceRecallThreshold = 0.99f;
constexpr size_t kTopk = 1;
const std::string kMetric = knowhere::metric::L2;
}  // namespace

TEST_CASE("Test All Mem Index Search", "[search]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 100;
    int64_t dim = 128;
    int64_t seed = 42;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = kMetric;
        json[knowhere::meta::TOPK] = kTopk;
        json[knowhere::meta::RADIUS] = 10.0;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 16;
        return json;
    };

    auto ivfflatcc_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    auto ivfsq_gen = ivfflat_gen;

    auto flat_gen = base_gen;

    auto ivfpq_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 4;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 32;
        return json;
    };

    auto load_raw_data = [](knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSet& dataset,
                            const knowhere::Json& conf) {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto p_data = dataset.GetTensor();
        knowhere::BinarySet bs;
        auto res = index.Serialize(bs);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
        bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
        bptr->size = dim * rows * sizeof(float);
        bs.Append("RAW_DATA", bptr);
        res = index.Deserialize(bs);
        REQUIRE(res == knowhere::Status::success);
    };

    const auto train_ds = GenDataSet(nb, dim, seed);
    const auto query_ds = GenDataSet(nq, dim, seed);
    const auto gt = GetKNNGroundTruth(*train_ds, *query_ds, kMetric, kTopk);
    SECTION("Test Cpu Index Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx, *train_ds, json);
        }
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt, *results.value());
        REQUIRE(recall > kKnnRecallThreshold);
    }

    SECTION("Test Cpu Index Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            // make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx, *train_ds, json);
        }
        auto results = idx.RangeSearch(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        for (int i = 0; i < nq; ++i) {
            CHECK(ids[lims[i]] == i);
        }
    }

    SECTION("Test Cpu Index Search with Bitset") {
        using std::make_tuple;
        auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswBruteForceFilterRate),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);

        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        const auto bitset_percentages = GetBitsetTestPercentagesFromThreshold(threshold);
        for (const float percentage : bitset_percentages) {
            for (const auto& gen_func : gen_bitset_funcs) {
                auto bitset_data = gen_func(nb, percentage * nb);
                knowhere::BitsetView bitset(bitset_data.data(), nb);
                auto results = idx.Search(*query_ds, json, bitset);
                auto gt = GetKNNGroundTruth(*train_ds, *query_ds, json[knowhere::meta::METRIC_TYPE],
                                            json[knowhere::meta::TOPK], bitset);
                float recall = GetKNNRecall(*gt, *results.value());
                if (percentage > threshold) {
                    REQUIRE(recall > kBruteForceRecallThreshold);
                } else {
                    REQUIRE(recall > kKnnRecallThreshold);
                }
            }
        }
    }

    SECTION("Test Cpu Index Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_ = knowhere::IndexFactory::Instance().Create(name);
        idx_.Deserialize(bs);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx_, *train_ds, json);
        }
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 0; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test build IVFPQ with invalid params") {
        auto idx = knowhere::IndexFactory::Instance().Create(knowhere::IndexEnum::INDEX_FAISS_IVFPQ);
        uint32_t nb = 1000;
        uint32_t dim = 128;
        auto ivf_pq_gen = [&]() {
            knowhere::Json json;
            json[knowhere::meta::DIM] = dim;
            json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
            json[knowhere::meta::TOPK] = 10;
            json[knowhere::indexparam::M] = 15;
            json[knowhere::indexparam::NBITS] = 8;
            return json;
        };
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(*train_ds, ivf_pq_gen());
        REQUIRE(res == knowhere::Status::faiss_inner_error);
    }
}
