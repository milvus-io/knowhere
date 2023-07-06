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
#include "faiss/utils/binary_distances.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.99f;
}  // namespace

TEST_CASE("Test Mem Index With Float Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
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
        json[knowhere::indexparam::EF] = 64;
        return json;
    };

    auto load_raw_data = [](knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSet& dataset,
                            const knowhere::Json& conf) {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto p_data = dataset.GetTensor();
        knowhere::BinarySet bs;
        REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
        bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
        bptr->size = dim * rows * sizeof(float);
        bs.Append("RAW_DATA", bptr);
        REQUIRE(index.Deserialize(bs) == knowhere::Status::success);
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search(train_ds, query_ds, conf, nullptr);

    SECTION("Test Search") {
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
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx, *train_ds, json);
        }
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        if (name != "IVF_PQ") {
            REQUIRE(recall > kKnnRecallThreshold);
        }
    }

    SECTION("Test Range Search") {
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
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx, *train_ds, json);
        }
        auto results = idx.RangeSearch(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        if (name != "IVF_PQ") {
            for (int i = 0; i < nq; ++i) {
                CHECK(ids[lims[i]] == i);
            }
        }
    }

    SECTION("Test Search with Bitset") {
        using std::make_tuple;
        auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFThreshold),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        const auto bitset_percentages = {0.4f, 0.98f};
        for (const float percentage : bitset_percentages) {
            for (const auto& gen_func : gen_bitset_funcs) {
                auto bitset_data = gen_func(nb, percentage * nb);
                knowhere::BitsetView bitset(bitset_data.data(), nb);
                auto results = idx.Search(*query_ds, json, bitset);
                auto gt = knowhere::BruteForce::Search(train_ds, query_ds, json, bitset);
                float recall = GetKNNRecall(*gt.value(), *results.value());
                if (percentage > threshold) {
                    REQUIRE(recall > kBruteForceRecallThreshold);
                } else {
                    REQUIRE(recall > kKnnRecallThreshold);
                }
            }
        }
    }

    SECTION("Test Serialize/Deserialize") {
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
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_ = knowhere::IndexFactory::Instance().Create(name);
        idx_.Deserialize(bs);
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx_, *train_ds, json);
        }
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }

    SECTION("Test IVFPQ with invalid params") {
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
        auto train_ds = GenDataSet(nb, dim);
        auto res = idx.Build(*train_ds, ivf_pq_gen());
        REQUIRE(res == knowhere::Status::faiss_inner_error);
    }
}

TEST_CASE("Test Mem Index With Binary Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 1024;
    const int64_t topk = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD);
    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::HAMMING) ? 10.0 : 0.1;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto flat_gen = base_gen;
    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 64;
        return json;
    };

    const auto train_ds = GenBinDataSet(nb, dim);
    const auto query_ds = GenBinDataSet(nq, dim);
    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };

    auto gt = knowhere::BruteForce::Search(train_ds, query_ds, conf, nullptr);
    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall > kKnnRecallThreshold);
    }

    SECTION("Test Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ivfflat_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        auto results = idx.RangeSearch(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        for (int i = 0; i < nq; ++i) {
            CHECK(ids[lims[i]] == i);
        }
    }

    SECTION("Test Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ivfflat_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_ = knowhere::IndexFactory::Instance().Create(name);
        idx_.Deserialize(bs);
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }
}

TEST_CASE("Test Mem Index With Binary Vector", "[bool metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t topk = 5;

    auto dim = GENERATE(as<int64_t>{}, 8, 16, 32, 64, 128, 256, 512, 160);
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::SUPERSTRUCTURE, knowhere::metric::SUBSTRUCTURE);

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto flat_gen = base_gen;
    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
        return json;
    };

    auto GenTestDataSet = [](int rows, int dim) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<> distrib(0.0, 100.0);
        int uint8_num = dim / 8;
        uint8_t* ts = new uint8_t[rows * uint8_num];
        for (int i = 0; i < rows; ++i) {
            auto v = (uint8_t)distrib(rng);
            for (int j = 0; j < uint8_num; ++j) {
                ts[i * uint8_num + j] = v;
            }
        }
        auto ds = knowhere::GenDataSet(rows, dim, ts);
        ds->SetIsOwner(true);
        return ds;
    };
    const auto train_ds = GenTestDataSet(nb, dim);
    const auto query_ds = GenTestDataSet(nq, dim);

    SECTION("Test Search") {
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, flat_gen),
            std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ivfflat_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        if (name == knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP) {
            REQUIRE(res == knowhere::Status::success);
        } else {
            REQUIRE(res == knowhere::Status::faiss_inner_error);
            return;
        }
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();

        auto code_size = dim / 8;
        for (int64_t i = 0; i < nq; i++) {
            const uint8_t* query_vector = (const uint8_t*)query_ds->GetTensor() + i * code_size;
            std::vector<int64_t> ids_v(ids + i * topk, ids + (i + 1) * topk);
            auto ds = GenIdsDataSet(topk, ids_v);
            auto gv_res = idx.GetVectorByIds(*ds);
            REQUIRE(gv_res.has_value());
            for (int64_t j = 0; j < topk; j++) {
                const uint8_t* res_vector = (const uint8_t*)gv_res.value()->GetTensor() + j * code_size;
                if (metric == knowhere::metric::SUPERSTRUCTURE) {
                    REQUIRE(faiss::is_subset(query_vector, res_vector, code_size));
                } else {
                    REQUIRE(faiss::is_subset(res_vector, query_vector, code_size));
                }
            }
        }
    }

    SECTION("Test Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ivfflat_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create(name);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(*train_ds, json);
        if (name == knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP) {
            REQUIRE(res == knowhere::Status::success);
        } else {
            REQUIRE(res == knowhere::Status::faiss_inner_error);
            return;
        }
        auto results = idx.RangeSearch(*query_ds, json, nullptr);
        REQUIRE(results.error() == knowhere::Status::faiss_inner_error);
    }
}
