#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/factory.h"
#include "utils.h"

TEST_CASE("Test All Index Search.", "[search]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;

    auto base_gen = [&]() {
        knowhere::Json json;
        json["dim"] = dim;
        json["metric_type"] = "L2";
        json["k"] = 1;
        return json;
    };

    auto annoy_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json["n_trees"] = 16;
        json["search_k"] = 100;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json["nlist"] = 1024;
        json["nprobe"] = 1024;
        return json;
    };

    auto ivfsq_gen = ivfflat_gen;

    auto flat_gen = base_gen;

    auto ivfpq_gen = [&ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json["m"] = 4;
        json["nbits"] = 8;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json["M"] = 128;
        json["efConstruction"] = 200;
        json["ef"] = 32;
        json["range_k"] = 20;
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

#ifdef USE_CUDA
    auto gpu_flat_gen = [&base_gen]() {
        auto json = base_gen();
        return json;
    };
#endif
    SECTION("Test Cpu Index Search.") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({

            make_tuple(knowhere::IndexEnum::INDEX_ANNOY, annoy_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
#ifdef USE_CUDA
            make_tuple("GPUFLAT", gpu_flat_gen),
            make_tuple("GPUIVFFLAT", ivfflat_gen),
            make_tuple("GPUIVFPQ", ivfpq_gen),
            make_tuple("GPUIVFSQ", ivfsq_gen),
#endif
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
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx, *train_ds, json);
        }
        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 0; i < 1000; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Cpu Index Serial/Deserial.") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({

            make_tuple(knowhere::IndexEnum::INDEX_ANNOY, annoy_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
#ifdef USE_CUDA
            make_tuple("GPUFLAT", gpu_flat_gen),
            make_tuple("GPUIVFFLAT", ivfflat_gen),
            make_tuple("GPUIVFPQ", ivfpq_gen),
            make_tuple("GPUIVFSQ", ivfsq_gen),
#endif
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
        if (name == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
            load_raw_data(idx_, *train_ds, json);
        }
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 0; i < 1000; ++i) {
            CHECK(ids[i] == i);
        }
    }
}
