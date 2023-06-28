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

#include <string>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/diskann/diskann.cc"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "utils.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>

namespace {
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 100;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLargeDim = 1536;
constexpr uint32_t kK = 10;
constexpr float kL2KnnRecall = 0.8;
constexpr float kL2RangeAp = 0.7;
constexpr float kIpRangeAp = 0.5;

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * num * dim);
    writer.close();
}

}  // namespace

TEST_CASE("Invalid diskann params test", "[diskann]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    auto test_gen = []() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = "L2";
        json["k"] = 100;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 24;
        json["search_list_size"] = 64;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.05 / (1024 * 1024 * 1024);
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        json["search_list_and_k_ratio"] = 2.0;
        return json;
    };
    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, kDim);
    // build process
    SECTION("Invalid build params test") {
        knowhere::DataSet* ds_ptr = nullptr;
        auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
        knowhere::Json test_json;
        knowhere::Status test_stat;
        // invalid metric type
        test_json = test_gen();
        test_json["metric_type"] = knowhere::metric::TANIMOTO;
        test_stat = diskann.Build(*ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
        // raw data path not exist
        test_json = test_gen();
        test_json["data_path"] = kL2IndexPrefix + ".temp";
        test_stat = diskann.Build(*ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::diskann_file_error);
    }

    SECTION("Invalid search params test") {
        knowhere::DataSet* ds_ptr = nullptr;
        auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
        diskann.Build(*ds_ptr, test_gen());
        diskann.Deserialize(knowhere::BinarySet(), test_gen());

        knowhere::Json test_json;
        auto query_ds = GenDataSet(kNumQueries, kDim, 42);

        // search list size < topk
        {
            test_json = test_gen();
            test_json["search_list_size"] = 1;
            auto res = diskann.Search(*query_ds, test_json, nullptr);
            REQUIRE_FALSE(res.has_value());
            REQUIRE(res.error() == knowhere::Status::out_of_range_in_json);
        }
        // min_k > max_k
        {
            test_json = test_gen();
            test_json["min_k"] = 10000;
            test_json["max_k"] = 100;
            auto res = diskann.RangeSearch(*query_ds, test_json, nullptr);
            REQUIRE_FALSE(res.has_value());
            REQUIRE(res.error() == knowhere::Status::invalid_args);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANNIndexNode.", "[diskann]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else {
            json["radius"] = CFG_FLOAT::value_type(50000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        return json;
    };

    auto range_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        json["search_list_and_k_ratio"] = 2.0;
        return json;
    };

    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    {
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test L2/IP metric.") {
        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;
        // build process
        {
            knowhere::DataSet* ds_ptr = nullptr;
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
            auto build_json = build_gen().dump();
            knowhere::Json json = knowhere::Json::parse(build_json);
            diskann.Build(*ds_ptr, json);
        }
        {
            // knn search
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
            diskann.Deserialize(binset, deserialize_json);
            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(*query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(recall > kL2KnnRecall);

            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
            const auto bitset_percentages = {0.4f, 0.98f};
            const auto bitset_thresholds = {-1.0f, 0.9f};
            for (const float threshold : bitset_thresholds) {
                knn_json["filter_threshold"] = threshold;
                for (const float percentage : bitset_percentages) {
                    for (const auto& gen_func : gen_bitset_funcs) {
                        auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                        auto results = diskann.Search(*query_ds, knn_json, bitset);
                        auto gt = knowhere::BruteForce::Search(base_ds, query_ds, knn_json, bitset);
                        float recall = GetKNNRecall(*gt.value(), *results.value());
                        if (percentage == 0.98f) {
                            REQUIRE(recall >= 0.9f);
                        } else {
                            REQUIRE(recall >= kL2KnnRecall);
                        }
                    }
                }
            }

            // range search process
            auto range_search_json = range_search_gen().dump();
            knowhere::Json range_json = knowhere::Json::parse(range_search_json);
            auto range_search_res = diskann.RangeSearch(*query_ds, range_json, nullptr);
            REQUIRE(range_search_res.has_value());
            auto ap = GetRangeSearchRecall(*range_search_gt_ptr, *range_search_res.value());
            float standard_ap = metric_str == knowhere::metric::L2 ? kL2RangeAp : kIpRangeAp;
            REQUIRE(ap > standard_ap);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// This test case only check L2
TEST_CASE("Test DiskANN GetVectorByIds", "[diskann]") {
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [&] {
            knowhere::Json json;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [&]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 5;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            return json;
        };

        auto query_ds = GenDataSet(kNumQueries, dim, 42);
        auto base_ds = GenDataSet(kNumRows, dim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSet* ds_ptr = nullptr;
        auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        diskann.Build(*ds_ptr, json);
        knowhere::BinarySet binset;
        {
            std::vector<double> cache_sizes = {0, 1.0f * sizeof(float) * dim * kNumRows * 0.125 / (1024 * 1024 * 1024)};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);
                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    std::cout << "Testing dim = " << dim << ", cache_size = " << cache_size
                              << ", ids_size = " << ids_size << std::endl;
                    auto ids_ds = GenIdsDataSet(ids_size);
                    auto results = index.GetVectorByIds(*ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (float*)base_ds->GetTensor();
                    auto data = (float*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANNIndexNode with cache opt", "[diskann]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2);

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        return json;
    };

    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    auto base_ds = GenDataSet(kNumRows, kDim, 30);

    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, kDim);

    {
        auto base_json = base_gen();
        auto result_knn = knowhere::BruteForce::Search(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
    }

    SECTION("Test L2/IP metric.") {
        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        {
            knowhere::DataSet* ds_ptr = nullptr;
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
            auto build_json = build_gen().dump();

            knowhere::Json json = knowhere::Json::parse(build_json);
            auto index_prefix = json["index_prefix"];
            fs::remove_all(index_prefix);
            fs::remove(index_prefix);
            fs::create_directory(index_prefix);

            auto build_status = diskann.Build(*ds_ptr, json);
            REQUIRE(build_status == knowhere::Status::success);
        }
        float recall1, recall2;
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;
        {
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
            diskann.Deserialize(binset, deserialize_json);
            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(*query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            recall1 = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(recall1 > kL2KnnRecall);
        }
        {
            std::string cached_nodes_file_path =
                std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
            REQUIRE(fs::exists(cached_nodes_file_path));
            fs::remove(cached_nodes_file_path);
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANN", diskann_index_pack);
            diskann.Deserialize(binset, deserialize_json);
            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(*query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            recall2 = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(recall2 > kL2KnnRecall);
        }
        REQUIRE(recall1 == recall2);
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}
