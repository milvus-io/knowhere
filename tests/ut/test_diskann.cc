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
#include "index/diskann/diskann.cc"
#include "index/diskann/diskann_config.h"
#include "knowhere/factory.h"
#include "local_file_manager.h"
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
            json["radius_low_bound"] = CFG_FLOAT(0);
            json["radius_high_bound"] = CFG_FLOAT(200000);
        } else {
            json["radius_low_bound"] = CFG_FLOAT(50000);
            json["radius_high_bound"] = std::numeric_limits<CFG_FLOAT>::max();
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
        json["build_threads_num"] = 8;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_threads_num"] = 8;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        return json;
    };

    auto range_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_threads_num"] = 8;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        json["search_list_and_k_ratio"] = 2.0;
        return json;
    };

    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    {
        auto base_ds = GenDataSet(kNumRows, kDim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();
        knn_gt_ptr = GetKNNGroundTruth(*base_ds, *query_ds, metric_str, kK);
        range_search_gt_ptr = GetRangeSearchGroundTruth(*base_ds, *query_ds, metric_str, base_json["radius_low_bound"],
                                                        base_json["radius_high_bound"]);
    }

    SECTION("Test L2/IP metric.") {
        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        // build process
        {
            knowhere::DataSet* ds_ptr = nullptr;
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANNFLOAT", diskann_index_pack);
            auto build_json = build_gen().dump();
            knowhere::Json json = knowhere::Json::parse(build_json);
            diskann.Build(*ds_ptr, json);
        }
        //  knn search & range search process
        {
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANNFLOAT", diskann_index_pack);
            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(*query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(recall > kL2KnnRecall);

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
