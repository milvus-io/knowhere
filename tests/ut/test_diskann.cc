#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/diskann/diskann.cc"
#include "index/diskann/diskann_config.h"
#include "knowhere/knowhere.h"
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

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * num * dim);
    writer.close();
}

float
GetTopKRecall(const knowhere::DataSetPtr ground_truth, const knowhere::DataSetPtr result) {
    REQUIRE(ground_truth->GetDim() >= result->GetDim());
    REQUIRE(ground_truth->GetRows() >= result->GetRows());

    auto nq = result->GetRows();
    auto gt_k = ground_truth->GetDim();
    auto res_k = result->GetDim();
    auto gt_ids = ground_truth->GetIds();
    auto res_ids = result->GetIds();

    uint32_t matched_num = 0;
    for (auto i = 0; i < nq; ++i) {
        std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + res_k);
        std::vector<int64_t> ids_1(res_ids + i * res_k, res_ids + i * res_k + res_k);

        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());

        std::vector<int64_t> v(nq * 2);
        std::vector<int64_t>::iterator it;
        it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it - v.begin());
        matched_num += v.size();
    }
    return ((float)matched_num) / ((float)nq * res_k);
}

}  // namespace

TEST_CASE("Test DiskANNIndexNode.", "[diskann]") {
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, "IP", "L2");

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == "L2" ? kL2IndexPrefix : kIPIndexPrefix);
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["build_list_size"] = 110;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        json["build_threads_num"] = 8;
        return json;
    };

    auto search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = (metric_str == "L2" ? kL2IndexPrefix : kIPIndexPrefix);
        json["search_threads_num"] = 8;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_list_size"] = 10;
        json["beamwidth"] = 8;
        return json;
    };

    auto query_ds = GenDataSet(kNumQueries, kDim);
    knowhere::DataSetPtr gt_ptr = nullptr;
    {
        auto base_ds = GenDataSet(kNumRows, kDim);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk(kRawDataPath, base_ptr, kNumRows, kDim);

        auto flat = knowhere::IndexFactory::Instance().Create("FLAT");
        auto cfg_json = base_gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto build_res = flat.Build(*base_ds, json);
        REQUIRE(build_res == knowhere::Status::success);
        auto results = flat.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        gt_ptr = results.value();
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
        //  knn search process
        {
            auto diskann = knowhere::IndexFactory::Instance().Create("DISKANNFLOAT", diskann_index_pack);
            auto search_json = search_gen().dump();
            knowhere::Json json = knowhere::Json::parse(search_json);
            auto res = diskann.Search(*query_ds, json, nullptr);
            REQUIRE(res.has_value());
            auto recall = GetTopKRecall(gt_ptr, res.value());
            REQUIRE(recall > 0.90);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}
