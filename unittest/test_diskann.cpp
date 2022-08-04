// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>

#include <cmath>
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
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/LocalFileManager.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using IdDisPair = std::pair<int64_t, float>;
using GroundTruth = std::vector<std::vector<int64_t>>;
using GroundTruthPtr = std::shared_ptr<GroundTruth>;

namespace {

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 1;
constexpr uint32_t kDim = 56;
constexpr float kMax = 100;
constexpr uint32_t kK = 10;
constexpr uint32_t kBigK = kNumRows * 2;
constexpr float kIpRadius = 0;
constexpr float kL2Radius = 330000;

std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kIpIndexDir = kDir + "/ip_index";
std::string kL2IndexDir = kDir + "/l2_index";

const knowhere::DiskANNBuildConfig build_conf{kRawDataPath, 100, 150, 0.2, 0.2, 4, 0};
const knowhere::DiskANNPrepareConfig prep_conf{4, 0, false, false};
const knowhere::DiskANNQueryConfig query_conf{kK, kK * 10, 3};
const knowhere::DiskANNQueryByRangeConfig l2_range_search_conf{kL2Radius, 10, 10000, 3};
const knowhere::DiskANNQueryByRangeConfig ip_range_search_conf{kIpRadius, 10, 10000, 3};

std::random_device rd;
size_t x = rd();
std::mt19937 generator((unsigned)x);
std::uniform_real_distribution<float> distribution(-1, 1);

float*
GenData(size_t num) {
    float* data_p = new float[num];

    for (int i = 0; i < num; ++i) {
        float rnd_val = distribution(generator) * static_cast<float>(kMax);
        data_p[i] = rnd_val;
    }

    return data_p;
}

struct DisPairLess {
    bool
    operator()(const IdDisPair& p1, const IdDisPair& p2) {
        return p1.second < p2.second;
    }
};

GroundTruthPtr
GenGroundTruth(const float* data_p, const float* query_p, const std::string metric) {
    GroundTruthPtr ground_truth = std::make_shared<GroundTruth>();
    ground_truth->resize(kNumQueries);

    for (uint32_t query_index = 0; query_index < kNumQueries; ++query_index) {  // for each query
        // use priority_queue to keep the topK;
        std::priority_queue<IdDisPair, std::vector<IdDisPair>, DisPairLess> pq;
        for (int64_t row = 0; row < kNumRows; ++row) {  // for each row
            float dis = 0;
            for (uint32_t dim = 0; dim < kDim; ++dim) {  // for every dim
                if (metric == knowhere::metric::IP) {
                    dis -= (data_p[kDim * row + dim] * query_p[query_index * kDim + dim]);
                } else {
                    dis += ((data_p[kDim * row + dim] - query_p[query_index * kDim + dim]) *
                            (data_p[kDim * row + dim] - query_p[query_index * kDim + dim]));
                }
            }
            if (pq.size() < kK) {
                pq.push(std::make_pair(row, dis));
            } else if (pq.top().second > dis) {
                pq.pop();
                pq.push(std::make_pair(row, dis));
            }
        }

        auto& result_ids = ground_truth->at(query_index);

        // write id in priority_queue to vector for sorting.
        int pq_size = pq.size();
        for (uint32_t index = 0; index < pq_size; ++index) {
            auto& id_dis_pair = pq.top();
            result_ids.push_back(id_dis_pair.first);
            pq.pop();
        }
    }
    return ground_truth;
}

GroundTruthPtr
GenRangeSearchGrounTruth(const float* data_p, const float* query_p, const std::string metric) {
    GroundTruthPtr ground_truth = std::make_shared<GroundTruth>();
    ground_truth->resize(kNumQueries);
    float radius = metric == knowhere::metric::L2 ? kL2Radius : kIpRadius;
    for (uint32_t query_index = 0; query_index < kNumQueries; ++query_index) {
        std::vector<IdDisPair> paris;
        for (int64_t row = 0; row < kNumRows; ++row) {  // for each row
            float dis = 0;
            for (uint32_t dim = 0; dim < kDim; ++dim) {  // for every dim
                if (metric == knowhere::metric::IP) {
                    dis -= (data_p[kDim * row + dim] * query_p[query_index * kDim + dim]);
                } else {
                    dis += ((data_p[kDim * row + dim] - query_p[query_index * kDim + dim]) *
                            (data_p[kDim * row + dim] - query_p[query_index * kDim + dim]));
                }
            }
            if (dis <= radius) {
                ground_truth->at(query_index).emplace_back(row);
            }
        }
    }
    return ground_truth;
}

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&kNumRows, sizeof(uint32_t));
    writer.write((char*)&kDim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * kNumRows * kDim);
    writer.close();
}

uint32_t
GetMatchedNum(const std::vector<int64_t>& ground_truth, const int64_t* result, const int32_t limit) {
    uint32_t matched_num = 0;
    int missed = 0;
    for (uint32_t index = 0; index < limit; ++index) {
        if (std::find(ground_truth.begin(), ground_truth.end(), result[index]) != ground_truth.end()) {
            matched_num++;
        }
    }
    return matched_num;
}

float
CheckTopKRecall(GroundTruthPtr ground_truth, const int64_t* result, const int32_t k) {
    uint32_t recall = 0;
    for (uint32_t n = 0; n < kNumQueries; ++n) {
        recall += GetMatchedNum(ground_truth->at(n), result + (n * k), ground_truth->at(n).size());
    }
    return ((float)recall) / ((float)kNumQueries * k);
}

float
CheckRangeSearchRecall(GroundTruthPtr ground_truth, const int64_t* result, const size_t* limits) {
    uint32_t recall = 0;
    uint32_t total = 0;
    for (uint32_t n = 0; n < kNumQueries; ++n) {
        recall += GetMatchedNum(ground_truth->at(n), result + limits[n], limits[n + 1] - limits[n]);
        total += ground_truth->at(n).size();
    }
    if (total == 0) {
        return 1;
    }
    return ((float)recall) / ((float)total);
}

template <typename DiskANNConfig>
void
check_config_error(DiskANNConfig& config_to_test) {
    knowhere::Config cfg;
    DiskANNConfig::Set(cfg, config_to_test);
    EXPECT_THROW(DiskANNConfig::Get(cfg), knowhere::KnowhereException);
}

}  // namespace

class DiskANNTest : public TestWithParam<std::string> {
 public:
    DiskANNTest() {
        metric_ = GetParam();
        InitDiskANN();
    }

    ~DiskANNTest() {
    }

    static void
    SetUpTestCase() {
        LOG_KNOWHERE_INFO_ << "Setting up the test environment for DiskANN Unittest.";
        fs::remove_all(kDir);
        fs::remove(kDir);

        raw_data_ = GenData(kNumRows * kDim);
        query_data_ = GenData(kNumQueries * kDim);

        ip_ground_truth_ = GenGroundTruth(raw_data_, query_data_, knowhere::metric::IP);
        l2_ground_truth_ = GenGroundTruth(raw_data_, query_data_, knowhere::metric::L2);
        ip_range_search_ground_truth_ = GenRangeSearchGrounTruth(raw_data_, query_data_, knowhere::metric::IP);
        l2_range_search_ground_truth_ = GenRangeSearchGrounTruth(raw_data_, query_data_, knowhere::metric::L2);

        big_query_ground_truth_ = std::make_shared<GroundTruth>();
        big_query_ground_truth_->resize(kNumQueries);
        for (uint32_t query_index = 0; query_index < kNumQueries; ++query_index) {
            for (uint32_t row = 0; row < kNumRows; ++row) {
                (big_query_ground_truth_->at(query_index)).emplace_back(row);
            }
            for (uint32_t row = kNumRows; row < kBigK; ++row) {
                (big_query_ground_truth_->at(query_index)).emplace_back(-1);
            }
        }

        // prepare the dir
        ASSERT_TRUE(fs::create_directory(kDir));
        ASSERT_TRUE(fs::create_directory(kIpIndexDir));
        ASSERT_TRUE(fs::create_directory(kL2IndexDir));

        WriteRawDataToDisk(kRawDataPath, raw_data_);

        knowhere::Config cfg;
        knowhere::DiskANNBuildConfig::Set(cfg, build_conf);

        auto diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
            kIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        diskann_ip->BuildAll(nullptr, cfg);
        auto diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
            kL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        diskann_l2->BuildAll(nullptr, cfg);
    }

    static void
    TearDownTestCase() {
        LOG_KNOWHERE_INFO_ << "Cleaning up the test environment for DiskANN Unittest.";
        delete[] raw_data_;
        delete[] query_data_;
        // Clean up the dir

        fs::remove_all(kDir);
        fs::remove(kDir);
    }

 protected:
    void
    InitDiskANN() {
        auto index_dir = metric_ == knowhere::metric::L2 ? kL2IndexDir : kIpIndexDir;
        diskann = std::make_unique<knowhere::IndexDiskANN<float>>(index_dir + "/diskann", metric_,
                                                                  std::make_unique<knowhere::LocalFileManager>());
    }
    static float* raw_data_;
    static float* query_data_;
    static GroundTruthPtr ip_ground_truth_;
    static GroundTruthPtr l2_ground_truth_;
    static GroundTruthPtr ip_range_search_ground_truth_;
    static GroundTruthPtr l2_range_search_ground_truth_;
    static GroundTruthPtr big_query_ground_truth_;
    std::string metric_;
    std::unique_ptr<knowhere::VecIndex> diskann;
};

float* DiskANNTest::query_data_ = nullptr;
float* DiskANNTest::raw_data_ = nullptr;
GroundTruthPtr DiskANNTest::ip_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::ip_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::big_query_ground_truth_ = nullptr;

INSTANTIATE_TEST_CASE_P(DiskANNParameters, DiskANNTest, Values(knowhere::metric::L2, knowhere::metric::IP));

TEST_P(DiskANNTest, knn_search_test) {
    knowhere::Config cfg;
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(kNumQueries, kDim, (void*)query_data_);
    // test query before preparation
    EXPECT_THROW(diskann->Query(data_set_ptr, cfg, nullptr), knowhere::KnowhereException);

    // test preparation
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);

    if (metric_ == knowhere::metric::IP) {
        auto ip_recall = CheckTopKRecall(ip_ground_truth_, ids, kK);
        EXPECT_GT(ip_recall, 0.8);
    } else {
        auto l2_recall = CheckTopKRecall(l2_ground_truth_, ids, kK);
        EXPECT_GT(l2_recall, 0.8);
    }
}

TEST_P(DiskANNTest, knn_search_big_k_test) {
    knowhere::Config cfg;

    // test preparation
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig query_conf_to_test = query_conf;
    query_conf_to_test.k = kBigK;
    query_conf_to_test.search_list_size = kBigK * 10;
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf_to_test);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(kNumQueries, kDim, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);

    auto recall = CheckTopKRecall(big_query_ground_truth_, ids, kBigK);
    EXPECT_GT(recall, 0.8);
}

TEST_P(DiskANNTest, range_search_test) {
    knowhere::Config cfg;
    auto range_search_conf = metric_ == knowhere::metric::IP ? ip_range_search_conf : l2_range_search_conf;
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(kNumQueries, kDim, (void*)query_data_);
    // test query before preparation
    EXPECT_THROW(diskann->QueryByRange(data_set_ptr, cfg, nullptr), knowhere::KnowhereException);

    // test preparation
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query by range
    cfg.clear();
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf);
    auto result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);
    auto lims = knowhere::GetDatasetLims(result);

    if (metric_ == knowhere::metric::IP) {
        auto ip_recall = CheckRangeSearchRecall(ip_range_search_ground_truth_, ids, lims);
        EXPECT_GT(ip_recall, 0.5);
    } else {
        auto l2_recall = CheckRangeSearchRecall(l2_range_search_ground_truth_, ids, lims);
        EXPECT_GT(l2_recall, 0.8);
    }
}

TEST_P(DiskANNTest, cached_warmup_test) {
    knowhere::Config cfg;

    // search cache + warmup preparation
    knowhere::DiskANNPrepareConfig prep_conf_to_test = prep_conf;
    prep_conf_to_test.warm_up = true;
    prep_conf_to_test.num_nodes_to_cache = 1000;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf_to_test);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(kNumQueries, kDim, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);
    auto ids = knowhere::GetDatasetIDs(result);

    if (metric_ == knowhere::metric::IP) {
        auto ip_recall = CheckTopKRecall(ip_ground_truth_, ids, kK);
        EXPECT_GT(ip_recall, 0.8);
    } else {
        auto l2_recall = CheckTopKRecall(l2_ground_truth_, ids, kK);
        EXPECT_GT(l2_recall, 0.8);
    }

    // bfs cache + warmup preparation
    InitDiskANN();
    cfg.clear();
    prep_conf_to_test.use_bfs_cache = true;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf_to_test);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query by range
    cfg.clear();
    auto range_search_conf = metric_ == knowhere::metric::IP ? ip_range_search_conf : l2_range_search_conf;
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf);
    data_set_ptr = knowhere::GenDataset(kNumQueries, kDim, (void*)query_data_);
    result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);

    ids = knowhere::GetDatasetIDs(result);
    auto lims = knowhere::GetDatasetLims(result);
    // p(ids, lims[1], ip_range_search_ground_truth_);

    if (metric_ == knowhere::metric::IP) {
        auto ip_recall = CheckRangeSearchRecall(ip_range_search_ground_truth_, ids, lims);
        EXPECT_GT(ip_recall, 0.5);
    } else {
        auto l2_recall = CheckRangeSearchRecall(l2_range_search_ground_truth_, ids, lims);
        EXPECT_GT(l2_recall, 0.8);
    }
}

TEST_P(DiskANNTest, config_test) {
    // build config
    knowhere::DiskANNBuildConfig build_conf_to_test = build_conf;
    build_conf_to_test.max_degree = 0;
    check_config_error<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test.max_degree = 513;
    check_config_error<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test = build_conf;
    build_conf_to_test.num_threads = 0;
    check_config_error<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test.num_threads = 129;
    check_config_error<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    // prepare config
    knowhere::DiskANNPrepareConfig prep_conf_to_test = prep_conf;
    prep_conf_to_test.num_threads = 0;
    check_config_error<knowhere::DiskANNPrepareConfig>(prep_conf_to_test);

    prep_conf_to_test.num_threads = 129;
    check_config_error<knowhere::DiskANNPrepareConfig>(prep_conf_to_test);

    // query config
    knowhere::DiskANNQueryConfig query_conf_to_test = query_conf;
    query_conf_to_test.k = 10;
    query_conf_to_test.search_list_size = 9;
    check_config_error<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    query_conf_to_test = query_conf;
    query_conf_to_test.beamwidth = 0;
    check_config_error<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    query_conf_to_test.beamwidth = 129;
    check_config_error<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    // query by range config
    knowhere::DiskANNQueryByRangeConfig range_search_conf_to_test = l2_range_search_conf;
    range_search_conf_to_test.min_k = 10;
    range_search_conf_to_test.max_k = 9;
    check_config_error<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);

    range_search_conf_to_test = l2_range_search_conf;
    range_search_conf_to_test.beamwidth = 0;
    check_config_error<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);

    range_search_conf_to_test.beamwidth = 129;
    check_config_error<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);
}