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
#include <unordered_set>
#include <utility>
#include <vector>

#include "knowhere/feder/DiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/RangeUtil.h"
#include "unittest/LocalFileManager.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using IdDisPair = std::pair<int64_t, float>;
using GroundTruth = std::vector<std::vector<int64_t>>;
using GroundTruthPtr = std::shared_ptr<GroundTruth>;

namespace {

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 56;
constexpr float kMax = 100;
constexpr uint32_t kK = 10;
constexpr uint32_t kBigK = kNumRows * 2;
constexpr float kL2RadiusLowBound = 0;
constexpr float kL2RadiusHighBound = 300000;
constexpr float kIPRadiusLowBound = 50000;
constexpr float kIPRadiusHighBound = std::numeric_limits<float>::max();
constexpr float kDisLossTolerance = 0.5;

constexpr uint32_t kLargeDimNumRows = 1000;
constexpr uint32_t kLargeDimNumQueries = 10;
constexpr uint32_t kLargeDim = 5600;
constexpr uint32_t kLargeDimBigK = kLargeDimNumRows * 2;
constexpr float kLargeDimL2RadiusLowBound = 0;
constexpr float kLargeDimL2RadiusHighBound = 36000000;
constexpr float kLargeDimIPRadiusLowBound = 400000;
constexpr float kLargeDimIPRadiusHighBound = std::numeric_limits<float>::max();

std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kLargeDimRawDataPath = kDir + "/large_dim_raw_data";
std::string kIpIndexDir = kDir + "/ip_index";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kLargeDimIpIndexDir = kDir + "/large_dim_ip_index";
std::string kLargeDimL2IndexDir = kDir + "/large_dim_l2_index";

const knowhere::DiskANNBuildConfig build_conf{kRawDataPath, 50, 90, 0.2, 0.2, 4, 0};
const knowhere::DiskANNBuildConfig large_dim_build_conf{kLargeDimRawDataPath, 50, 90, 0.2, 0.2, 4, 0};
const knowhere::DiskANNPrepareConfig prep_conf{4, 0, false, false};
const knowhere::DiskANNQueryConfig query_conf{kK, kK * 10, 3};
const knowhere::DiskANNQueryByRangeConfig l2_range_search_conf{kL2RadiusLowBound, kL2RadiusHighBound, 10, 10000, 3};
const knowhere::DiskANNQueryByRangeConfig ip_range_search_conf{kIPRadiusLowBound, kIPRadiusHighBound, 10, 10000, 3};
const knowhere::DiskANNQueryByRangeConfig large_dim_l2_range_search_conf{kLargeDimL2RadiusLowBound, kLargeDimL2RadiusHighBound, 10, 1000, 3};
const knowhere::DiskANNQueryByRangeConfig large_dim_ip_range_search_conf{kLargeDimIPRadiusLowBound, kLargeDimIPRadiusHighBound, 10, 1000, 3};

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
GenGroundTruth(const float* data_p, const float* query_p, const std::string metric, const uint32_t num_rows,
               const uint32_t num_dims, const uint32_t num_queries, const faiss::BitsetView bitset = nullptr) {
    GroundTruthPtr ground_truth = std::make_shared<GroundTruth>();
    ground_truth->resize(num_queries);

    for (uint32_t query_index = 0; query_index < num_queries; ++query_index) {  // for each query
        // use priority_queue to keep the topK;
        std::priority_queue<IdDisPair, std::vector<IdDisPair>, DisPairLess> pq;
        for (int64_t row = 0; row < num_rows; ++row) {  // for each row
            if (!bitset.empty() && bitset.test(row)) {
                continue;
            }
            float dis = 0;
            for (uint32_t dim = 0; dim < num_dims; ++dim) {  // for every dim
                if (metric == knowhere::metric::IP) {
                    dis -= (data_p[num_dims * row + dim] * query_p[query_index * num_dims + dim]);
                } else {
                    dis += ((data_p[num_dims * row + dim] - query_p[query_index * num_dims + dim]) *
                            (data_p[num_dims * row + dim] - query_p[query_index * num_dims + dim]));
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
GenRangeSearchGrounTruth(const float* data_p, const float* query_p, const std::string metric, const uint32_t num_rows,
                         const uint32_t num_dims, const uint32_t num_queries, const float radius_low_bound,
                         const float radius_high_bound, const faiss::BitsetView bitset = nullptr) {
    GroundTruthPtr ground_truth = std::make_shared<GroundTruth>();
    ground_truth->resize(num_queries);
    bool is_ip = (metric == knowhere::metric::IP);
    for (uint32_t query_index = 0; query_index < num_queries; ++query_index) {
        std::vector<IdDisPair> paris;
        const float* xq = query_p + query_index * num_dims;
        for (int64_t row = 0; row < num_rows; ++row) {  // for each row
            if (!bitset.empty() && bitset.test(row)) {
                continue;
            }
            const float* xb = data_p + row * num_dims;
            float dis = 0;
            if (metric == knowhere::metric::IP) {
                for (uint32_t dim = 0; dim < num_dims; ++dim) {  // for every dim
                    dis += xb[dim] * xq[dim];
                }
            } else {
                for (uint32_t dim = 0; dim < num_dims; ++dim) {  // for every dim
                    dis += std::pow(xb[dim] - xq[dim], 2);
                }
            }
            if (knowhere::distance_in_range(dis, radius_low_bound, radius_high_bound, is_ip)) {
                ground_truth->at(query_index).emplace_back(row);
            }
        }
    }
    return ground_truth;
}

void
WriteRawDataToDisk(const std::string data_path, const float* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(float) * num * dim);
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
CheckTopKRecall(GroundTruthPtr ground_truth, const int64_t* result, const int32_t k, const uint32_t num_queries) {
    uint32_t recall = 0;
    for (uint32_t n = 0; n < num_queries; ++n) {
        recall += GetMatchedNum(ground_truth->at(n), result + (n * k), ground_truth->at(n).size());
    }
    return ((float)recall) / ((float)num_queries * k);
}

float
CheckRangeSearchRecall(GroundTruthPtr ground_truth, const int64_t* result, const size_t* limits,
                       const uint32_t num_queries) {
    uint32_t recall = 0;
    uint32_t total = 0;
    for (uint32_t n = 0; n < num_queries; ++n) {
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
CheckConfigError(DiskANNConfig& config_to_test) {
    knowhere::Config cfg;
    DiskANNConfig::Set(cfg, config_to_test);
    EXPECT_THROW(DiskANNConfig::Get(cfg), knowhere::KnowhereException);
}

void
CheckDistanceError(const float* data_p, const float* query_p, const knowhere::DatasetPtr result,
                   const std::string metric, const uint32_t num_query, const uint32_t dim_query, const uint32_t topk,
                   const uint32_t row_nums, const bool is_large_dim) {
    if (is_large_dim)
        return;
    auto res_ids_p = knowhere::GetDatasetIDs(result);
    auto res_dis_p = knowhere::GetDatasetDistance(result);
    uint32_t valid_res_num = topk < row_nums ? topk : row_nums;
    for (auto q = 0; q < num_query; q++) {
        for (auto k = 0; k < valid_res_num; k++) {
            auto id_q_k = res_ids_p[q * topk + k];
            EXPECT_NE(id_q_k, -1);

            float true_dis = 0;
            if (metric == knowhere::metric::IP) {
                for (int d = 0; d < dim_query; d++) {
                    true_dis += (data_p[dim_query * id_q_k + d] * query_p[dim_query * q + d]);
                }
            } else if (metric == knowhere::metric::L2) {
                for (int d = 0; d < dim_query; d++) {
                    true_dis += ((data_p[dim_query * id_q_k + d] - query_p[dim_query * q + d]) *
                                 (data_p[dim_query * id_q_k + d] - query_p[dim_query * q + d]));
                }
            }
            EXPECT_NEAR(true_dis, res_dis_p[q * topk + k], kDisLossTolerance);
        }
    }
}

}  // namespace

class DiskANNTest : public TestWithParam<std::tuple<std::string, bool>> {
 public:
    DiskANNTest() {
        std::tie(metric_, is_large_dim_) = GetParam();
        if (!is_large_dim_) {
            dim_ = kDim;
            num_rows_ = kNumRows;
            num_queries_ = kNumQueries;
            big_k_ = kBigK;
            raw_data_ = global_raw_data_;
            query_data_ = global_query_data_;
            ground_truth_ = metric_ == knowhere::metric::L2 ? l2_ground_truth_ : ip_ground_truth_;
            range_search_ground_truth_ =
                metric_ == knowhere::metric::L2 ? l2_range_search_ground_truth_ : ip_range_search_ground_truth_;
            range_search_conf_ = metric_ == knowhere::metric::L2 ? l2_range_search_conf : ip_range_search_conf;
            radius_low_bound_ = metric_ == knowhere::metric::L2 ? kL2RadiusLowBound : kIPRadiusLowBound;
            radius_high_bound_ = metric_ == knowhere::metric::L2 ? kL2RadiusHighBound : kIPRadiusHighBound;
        } else {
            dim_ = kLargeDim;
            num_rows_ = kLargeDimNumRows;
            num_queries_ = kLargeDimNumQueries;
            big_k_ = kLargeDimBigK;
            raw_data_ = global_large_dim_raw_data_;
            query_data_ = global_large_dim_query_data_;
            ground_truth_ = metric_ == knowhere::metric::L2 ? large_dim_l2_ground_truth_ : large_dim_ip_ground_truth_;
            range_search_ground_truth_ = metric_ == knowhere::metric::L2 ? large_dim_l2_range_search_ground_truth_
                                                                         : large_dim_ip_range_search_ground_truth_;
            range_search_conf_ =
                metric_ == knowhere::metric::L2 ? large_dim_l2_range_search_conf : large_dim_ip_range_search_conf;
            radius_low_bound_ = metric_ == knowhere::metric::L2 ? kLargeDimL2RadiusLowBound : kLargeDimIPRadiusLowBound;
            radius_high_bound_ = metric_ == knowhere::metric::L2 ? kLargeDimL2RadiusHighBound : kLargeDimIPRadiusHighBound;
        }
        InitDiskANN();
    }

    ~DiskANNTest() {
    }

    static void
    SetUpTestCase() {
        LOG_KNOWHERE_INFO_ << "Setting up the test environment for DiskANN Unittest.";
        fs::remove_all(kDir);
        fs::remove(kDir);

        global_raw_data_ = GenData(kNumRows * kDim);
        global_large_dim_raw_data_ = GenData(kLargeDimNumRows * kLargeDim);
        global_query_data_ = GenData(kNumQueries * kDim);
        global_large_dim_query_data_ = GenData(kLargeDimNumQueries * kLargeDim);

        // global_raw_data_ = new float[kNumRows * kDim];
        // std::ifstream r(kRawDataPath, std::ios::binary);
        // uint32_t tem;
        // r.read((char*)&tem, sizeof(uint32_t));
        // r.read((char*)&tem, sizeof(uint32_t));
        // r.read((char*)global_raw_data_, kNumRows * kDim * sizeof(float));
        // r.close();
        // global_large_dim_raw_data_ = new float[kLargeDimNumRows * kLargeDim];
        // std::ifstream lr(kLargeDimRawDataPath, std::ios::binary);
        // lr.read((char*)&tem, sizeof(uint32_t));
        // lr.read((char*)&tem, sizeof(uint32_t));
        // lr.read((char*)global_large_dim_raw_data_, kLargeDimNumRows * kLargeDim * sizeof(float));
        // lr.close();

        ip_ground_truth_ =
            GenGroundTruth(global_raw_data_, global_query_data_, knowhere::metric::IP, kNumRows, kDim, kNumQueries);
        l2_ground_truth_ =
            GenGroundTruth(global_raw_data_, global_query_data_, knowhere::metric::L2, kNumRows, kDim, kNumQueries);
        ip_range_search_ground_truth_ = GenRangeSearchGrounTruth(
            global_raw_data_, global_query_data_, knowhere::metric::IP, kNumRows, kDim, kNumQueries, kIPRadiusLowBound, kIPRadiusHighBound);
        l2_range_search_ground_truth_ = GenRangeSearchGrounTruth(
            global_raw_data_, global_query_data_, knowhere::metric::L2, kNumRows, kDim, kNumQueries, kL2RadiusLowBound, kL2RadiusHighBound);

        large_dim_ip_ground_truth_ =
            GenGroundTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::IP,
                           kLargeDimNumRows, kLargeDim, kLargeDimNumQueries);
        large_dim_l2_ground_truth_ =
            GenGroundTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::L2,
                           kLargeDimNumRows, kLargeDim, kLargeDimNumQueries);
        large_dim_ip_range_search_ground_truth_ =
            GenRangeSearchGrounTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::IP,
                                     kLargeDimNumRows, kLargeDim, kLargeDimNumQueries, kLargeDimIPRadiusLowBound, kLargeDimIPRadiusHighBound);
        large_dim_l2_range_search_ground_truth_ =
            GenRangeSearchGrounTruth(global_large_dim_raw_data_, global_large_dim_query_data_, knowhere::metric::L2,
                                     kLargeDimNumRows, kLargeDim, kLargeDimNumQueries, kLargeDimL2RadiusLowBound, kLargeDimL2RadiusHighBound);

        // prepare the dir
        ASSERT_TRUE(fs::create_directory(kDir));
        ASSERT_TRUE(fs::create_directory(kIpIndexDir));
        ASSERT_TRUE(fs::create_directory(kL2IndexDir));
        ASSERT_TRUE(fs::create_directory(kLargeDimL2IndexDir));
        ASSERT_TRUE(fs::create_directory(kLargeDimIpIndexDir));

        WriteRawDataToDisk(kRawDataPath, global_raw_data_, kNumRows, kDim);
        WriteRawDataToDisk(kLargeDimRawDataPath, global_large_dim_raw_data_, kLargeDimNumRows, kLargeDim);

        knowhere::Config cfg;
        knowhere::DiskANNBuildConfig::Set(cfg, build_conf);
        auto diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
            kIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        diskann_ip->BuildAll(nullptr, cfg);
        auto diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
            kL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        diskann_l2->BuildAll(nullptr, cfg);

        knowhere::Config large_dim_cfg;
        knowhere::DiskANNBuildConfig::Set(large_dim_cfg, large_dim_build_conf);
        auto large_dim_diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
            kLargeDimIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        large_dim_diskann_ip->BuildAll(nullptr, large_dim_cfg);
        auto large_dim_diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
            kLargeDimL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        large_dim_diskann_l2->BuildAll(nullptr, large_dim_cfg);
    }

    static void
    TearDownTestCase() {
        LOG_KNOWHERE_INFO_ << "Cleaning up the test environment for DiskANN Unittest.";
        delete[] global_raw_data_;
        delete[] global_large_dim_raw_data_;
        delete[] global_query_data_;
        delete[] global_large_dim_query_data_;
        // Clean up the dir

        fs::remove_all(kDir);
        fs::remove(kDir);
    }

 protected:
    void
    InitDiskANN() {
        std::string index_dir = "";
        if (metric_ == knowhere::metric::L2) {
            index_dir = is_large_dim_ ? kLargeDimL2IndexDir : kL2IndexDir;
        } else {
            index_dir = is_large_dim_ ? kLargeDimIpIndexDir : kIpIndexDir;
        }
        diskann = std::make_unique<knowhere::IndexDiskANN<float>>(index_dir + "/diskann", metric_,
                                                                  std::make_unique<knowhere::LocalFileManager>());
    }
    static float* global_raw_data_;
    static float* global_large_dim_raw_data_;
    static float* global_query_data_;
    static float* global_large_dim_query_data_;
    static GroundTruthPtr ip_ground_truth_;
    static GroundTruthPtr l2_ground_truth_;
    static GroundTruthPtr l2_range_search_ground_truth_;
    static GroundTruthPtr ip_range_search_ground_truth_;
    static GroundTruthPtr large_dim_ip_ground_truth_;
    static GroundTruthPtr large_dim_l2_ground_truth_;
    static GroundTruthPtr large_dim_l2_range_search_ground_truth_;
    static GroundTruthPtr large_dim_ip_range_search_ground_truth_;
    std::string metric_;
    bool is_large_dim_;
    uint32_t dim_;
    uint32_t num_rows_;
    uint32_t num_queries_;
    uint32_t big_k_;
    GroundTruthPtr ground_truth_;
    GroundTruthPtr range_search_ground_truth_;
    float* raw_data_;
    float* query_data_;
    float radius_low_bound_;
    float radius_high_bound_;
    knowhere::DiskANNQueryByRangeConfig range_search_conf_;
    std::unique_ptr<knowhere::VecIndex> diskann;
};

float* DiskANNTest::global_raw_data_ = nullptr;
float* DiskANNTest::global_large_dim_raw_data_ = nullptr;
float* DiskANNTest::global_query_data_ = nullptr;
float* DiskANNTest::global_large_dim_query_data_ = nullptr;
GroundTruthPtr DiskANNTest::ip_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::l2_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::ip_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_ip_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_l2_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_l2_range_search_ground_truth_ = nullptr;
GroundTruthPtr DiskANNTest::large_dim_ip_range_search_ground_truth_ = nullptr;

INSTANTIATE_TEST_CASE_P(DiskANNParameters, DiskANNTest,
                        Values(std::make_tuple(knowhere::metric::L2, true /* high-dimension */),
                               std::make_tuple(knowhere::metric::L2, false /* low-dimension */),
                               std::make_tuple(knowhere::metric::IP, true /* high-dimension */),
                               std::make_tuple(knowhere::metric::IP, false /* low-dimension */)));

TEST_P(DiskANNTest, bitset_view_test) {
    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test for knn search
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    // pick ids to mask
    auto ids = knowhere::GetDatasetIDs(result);
    std::unordered_set<int64_t> ids_to_mask;
    for (int32_t q = 0; q < num_queries_; ++q) {
        for (int32_t k = 0; k < kK; k += 2) {
            ids_to_mask.insert(ids[q * kK + k]);
        }
    }

    // create bitset view
    std::vector<uint8_t> knn_bitset_data;
    knn_bitset_data.resize(num_rows_ / 8);
    for (auto& id_to_mask : ids_to_mask) {
        set_bit(knn_bitset_data.data(), id_to_mask);
    }
    faiss::BitsetView knn_bitset(knn_bitset_data.data(), num_rows_);
    auto ground_truth = GenGroundTruth(raw_data_, query_data_, metric_, num_rows_, dim_, num_queries_, knn_bitset);

    // query with bitset view
    result = diskann->Query(data_set_ptr, cfg, knn_bitset);
    ids = knowhere::GetDatasetIDs(result);

    auto recall = CheckTopKRecall(ground_truth, ids, kK, num_queries_);
    EXPECT_GT(recall, 0.8);

    // test for range search
    cfg.clear();
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf_);
    result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);

    // pick ids to mask
    ids = knowhere::GetDatasetIDs(result);
    auto lims = knowhere::GetDatasetLims(result);
    ids_to_mask.clear();
    for (int32_t q = 0; q < num_queries_; ++q) {
        for (int32_t i = 0; i < lims[q + 1] - lims[q]; i += 5) {
            ids_to_mask.insert(ids[lims[q] + i]);
        }
    }

    // create bitset view
    std::vector<uint8_t> range_search_bitset_data;
    range_search_bitset_data.resize(num_rows_ / 8);
    for (auto& id_to_mask : ids_to_mask) {
        set_bit(range_search_bitset_data.data(), id_to_mask);
    }
    faiss::BitsetView range_search_bitset(range_search_bitset_data.data(), num_rows_);
    ground_truth = GenRangeSearchGrounTruth(raw_data_, query_data_, metric_, num_rows_, dim_, num_queries_,
                                            radius_low_bound_, radius_high_bound_, range_search_bitset);

    // query with bitset view
    result = diskann->QueryByRange(data_set_ptr, cfg, range_search_bitset);

    ids = knowhere::GetDatasetIDs(result);
    lims = knowhere::GetDatasetLims(result);

    recall = CheckRangeSearchRecall(ground_truth, ids, lims, num_queries_);
    EXPECT_GT(recall, 0.5);
}

TEST_P(DiskANNTest, meta_test) {
    EXPECT_THROW(diskann->Dim(), knowhere::KnowhereException);
    EXPECT_THROW(diskann->Count(), knowhere::KnowhereException);

    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    EXPECT_EQ(dim_, diskann->Dim());
    EXPECT_EQ(num_rows_, diskann->Count());
}

TEST_P(DiskANNTest, knn_search_test) {
    knowhere::Config cfg;
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
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
    auto diss = knowhere::GetDatasetDistance(result);

    auto recall = CheckTopKRecall(ground_truth_, ids, kK, num_queries_);
    EXPECT_GT(recall, 0.8);

    CheckDistanceError(raw_data_, query_data_, result, metric_, num_queries_, dim_, kK, num_rows_, is_large_dim_);
}

TEST_P(DiskANNTest, knn_search_with_accelerate_build_test) {
    if (is_large_dim_) {
        GTEST_SKIP() << "Skip build accelerate test for large dim.";
    }
    std::string index_dir =
        metric_ == knowhere::metric::L2 ? kDir + "/accelearte_build_l2_index" : kDir + "/accelearte_build_ip_index";
    ASSERT_TRUE(fs::create_directory(index_dir));

    auto accelerate_diskann = std::make_unique<knowhere::IndexDiskANN<float>>(
        index_dir + "/diskann", metric_, std::make_unique<knowhere::LocalFileManager>());

    // build
    knowhere::Config cfg;
    knowhere::DiskANNBuildConfig accelerate_build_conf = build_conf;
    accelerate_build_conf.accelerate_build = true;
    knowhere::DiskANNBuildConfig::Set(cfg, accelerate_build_conf);
    accelerate_diskann->BuildAll(nullptr, cfg);

    EXPECT_EQ(dim_, accelerate_diskann->Dim());
    EXPECT_EQ(num_rows_, accelerate_diskann->Count());

    // prepare
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);

    auto recall = CheckTopKRecall(ground_truth_, ids, kK, num_queries_);
    EXPECT_GT(recall, 0.8);

    CheckDistanceError(raw_data_, query_data_, result, metric_, num_queries_, dim_, kK, num_rows_, is_large_dim_);

    fs::remove_all(index_dir);
    fs::remove(index_dir);
}

TEST_P(DiskANNTest, merged_vamana_knn_search_test) {
    if (is_large_dim_) {
        GTEST_SKIP() << "Skip merge vamana test for large dim.";
    }
    std::string kMergedL2IndexDir = kDir + "/merged_l2_index";
    std::string kMergedIpIndexDir = kDir + "/merged_ip_index";

    float merged_build_ram_limit =
        static_cast<float>(num_rows_ * dim_) * sizeof(float) / static_cast<float>(1024 * 1024 * 1024) +
        std::numeric_limits<float>::epsilon();
    knowhere::DiskANNBuildConfig merged_vamana_build_conf{kRawDataPath, 50, 90, 0.2, merged_build_ram_limit, 4, 0};

    knowhere::Config merged_cfg;
    knowhere::DiskANNBuildConfig::Set(merged_cfg, merged_vamana_build_conf);

    if (metric_ == knowhere::metric::IP) {
        ASSERT_TRUE(fs::create_directory(kMergedIpIndexDir));
        auto merged_diskann_ip = std::make_unique<knowhere::IndexDiskANN<float>>(
            kMergedIpIndexDir + "/diskann", knowhere::metric::IP, std::make_unique<knowhere::LocalFileManager>());
        merged_diskann_ip->BuildAll(nullptr, merged_cfg);
    } else {
        ASSERT_TRUE(fs::create_directory(kMergedL2IndexDir));
        auto merged_diskann_l2 = std::make_unique<knowhere::IndexDiskANN<float>>(
            kMergedL2IndexDir + "/diskann", knowhere::metric::L2, std::make_unique<knowhere::LocalFileManager>());
        merged_diskann_l2->BuildAll(nullptr, merged_cfg);
    }
    std::string index_dir = metric_ == knowhere::metric::L2 ? kMergedL2IndexDir : kMergedIpIndexDir;
    std::unique_ptr<knowhere::VecIndex> merged_diskann = std::make_unique<knowhere::IndexDiskANN<float>>(
        index_dir + "/diskann", metric_, std::make_unique<knowhere::LocalFileManager>());
    knowhere::Config cfg;
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    // test query before preparation
    EXPECT_THROW(merged_diskann->Query(data_set_ptr, cfg, nullptr), knowhere::KnowhereException);

    // test preparation
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(merged_diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    auto result = merged_diskann->Query(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);

    auto recall = CheckTopKRecall(ground_truth_, ids, kK, num_queries_);
    EXPECT_GT(recall, 0.8);

    CheckDistanceError(raw_data_, query_data_, result, metric_, num_queries_, dim_, kK, num_rows_, is_large_dim_);

    fs::remove_all(index_dir);
    fs::remove(index_dir);
}

TEST_P(DiskANNTest, knn_search_big_k_test) {
    // gen new ground truth
    auto ground_truth = std::make_shared<GroundTruth>();
    ground_truth->resize(num_queries_);
    for (uint32_t query_index = 0; query_index < num_queries_; ++query_index) {
        for (uint32_t row = 0; row < num_rows_; ++row) {
            (ground_truth->at(query_index)).emplace_back(row);
        }
        for (uint32_t row = num_rows_; row < kBigK; ++row) {
            (ground_truth->at(query_index)).emplace_back(-1);
        }
    }

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
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);
    auto recall = CheckTopKRecall(ground_truth, ids, kBigK, num_queries_);
    EXPECT_GT(recall, 0.8);

    EXPECT_EQ(diskann->Count(), num_rows_);
    CheckDistanceError(raw_data_, query_data_, result, metric_, num_queries_, dim_, kBigK, num_rows_, is_large_dim_);
}

TEST_P(DiskANNTest, range_search_test) {
    knowhere::Config cfg;
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf_);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    // test query before preparation
    EXPECT_THROW(diskann->QueryByRange(data_set_ptr, cfg, nullptr), knowhere::KnowhereException);

    // test preparation
    cfg.clear();
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query by range
    cfg.clear();
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf_);
    auto result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);

    auto ids = knowhere::GetDatasetIDs(result);
    auto lims = knowhere::GetDatasetLims(result);

    auto recall = CheckRangeSearchRecall(range_search_ground_truth_, ids, lims, num_queries_);
    auto expected_recall = 0.8;
    EXPECT_GT(recall, expected_recall);
}

TEST_P(DiskANNTest, cached_warmup_test) {
    knowhere::Config cfg;

    // search cache + warmup preparation
    knowhere::DiskANNPrepareConfig prep_conf_to_test = prep_conf;
    prep_conf_to_test.warm_up = true;
    prep_conf_to_test.search_cache_budget_gb = 0.00001;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf_to_test);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);
    auto ids = knowhere::GetDatasetIDs(result);

    auto recall = CheckTopKRecall(ground_truth_, ids, kK, num_queries_);
    EXPECT_GT(recall, 0.8);
    CheckDistanceError(raw_data_, query_data_, result, metric_, num_queries_, dim_, kK, num_rows_, is_large_dim_);
    // bfs cache + warmup preparation
    InitDiskANN();
    cfg.clear();
    prep_conf_to_test.use_bfs_cache = true;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf_to_test);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test query by range
    cfg.clear();
    knowhere::DiskANNQueryByRangeConfig::Set(cfg, range_search_conf_);
    data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);

    ids = knowhere::GetDatasetIDs(result);
    auto lims = knowhere::GetDatasetLims(result);

    recall = CheckRangeSearchRecall(range_search_ground_truth_, ids, lims, num_queries_);
    auto expected_recall = 0.8;
    EXPECT_GT(recall, expected_recall);
}

TEST_P(DiskANNTest, config_test) {
    // build config
    knowhere::DiskANNBuildConfig build_conf_to_test = build_conf;
    build_conf_to_test.max_degree = 0;
    CheckConfigError<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test.max_degree = 513;
    CheckConfigError<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test = build_conf;
    build_conf_to_test.num_threads = 0;
    CheckConfigError<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    build_conf_to_test.num_threads = 129;
    CheckConfigError<knowhere::DiskANNBuildConfig>(build_conf_to_test);

    // query config
    knowhere::DiskANNQueryConfig query_conf_to_test = query_conf;
    query_conf_to_test.k = 10;
    query_conf_to_test.search_list_size = 9;
    CheckConfigError<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    query_conf_to_test = query_conf;
    query_conf_to_test.beamwidth = 0;
    CheckConfigError<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    query_conf_to_test.beamwidth = 129;
    CheckConfigError<knowhere::DiskANNQueryConfig>(query_conf_to_test);

    // query by range config
    knowhere::DiskANNQueryByRangeConfig range_search_conf_to_test = l2_range_search_conf;
    range_search_conf_to_test.min_k = 10;
    range_search_conf_to_test.max_k = 9;
    CheckConfigError<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);

    range_search_conf_to_test = l2_range_search_conf;
    range_search_conf_to_test.beamwidth = 0;
    CheckConfigError<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);

    range_search_conf_to_test.beamwidth = 129;
    CheckConfigError<knowhere::DiskANNQueryByRangeConfig>(range_search_conf_to_test);
}

TEST_P(DiskANNTest, build_config_test) {
    std::string test_dir = kDir + "/build_config_test";
    std::string test_data_path = test_dir + "/test_raw_data";
    std::string test_index_dir = test_dir + "/test_index";
    ASSERT_TRUE(fs::create_directory(test_dir));

    knowhere::Config illegal_cfg;
    knowhere::DiskANNBuildConfig illegal_build_config;

    // data size != file size
    {
        fs::remove(test_data_path);
        fs::remove(test_index_dir);
        fs::copy_file(kRawDataPath, test_data_path);
        std::ofstream writer(test_data_path, std::ios::binary);
        writer.seekp(0, std::ios::end);
        writer << "end of the file";
        writer.flush();
        writer.close();

        illegal_build_config = build_conf;  // init
        illegal_build_config.data_path = test_data_path;
        illegal_cfg.clear();
        knowhere::DiskANNBuildConfig::Set(illegal_cfg, illegal_build_config);
        knowhere::IndexDiskANN<float> test_diskann(test_index_dir, metric_,
                                                   std::make_unique<knowhere::LocalFileManager>());
        EXPECT_THROW(test_diskann.BuildAll(nullptr, illegal_cfg), knowhere::KnowhereException);
    }
    // raw data file not exist
    {
        fs::remove(test_data_path);
        fs::remove(test_index_dir);

        illegal_build_config = build_conf;
        illegal_build_config.data_path = test_data_path;
        illegal_cfg.clear();
        knowhere::DiskANNBuildConfig::Set(illegal_cfg, illegal_build_config);
        knowhere::IndexDiskANN<float> test_diskann(test_index_dir, metric_,
                                                   std::make_unique<knowhere::LocalFileManager>());
        EXPECT_THROW(test_diskann.BuildAll(nullptr, illegal_cfg), knowhere::KnowhereException);
    }
    // Re-build the index on the already built index files
    {
        if (metric_ == knowhere::metric::IP) {
            auto illegal_diskann_ip = knowhere::IndexDiskANN<float>(kIpIndexDir + "/diskann", knowhere::metric::IP,
                                                                    std::make_unique<knowhere::LocalFileManager>());
            illegal_cfg.clear();
            knowhere::DiskANNBuildConfig::Set(illegal_cfg, build_conf);
            EXPECT_THROW(illegal_diskann_ip.BuildAll(nullptr, illegal_cfg), knowhere::KnowhereException);
        } else if (metric_ == knowhere::metric::L2) {
            auto illegal_diskann_l2 = knowhere::IndexDiskANN<float>(kL2IndexDir + "/diskann", knowhere::metric::L2,
                                                                    std::make_unique<knowhere::LocalFileManager>());
            illegal_cfg.clear();
            knowhere::DiskANNBuildConfig::Set(illegal_cfg, build_conf);
            EXPECT_THROW(illegal_diskann_l2.BuildAll(nullptr, illegal_cfg), knowhere::KnowhereException);
        }
    }
    fs::remove_all(test_dir);
    fs::remove(test_dir);
}

TEST_P(DiskANNTest, generate_cache_list_test) {
    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig prep_conf_to_test = prep_conf;
    prep_conf_to_test.use_bfs_cache = false;
    prep_conf_to_test.search_cache_budget_gb = 3.0;

    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf_to_test);
    EXPECT_THROW(diskann->Prepare(cfg), knowhere::KnowhereException);
}

TEST_P(DiskANNTest, get_meta) {
    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    knowhere::DiskANNBuildConfig::Set(cfg, build_conf);
    auto result = diskann->GetIndexMeta(cfg);

    auto json_info = knowhere::GetDatasetJsonInfo(result);
    auto json_id_set = knowhere::GetDatasetJsonIdSet(result);
    //std::cout << json_info << std::endl;
    std::cout << "json_info size = " << json_info.size() << std::endl;
    std::cout << "json_id_set size = " << json_id_set.size() << std::endl;

    // check DiskANNMeta
    knowhere::feder::diskann::DiskANNMeta meta;
    knowhere::Config j1 = nlohmann::json::parse(json_info);
    ASSERT_NO_THROW(nlohmann::from_json(j1, meta));
    //std::cout << j1.dump(4) << std::endl;

    // check IDSet
    std::unordered_set<int64_t> id_set;
    knowhere::Config j2 = nlohmann::json::parse(json_id_set);
    ASSERT_NO_THROW(nlohmann::from_json(j2, id_set));
    //std::cout << j2.dump(4) << std::endl;
}

TEST_P(DiskANNTest, trace_visit) {
    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
    EXPECT_TRUE(diskann->Prepare(cfg));

    // test for knn search
    cfg.clear();
    knowhere::DiskANNQueryConfig::Set(cfg, query_conf);

    knowhere::SetMetaTraceVisit(cfg, true);
    knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(num_queries_, dim_, (void*)query_data_);
    ASSERT_ANY_THROW(diskann->Query(data_set_ptr, cfg, nullptr));

    data_set_ptr = knowhere::GenDataset(1, dim_, (void*)query_data_);
    auto result = diskann->Query(data_set_ptr, cfg, nullptr);

    auto json_info = knowhere::GetDatasetJsonInfo(result);
    auto json_id_set = knowhere::GetDatasetJsonIdSet(result);
    //std::cout << json_info << std::endl;
    std::cout << "json_info size = " << json_info.size() << std::endl;
    std::cout << "json_id_set size = " << json_id_set.size() << std::endl;

    // check DiskANNVisitInfo
    knowhere::feder::diskann::DiskANNVisitInfo visit_info;
    knowhere::Config j1 = nlohmann::json::parse(json_info);
    ASSERT_NO_THROW(nlohmann::from_json(j1, visit_info));
    //std::cout << j1.dump(4) << std::endl;

    // check IDSet
    std::unordered_set<int64_t> id_set;
    knowhere::Config j2 = nlohmann::json::parse(json_id_set);
    ASSERT_NO_THROW(nlohmann::from_json(j2, id_set));
    //std::cout << j2.dump(4) << std::endl;
}
