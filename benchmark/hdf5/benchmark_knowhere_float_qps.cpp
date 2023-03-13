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

#include <vector>

#include "benchmark_knowhere.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"

const int32_t GPU_DEVICE_ID = 0;
const int32_t CLIENT_NUM = 1;

class Benchmark_knowhere_float_qps : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int32_t>();

        auto find_smallest_nprobe = [&](float expected_recall) -> int32_t {
            int32_t nprobe = 1;
            float recall;
            while (nprobe <= NLIST_) {
                conf[knowhere::indexparam::NPROBE] = nprobe;
                conf[knowhere::meta::TOPK] = gt_k_;
                auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);

                auto result = index_.Search(*ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, gt_k_);
                printf("\n[%0.3f s] iterate IVF param for recall %.4f: nlist=%d, nprobe=%d, k=%d, R@=%.4f\n",
                       get_time_diff(), expected_recall, nlist, nprobe, gt_k_, recall);
                if (recall >= expected_recall) {
                    break;
                }
                nprobe *= 2;
            }
            return std::min(nprobe, NLIST_);
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto nprobe = find_smallest_nprobe(expected_recall);
            conf[knowhere::indexparam::NPROBE] = nprobe;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s | nlist=%d, nprobe=%d, k=%d, R@=%.4f\n", get_time_diff(),
                   ann_test_name_.c_str(), index_type_.c_str(), nlist, nprobe, topk_, expected_recall);
            printf("================================================================================\n");
            CALC_TIME_SPAN(task(conf, CLIENT_NUM, nq_));
            printf("  client_num = %d, elapse = %6.3fs, QPS = %.3f\n", CLIENT_NUM, t_diff, nq_ * CLIENT_NUM / t_diff);
            std::fflush(stdout);
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

    void
    test_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto M = conf[knowhere::indexparam::HNSW_M].get<int32_t>();
        auto efConstruction = conf[knowhere::indexparam::EFCONSTRUCTION].get<int32_t>();

        auto find_smallest_ef = [&](float expected_recall) -> int32_t {
            int32_t ef = 128, ef_max = 512;
            float recall;
            while (ef <= ef_max) {
                conf[knowhere::indexparam::EF] = ef;
                conf[knowhere::meta::TOPK] = gt_k_;
                auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);

                auto result = index_.Search(*ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, gt_k_);
                printf("\n[%0.3f s] iterate HNSW param for expected recall %.4f: ef=%d, R@=%.4f\n", get_time_diff(),
                       expected_recall, ef, recall);
                if (recall >= expected_recall) {
                    break;
                }
                ef *= 2;
            }
            return std::min(ef, ef_max);
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto ef = find_smallest_ef(expected_recall);
            conf[knowhere::indexparam::EF] = ef;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s | M=%d | efConstruction=%d, ef=%d, k=%d, R@=%.4f\n", get_time_diff(),
                   ann_test_name_.c_str(), index_type_.c_str(), M, efConstruction, ef, topk_, expected_recall);
            printf("================================================================================\n");
            CALC_TIME_SPAN(task(conf, CLIENT_NUM, nq_));
            printf("  client_num = %d, elapse = %6.3fs, QPS = %.3f\n", CLIENT_NUM, t_diff, nq_ * CLIENT_NUM / t_diff);
            std::fflush(stdout);
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

 private:
    void
    task(const knowhere::Json& conf, int32_t worker_num, int32_t nq) {
        auto worker = [&](int32_t client_no, int32_t nq) {
            for (int32_t i = 0; i < nq; i++) {
                knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(1, dim_, (const float*)xq_ + i * dim_);
                indices_[client_no].Search(*ds_ptr, conf, nullptr);
            }
        };

        std::vector<std::thread> thread_vector(worker_num);
        for (int32_t i = 0; i < worker_num; i++) {
            thread_vector[i] = std::thread(worker, i, nq);
        }
        for (int32_t i = 0; i < worker_num; i++) {
            thread_vector[i].join();
        }
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
        metric_type_ = (metric_str_ == METRIC_IP_STR) ? knowhere::metric::IP : knowhere::metric::L2;
        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
#ifdef USE_CUDA
        knowhere::KnowhereConfig::InitGPUResource(GPU_DEVICE_ID, CLIENT_NUM);
        cfg_[knowhere::meta::DEVICE_ID] = GPU_DEVICE_ID;
#endif
    }

    void
    TearDown() override {
        free_all();
#ifdef USE_CUDA
        knowhere::KnowhereConfig::FreeGPUResource();
#endif
    }

 protected:
    const int32_t topk_ = 10;
    const std::vector<float> EXPECTED_RECALLs_ = {0.9, 0.95};

    // IVF index params
    const int32_t NLIST_ = 1024;

    // HNSW index params
    const int32_t M_ = 16;
    const int32_t EFCON_ = 200;
};

TEST_F(Benchmark_knowhere_float_qps, TEST_IVF_FLAT_NM) {
#ifdef USE_CUDA
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT;
#else
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
#endif

    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::NLIST] = NLIST_;

    std::string index_file_name = get_index_name({NLIST_});

    // IVFFLAT_NM should load raw data
    knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
    bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
    bin->size = dim_ * nb_ * sizeof(float);
    binary_set_.Append("RAW_DATA", bin);

    for (int i = 0; i < CLIENT_NUM; i++) {
        indices_.emplace_back(create_index(index_file_name, conf));
        indices_.back().Deserialize(binary_set_);
    }
    binary_set_.clear();
    test_ivf(conf);
}

TEST_F(Benchmark_knowhere_float_qps, TEST_IVF_SQ8) {
#ifdef USE_CUDA
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8;
#else
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
#endif

    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::NLIST] = NLIST_;

    std::string index_file_name = get_index_name({NLIST_});

    for (int i = 0; i < CLIENT_NUM; i++) {
        indices_.emplace_back(create_index(index_file_name, conf));
        indices_.back().Deserialize(binary_set_);
    }
    binary_set_.clear();
    test_ivf(conf);
}

TEST_F(Benchmark_knowhere_float_qps, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::HNSW_M] = M_;
    conf[knowhere::indexparam::EFCONSTRUCTION] = EFCON_;

    std::string index_file_name = get_index_name({M_, EFCON_});

    for (int i = 0; i < CLIENT_NUM; i++) {
        indices_.emplace_back(create_index(index_file_name, conf));
        indices_.back().Deserialize(binary_set_);
    }
    binary_set_.clear();
    test_hnsw(conf);
}
