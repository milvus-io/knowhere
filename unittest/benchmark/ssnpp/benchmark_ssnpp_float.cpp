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

class Benchmark_ssnpp_float : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nprobe, nq, k, t_diff,
                           recall);
                    fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_hnsw(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto M = knowhere::GetIndexParamHNSWM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            knowhere::SetIndexParamEf(conf, ef);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, nq, k, t_diff, recall);
                    fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("ssnpp-256-euclidean");

        // load base dataset
        load_bin<float>("FB_ssnpp_database.1M.fbin", xb_, nb_, dim_);

        // load query dataset
        int32_t dim;
        load_bin<float>("FB_ssnpp_public_queries.fbin", xq_, nq_, dim);
        assert(dim_ == dim);

        int32_t gt_num;
        // load ground truth ids
        load_truthset("ssnpp-1M-gt", gt_ids_, gt_dist_, gt_num);
        assert(gt_num == nq_);

        metric_type_ = knowhere::metric::L2;

        knowhere::SetMetaMetricType(cfg_, metric_type_);
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        printf("faiss::distance_compute_blas_threshold: %ld\n", knowhere::KnowhereConfig::GetBlasThreshold());
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {100};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};
    const std::vector<int32_t> NPROBEs_ = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {200};
    const std::vector<int32_t> EFs_ = {16, 32, 64, 128, 256, 512};
};

// This testcase can be used to generate HDF5 file
#if 0
TEST_F(Benchmark_ssnpp_float, TEST_CREATE_HDF5) {
    assert(dim_ == 256);
    assert(nq_ == 100000);
    hdf5_write<false>("ssnpp-256-euclidean.hdf5", dim_, gt_k_, xb_, nb_, xq_, nq_, gt_ids_, gt_dist_);
}
#endif

TEST_F(Benchmark_ssnpp_float, TEST_IVF_FLAT_NM) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        knowhere::SetIndexParamNlist(conf, nlist);

        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);

        // IVFFLAT_NM should load raw data
        knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
        bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
        bin->size = dim_ * nb_ * sizeof(float);
        binary_set_.Append(RAW_DATA, bin);

        index_->Load(binary_set_);
        binary_set_.clear();
        test_ivf(conf);
    }
}

TEST_F(Benchmark_ssnpp_float, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Config conf = cfg_;
    for (auto M : HNSW_Ms_) {
        knowhere::SetIndexParamHNSWM(conf, M);
        for (auto efc : EFCONs_) {
            knowhere::SetIndexParamEfConstruction(conf, efc);

            std::string index_file_name = get_index_name({M, efc});
            create_index(index_file_name, conf);
            index_->Load(binary_set_);
            binary_set_.clear();
            test_hnsw(conf);
        }
    }
}
