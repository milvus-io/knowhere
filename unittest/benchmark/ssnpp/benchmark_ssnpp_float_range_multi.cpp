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
#include "unittest/range_utils.h"

class Benchmark_ssnpp_float_range_multi : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_ivf_range(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            double span = 0.0;
            int64_t hits = 0, total_gt = 0;
            for (int i = 0; i < nq_; i += nq_ / NQ_) {
                knowhere::SetMetaRadius(conf, std::sqrt(gt_radius_[i]));
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(1, dim_, (const float*)xq_ + (i * dim_));
                total_gt += gt_lims_[i + 1] - gt_lims_[i];
                CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
                span += t_diff;
                auto ids = knowhere::GetDatasetIDs(result);
                auto lims = knowhere::GetDatasetLims(result);
                hits += CalcHits(ids, lims, i, 1);
            }
            printf("  nprobe = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f\n", nprobe, NQ_, span,
                   (hits * 1.0f / total_gt));
            fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_hnsw_range(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto M = knowhere::GetIndexParamHNSWM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(),
               ann_test_name_.c_str(), index_type_.c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            knowhere::SetIndexParamEf(conf, ef);
            double span = 0.0;
            int32_t hits = 0, total_gt = 0;
            for (int i = 0; i < nq_; i += nq_ / NQ_) {
                knowhere::SetMetaRadius(conf, std::sqrt(gt_radius_[i]));
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(1, dim_, (const float*)xq_ + (i * dim_));
                total_gt += gt_lims_[i + 1] - gt_lims_[i];
                CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
                span += t_diff;
                auto ids = knowhere::GetDatasetIDs(result);
                auto lims = knowhere::GetDatasetLims(result);
                hits += CalcHits(ids, lims, i, 1);
            }
            printf("  ef = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, NQ_, span,
                   (hits * 1.0f / total_gt));
            fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("ssnpp-256-euclidean-range");

        // load base dataset
        load_bin<float>("FB_ssnpp_database.10M.fbin", xb_, nb_, dim_);

        // load query dataset
        int32_t dim;
        load_bin<float>("FB_ssnpp_public_queries.fbin", xq_, nq_, dim);
        assert(dim_ == dim);

        int32_t gt_num;
        // std::vector<std::string> gt_array = {"SSNPP-radius-exponential", "SSNPP-gt-exponential"};
        std::vector<std::string> gt_array = {"SSNPP-radius-norm", "SSNPP-gt-norm"};
        // std::vector<std::string> gt_array = {"SSNPP-radius-poisson", "SSNPP-gt-poisson"};
        // std::vector<std::string> gt_array = {"SSNPP-radius-uniform", "SSNPP-gt-uniform"};
        // load ground truth radius
        load_range_radius(gt_array[0], gt_radius_, gt_num);
        assert(gt_num == nq_);

        // load ground truth ids
        load_range_truthset(gt_array[1], gt_ids_, gt_lims_, gt_num);
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
    const int32_t NQ_ = 1000;

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
TEST_F(Benchmark_ssnpp_float_range, TEST_CREATE_HDF5) {
    // set this radius to get about 1M result dataset for 10k nq
    const float radius = 310.221;

    assert(dim_ == 256);
    assert(nq_ == 100000);
    hdf5_write_range<false>("ssnpp-256-euclidean-range.hdf5", dim_, xb_, nb_, xq_, nq_, radius,
                            gt_lims_, gt_ids_, gt_dist_);
}
#endif

TEST_F(Benchmark_ssnpp_float_range_multi, TEST_IVF_FLAT_NM) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_index(index_file_name, conf);

        // IVFFLAT_NM should load raw data
        knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
        bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
        bin->size = (size_t)dim_ * nb_ * sizeof(float);
        binary_set_.Append(RAW_DATA, bin);

        index_->Load(binary_set_);
        binary_set_.clear();
        test_ivf_range(conf);
    }
}

TEST_F(Benchmark_ssnpp_float_range_multi, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_index(index_file_name, conf);
        index_->Load(binary_set_);
        binary_set_.clear();
        test_ivf_range(conf);
    }
}

TEST_F(Benchmark_ssnpp_float_range_multi, TEST_HNSW) {
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
            test_hnsw_range(conf);
        }
    }
}
