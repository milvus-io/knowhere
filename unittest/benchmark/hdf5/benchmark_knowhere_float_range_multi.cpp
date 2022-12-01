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

class Benchmark_knowhere_float_range_multi : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        knowhere::SetMetaRadiusLowBound(conf, 0.0f);
        printf("\n[%0.3f s] %s | %s\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        double span = 0.0;
        int64_t hits = 0;
        for (int32_t i = 0; i < nq_; i++) {
            knowhere::SetMetaRadiusHighBound(conf, gt_radius_[i]);
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(1, dim_, (const float*)xq_ + (i * dim_));
            CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
            span += t_diff;
            auto ids = knowhere::GetDatasetIDs(result);
            auto lims = knowhere::GetDatasetLims(result);
            hits += CalcHits(ids, lims, i, 1);
        }
        printf("  nq = %4d, elapse = %6.3fs, R@ = %.4f\n", nq_, span, (hits * 1.0f) / gt_lims_[nq_]);
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        knowhere::SetMetaRadiusLowBound(conf, 0.0f);
        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            double span = 0.0;
            int64_t hits = 0;
            for (int32_t i = 0; i < nq_; i++) {
                knowhere::SetMetaRadiusHighBound(conf, gt_radius_[i]);
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(1, dim_, (const float*)xq_ + (i * dim_));
                CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
                span += t_diff;
                auto ids = knowhere::GetDatasetIDs(result);
                auto lims = knowhere::GetDatasetLims(result);
                hits += CalcHits(ids, lims, i, 1);
            }
            printf("  nprobe = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f\n", nprobe, nq_, span,
                   (hits * 1.0f) / gt_lims_[nq_]);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_hnsw(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto M = knowhere::GetIndexParamHNSWM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        knowhere::SetMetaRadiusLowBound(conf, 0.0f);
        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(),
               ann_test_name_.c_str(), index_type_.c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            knowhere::SetIndexParamEf(conf, ef);
            double span = 0.0;
            int64_t hits = 0;
            for (int32_t i = 0; i < nq_; i++) {
                knowhere::SetMetaRadiusHighBound(conf, gt_radius_[i]);
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(1, dim_, (const float*)xq_ + (i * dim_));
                CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
                span += t_diff;
                auto ids = knowhere::GetDatasetIDs(result);
                auto lims = knowhere::GetDatasetLims(result);
                hits += CalcHits(ids, lims, i, 1);
            }
            printf("  ef = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, nq_, span,
                   (hits * 1.0f) / gt_lims_[nq_]);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
#if 0  // used when create range sift HDF5
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();
#else
        set_ann_test_name("sift-128-euclidean-range-multi");
        parse_ann_test_name_with_range_multi();
        load_hdf5_data_range_multi<false>();
#endif

        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
        metric_type_ = (metric_str_ == METRIC_IP_STR) ? knowhere::metric::IP : knowhere::metric::L2;
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
// Following these steps:
//   1. set_ann_test_name, eg. "sift-128-euclidean" or "glove-200-angular"
//   2. use parse_ann_test_name() and load_hdf5_data<false>()
//   3. use the last element in the gt_dist_ as its radius for each nq
//   4. specify the hdf5 file name to generate
//   5. run this testcase
#if 0
TEST_F(Benchmark_knowhere_float_range_multi, TEST_CREATE_HDF5_WITH_MULTI_RADIUS) {
    std::vector<float> golden_radius(nq_);
    for (int32_t i = 0; i < nq_; i++) {
        golden_radius[i] = std::pow(gt_dist_[(i + 1) * gt_k_ - 1], 2.0) + 0.01;
    }

    std::vector<int32_t> golden_lims(nq_ + 1);
    for (int32_t i = 0; i <= nq_; i++) {
        golden_lims[i] = i * gt_k_;
    }

    for (int32_t i = 0; i < nq_ * gt_k_; i++) {
        gt_dist_[i] = std::pow(gt_dist_[i], 2.0);
    }

    assert(dim_ == 128);
    assert(nq_ == 10000);
    hdf5_write_range<false>("sift-128-euclidean-range-multi.hdf5", dim_, xb_, nb_, xq_, nq_,
                            golden_radius.data(), golden_lims.data(), gt_ids_, gt_dist_);
}
#endif

TEST_F(Benchmark_knowhere_float_range_multi, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_idmap(conf);
}

TEST_F(Benchmark_knowhere_float_range_multi, TEST_IVF_FLAT_NM) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
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

TEST_F(Benchmark_knowhere_float_range_multi, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_index(index_file_name, conf);
        index_->Load(binary_set_);
        binary_set_.clear();
        test_ivf(conf);
    }
}

TEST_F(Benchmark_knowhere_float_range_multi, TEST_IVF_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamNbits(conf, NBITS_);
    for (auto m : Ms_) {
        knowhere::SetIndexParamM(conf, m);
        for (auto nlist : NLISTs_) {
            std::string index_file_name = get_index_name({nlist, m});
            knowhere::SetIndexParamNlist(conf, nlist);
            create_index(index_file_name, conf);
            index_->Load(binary_set_);
            binary_set_.clear();
            test_ivf(conf);
        }
    }
}

TEST_F(Benchmark_knowhere_float_range_multi, TEST_HNSW) {
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
