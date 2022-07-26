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

#include "unittest/benchmark/benchmark_knowhere.h"
#include "unittest/range_utils.h"

class Benchmark_knowhere_binary_range : public Benchmark_knowhere {
 public:
    void
    test_binary_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto radius = knowhere::GetMetaRadius(conf);

        printf("\n[%0.3f s] %s | %s, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), radius);
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
            CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
            auto ids = knowhere::GetDatasetIDs(result);
            auto distances = knowhere::GetDatasetDistance(result);
            auto lims = knowhere::GetDatasetLims(result);
            CheckDistance(metric_type_, ids, distances, lims, nq);
            float recall = CalcRecall(ids, lims, nq);
            float accuracy = CalcAccuracy(ids, lims, nq);
            printf("  nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f\n", nq, t_diff, recall, accuracy);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_binary_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);
        auto radius = knowhere::GetMetaRadius(conf);

        printf("\n[%0.3f s] %s | %s | nlist=%ld, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), nlist, radius);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                CALC_TIME_SPAN(auto result = index_->QueryByRange(ds_ptr, conf, nullptr));
                auto ids = knowhere::GetDatasetIDs(result);
                auto lims = knowhere::GetDatasetLims(result);
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  nprobe = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f\n",
                       nprobe, nq, t_diff, recall, accuracy);
            }
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
        set_ann_test_name("sift-4096-hamming-range");
        parse_ann_range_test_name();
        load_hdf5_data_range<true>();
#endif

        assert(metric_str_ == METRIC_HAM_STR || metric_str_ == METRIC_JAC_STR || metric_str_ == METRIC_TAN_STR);
        metric_type_ = (metric_str_ == METRIC_HAM_STR)   ? knowhere::metric::HAMMING
                       : (metric_str_ == METRIC_JAC_STR) ? knowhere::metric::JACCARD
                                                         : knowhere::metric::TANIMOTO;
        knowhere::SetMetaMetricType(cfg_, metric_type_);
        knowhere::SetMetaRadius(cfg_, *gt_radius_);
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
};

// This testcase can be used to generate binary sift1m HDF5 file
// Following these steps:
//   1. set_ann_test_name("sift-128-euclidean")
//   2. use parse_ann_test_name() and load_hdf5_data<false>()
//   3. set expected distance calculation API for RunRangeSearchBF
//   4. specify the hdf5 file name to generate
//   5. run this testcase
#if 0
TEST_F(Benchmark_knowhere_binary_range, TEST_CREATE_HDF5) {
    // use sift1m data as binary data
    dim_ *= 32;

    // set this radius to get about 1M result dataset for 10k nq
    const float radius = 291.0;

    std::vector<int64_t> golden_labels;
    std::vector<float> golden_distances;
    std::vector<size_t> golden_lims;
    RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, metric_type_,
                                        (const uint8_t*)xb_, nb_, (const uint8_t*)xq_, nq_, dim_, radius, nullptr);

    // convert golden_lims and golden_ids to int32
    std::vector<int32_t> golden_lims_int(nq_ + 1);
    for (int32_t i = 0; i <= nq_; i++) {
        golden_lims_int[i] = golden_lims[i];
    }

    auto elem_cnt = golden_lims[nq_];
    std::vector<int32_t> golden_ids_int(elem_cnt);
    for (int32_t i = 0; i < elem_cnt; i++) {
        golden_ids_int[i] = golden_labels[i];
    }

    assert(dim_ == 4096);
    assert(nq_ == 10000);
    hdf5_write_range<true>("sift-4096-hamming-range.hdf5", dim_/32, xb_, nb_, xq_, nq_, radius,
                           golden_lims_int.data(), golden_ids_int.data(), golden_distances.data());
}
#endif

TEST_F(Benchmark_knowhere_binary_range, TEST_BINARY_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_binary_idmap(conf);
}

TEST_F(Benchmark_knowhere_binary_range, TEST_BINARY_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_cpu_index(index_file_name, conf);
        index_->Load(binary_set_);
        binary_set_.clear();
        test_binary_ivf(conf);
    }
}
