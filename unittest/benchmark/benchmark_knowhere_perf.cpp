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

class Benchmark_knowhere_perf : public Benchmark_knowhere {
 public:
    void
    test_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        int32_t no = 0;
        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str());
        printf("================================================================================\n");
        for (int32_t i = 0; i + NQ_STEP_ <= GT_NQ_; i = (i + NQ_STEP_) % GT_NQ_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(NQ_STEP_, dim_, (const float*)xq_ + (i * dim_));
            for (auto k : TOPKs_) {
                knowhere::SetMetaTopk(conf, k);
                CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                auto ids = knowhere::GetDatasetIDs(result);
                float recall = CalcRecall(ids, i, NQ_STEP_, k);
                printf("  No.%4d: nq = [%4d, %4d), k = %4d, elapse = %6.3fs, R@ = %.4f\n", no++, i, i + NQ_STEP_, k,
                       t_diff, recall);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_ivf(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        int32_t no = 0;
        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), nlist);
        printf("================================================================================\n");
        for (int32_t i = 0; i + NQ_STEP_ <= GT_NQ_; i = (i + NQ_STEP_) % GT_NQ_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(NQ_STEP_, dim_, (const float*)xq_ + (i * dim_));
            for (auto nprobe : NPROBEs_) {
                knowhere::SetIndexParamNprobe(conf, nprobe);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, i, NQ_STEP_, k);
                    printf("  No.%4d: nprobe = %4d, nq = [%4d, %4d), k = %4d, elapse = %6.3fs, R@ = %.4f\n", no++,
                           nprobe, i, i + NQ_STEP_, k, t_diff, recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_hnsw(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto M = knowhere::GetIndexParamHNSWM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        int32_t no = 0;
        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (int32_t i = 0; i + NQ_STEP_ <= GT_NQ_; i = (i + NQ_STEP_) % GT_NQ_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(NQ_STEP_, dim_, (const float*)xq_ + (i * dim_));
            for (auto ef : EFs_) {
                knowhere::SetIndexParamEf(conf, ef);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, i, NQ_STEP_, k);
                    printf("  No.%4d: ef = %4d, nq = [%4d, %4d), k = %4d, elapse = %6.3fs, R@ = %.4f\n", no++, ef, i,
                           i + NQ_STEP_, k, t_diff, recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void
    test_annoy(const knowhere::Config& cfg) {
        auto conf = cfg;
        auto n_trees = knowhere::GetIndexParamNtrees(conf);

        int32_t no = 0;
        printf("\n[%0.3f s] %s | %s | n_trees=%ld \n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str(), n_trees);
        printf("================================================================================\n");
        for (int32_t i = 0; i + NQ_STEP_ <= GT_NQ_; i = (i + NQ_STEP_) % GT_NQ_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(NQ_STEP_, dim_, (const float*)xq_ + (i * dim_));
            for (auto sk : SEARCH_Ks_) {
                knowhere::SetIndexParamSearchK(conf, sk);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);
                    CALC_TIME_SPAN(auto result = index_->Query(ds_ptr, conf, nullptr));
                    auto ids = knowhere::GetDatasetIDs(result);
                    float recall = CalcRecall(ids, i, NQ_STEP_, k);
                    printf("  No.%4d: search_k = %4d, nq = [%4d, %4d), k = %4d, elapse = %6.3fs, R@ = %.4f\n", no++, sk,
                           i, i + NQ_STEP_, k, t_diff, recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
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
        knowhere::SetMetaMetricType(cfg_, metric_type_);
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const int32_t GT_NQ_ = 10000;
    const int32_t NQ_STEP_ = 10;
    const std::vector<int32_t> TOPKs_ = {100};

    // IVF index params
    const int32_t NLIST_ = 1024;
    const std::vector<int32_t> NPROBEs_ = {16};

    // HNSW index params
    const int32_t M_ = 16;
    const int32_t EFCON_ = 200;
    const std::vector<int32_t> EFs_ = {64};

    // ANNOY index params
    const int32_t N_TREE_ = 8;
    const std::vector<int32_t> SEARCH_Ks_ = {100};
};

TEST_F(Benchmark_knowhere_perf, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});

    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_idmap(conf);
}

TEST_F(Benchmark_knowhere_perf, TEST_IVFFLAT_NM) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamNlist(conf, NLIST_);
    std::string index_file_name = get_index_name({NLIST_});

    create_cpu_index(index_file_name, conf);

    // IVFFLAT_NM should load raw data
    knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
    bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
    bin->size = dim_ * nb_ * sizeof(float);
    binary_set_.Append(RAW_DATA, bin);

    index_->Load(binary_set_);
    binary_set_.clear();
    test_ivf(conf);
}

TEST_F(Benchmark_knowhere_perf, TEST_IVFSQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamNlist(conf, NLIST_);
    std::string index_file_name = get_index_name({NLIST_});

    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_ivf(conf);
}

TEST_F(Benchmark_knowhere_perf, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamHNSWM(conf, M_);
    knowhere::SetIndexParamEfConstruction(conf, EFCON_);
    std::string index_file_name = get_index_name({M_, EFCON_});

    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_hnsw(conf);
}

TEST_F(Benchmark_knowhere_perf, TEST_ANNOY) {
    index_type_ = knowhere::IndexEnum::INDEX_ANNOY;

    knowhere::Config conf = cfg_;
    knowhere::SetIndexParamNtrees(conf, N_TREE_);
    std::string index_file_name = get_index_name({N_TREE_});

    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    binary_set_.clear();
    test_annoy(conf);
}
