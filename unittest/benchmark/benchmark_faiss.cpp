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

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <gtest/gtest.h>

#include <vector>

#include "unittest/benchmark/benchmark_sift.h"
#include "unittest/utils.h"

#define CALC_TIME_SPAN(X)       \
    double t_start = elapsed(); \
    X;                          \
    double t_diff = elapsed() - t_start;

using idx_t = int64_t;
using distance_t = float;

class Benchmark_faiss : public Benchmark_sift {
 public:
    void
    write_index(const std::string& filename) {
        faiss::write_index(index_, filename.c_str());
    }

    void
    read_index(const std::string& filename) {
        index_ = faiss::read_index(filename.c_str());
    }

    void
    create_cpu_index(const std::string& index_file_name) {
        try {
            printf("[%.3f s] Reading index file: %s\n", get_time_diff(), index_file_name.c_str());
            read_index(index_file_name);
        } catch (...) {
            printf("[%.3f s] Creating CPU index \"%s\" d=%d\n", get_time_diff(), index_key_.c_str(), dim_);
            index_ = faiss::index_factory(dim_, index_key_.c_str(), metric_type_);

            printf("[%.3f s] Training on %d vectors\n", get_time_diff(), nb_);
            index_->train(nb_, (const float*)xb_);

            printf("[%.3f s] Indexing on %d vectors\n", get_time_diff(), nb_);
            index_->add(nb_, (const float*)xb_);

            printf("[%.3f s] Writing index file: %s\n", get_time_diff(), index_file_name.c_str());
            write_index(index_file_name);
        }
    }

    void
    test_idmap() {
        idx_t* I = new idx_t[NQs_.back() * TOPKs_.back()];
        distance_t* D = new distance_t[NQs_.back() * TOPKs_.back()];

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_key_.c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            for (auto k : TOPKs_) {
                CALC_TIME_SPAN(index_->search(nq, (const float*)xq_, k, D, I));
                float recall = CalcRecall(I, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", nq, k, t_diff, recall);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_key_.c_str());

        delete[] I;
        delete[] D;
    }

    void
    test_ivf(const int32_t nlist) {
        idx_t* I = new idx_t[NQs_.back() * TOPKs_.back()];
        distance_t* D = new distance_t[NQs_.back() * TOPKs_.back()];

        printf("\n[%0.3f s] %s | %s | nlist=%d\n", get_time_diff(), ann_test_name_.c_str(), index_key_.c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            faiss::ParameterSpace params;
            std::string nprobe_str = "nprobe=" + std::to_string(nprobe);
            params.set_index_parameters(index_, nprobe_str.c_str());
            for (auto nq : NQs_) {
                for (auto k : TOPKs_) {
                    CALC_TIME_SPAN(index_->search(nq, (const float*)xq_, k, D, I));
                    float recall = CalcRecall(I, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", nprobe, nq, k, t_diff,
                           recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_key_.c_str());

        delete[] I;
        delete[] D;
    }

    void
    test_hnsw(const int64_t M, const int64_t efConstruction) {
        idx_t* I = new idx_t[NQs_.back() * TOPKs_.back()];
        distance_t* D = new distance_t[NQs_.back() * TOPKs_.back()];

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               index_key_.c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            for (auto nq : NQs_) {
                for (auto k : TOPKs_) {
                    CALC_TIME_SPAN(index_->search(nq_, (const float*)xq_, k, D, I));
                    float recall = CalcRecall(I, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n", ef, nq, k, t_diff, recall);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_key_.c_str());

        delete[] I;
        delete[] D;
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
        metric_type_ = (metric_str_ == METRIC_IP_STR) ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        printf("faiss::distance_compute_blas_threshold: %d\n", faiss::distance_compute_blas_threshold);
    }

 protected:
    faiss::MetricType metric_type_;
    std::string index_key_;
    faiss::Index* index_ = nullptr;

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

TEST_F(Benchmark_faiss, TEST_IDMAP) {
    std::string index_type = "Flat";

    index_key_ = index_type;
    std::string index_file_name = ann_test_name_ + "_" + index_type + ".index";
    create_cpu_index(index_file_name);
    test_idmap();
}

TEST_F(Benchmark_faiss, TEST_IVF_FLAT) {
    std::string index_type = "Flat";
    for (auto nlist : NLISTs_) {
        index_key_ = "IVF" + std::to_string(nlist) + "," + index_type;
        std::string index_file_name = ann_test_name_ + "_IVF" + std::to_string(nlist) + "_" + index_type + ".index";
        create_cpu_index(index_file_name);
        test_ivf(nlist);
    }
}

TEST_F(Benchmark_faiss, TEST_IVF_SQ8) {
    std::string index_type = "SQ8";
    for (auto nlist : NLISTs_) {
        index_key_ = "IVF" + std::to_string(nlist) + "," + index_type;
        std::string index_file_name = ann_test_name_ + "_IVF" + std::to_string(nlist) + "_" + index_type + ".index";
        create_cpu_index(index_file_name);
        test_ivf(nlist);
    }
}

TEST_F(Benchmark_faiss, TEST_IVF_PQ) {
    std::string index_type = "PQ";
    for (auto m : Ms_) {
        for (auto nlist : NLISTs_) {
            index_key_ =
                "IVF" + std::to_string(nlist) + "," + index_type + std::to_string(m) + "x" + std::to_string(NBITS_);
            std::string index_file_name =
                ann_test_name_ + "_IVF" + std::to_string(nlist) + "_" + std::to_string(m) + "_" + index_type + ".index";
            create_cpu_index(index_file_name);
            test_ivf(nlist);
        }
    }
}

TEST_F(Benchmark_faiss, TEST_HNSW) {
    std::string index_type = "Flat";
    for (auto M : HNSW_Ms_) {
        index_key_ = "HNSW" + std::to_string(M) + "," + index_type;
        for (auto efc : EFCONs_) {
            std::string index_file_name =
                ann_test_name_ + "_HNSW" + std::to_string(M) + "_" + std::to_string(efc) + "_" + index_type + ".index";
            create_cpu_index(index_file_name);
            test_hnsw(M, efc);
        }
    }
}