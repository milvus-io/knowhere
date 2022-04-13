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
#include <hdf5.h>
#include <math.h>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include "unittest/utils.h"

/*****************************************************
 * To run this test, please download the HDF5 from
 *  https://support.hdfgroup.org/ftp/HDF5/releases/
 * and install it to /usr/local/hdf5 .
 *****************************************************/
#define DEBUG_VERBOSE 0

const char HDF5_POSTFIX[] = ".hdf5";
const char HDF5_DATASET_TRAIN[] = "train";
const char HDF5_DATASET_TEST[] = "test";
const char HDF5_DATASET_NEIGHBORS[] = "neighbors";
const char HDF5_DATASET_DISTANCES[] = "distances";

enum QueryMode { MODE_CPU = 0, MODE_MIX, MODE_GPU };

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void normalize(float* arr, int32_t nq, int32_t dim) {
    for (int32_t i = 0; i < nq; i++) {
        double vecLen = 0.0, inv_vecLen = 0.0;
        for (int32_t j = 0; j < dim; j++) {
            double val = arr[i * dim + j];
            vecLen += val * val;
        }
        inv_vecLen = 1.0 / std::sqrt(vecLen);
        for (int32_t j = 0; j < dim; j++) {
            arr[i * dim + j] = (float)(arr[i * dim + j] * inv_vecLen);
        }
    }
}

void* hdf5_read(
    const std::string& file_name,
    const std::string& dataset_name,
    H5T_class_t dataset_class,
    int32_t& d_out,
    int32_t& n_out) {

    hid_t file, dataset, datatype, dataspace, memspace;
    H5T_class_t t_class;      /* data type class */
    hsize_t dimsm[3];         /* memory space dimensions */
    hsize_t dims_out[2];      /* dataset dimensions */
    hsize_t count[2];         /* size of the hyperslab in the file */
    hsize_t offset[2];        /* hyperslab offset in the file */
    hsize_t count_out[3];     /* size of the hyperslab in memory */
    hsize_t offset_out[3];    /* hyperslab offset in memory */
    void* data_out = nullptr; /* output buffer */

    /* Open the file and the dataset. */
    file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

    /* Get datatype and dataspace handles and then query
     * dataset class, order, size, rank and dimensions. */
    datatype = H5Dget_type(dataset); /* datatype handle */
    t_class = H5Tget_class(datatype);
    assert(t_class == dataset_class || !"Illegal dataset class type");

    dataspace = H5Dget_space(dataset); /* dataspace handle */
    H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
    n_out = dims_out[0];
    d_out = dims_out[1];

    /* Define hyperslab in the dataset. */
    offset[0] = offset[1] = 0;
    count[0] = dims_out[0];
    count[1] = dims_out[1];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    /* Define the memory dataspace. */
    dimsm[0] = dims_out[0];
    dimsm[1] = dims_out[1];
    dimsm[2] = 1;
    memspace = H5Screate_simple(3, dimsm, nullptr);

    /* Define memory hyperslab. */
    offset_out[0] = offset_out[1] = offset_out[2] = 0;
    count_out[0] = dims_out[0];
    count_out[1] = dims_out[1];
    count_out[2] = 1;
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);

    /* Read data from hyperslab in the file into the hyperslab in memory and display. */
    switch (t_class) {
        case H5T_INTEGER:
            data_out = new int[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_INT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        case H5T_FLOAT:
            data_out = new float[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        default:
            printf("Illegal dataset class type\n");
            break;
    }

    /* Close/release resources. */
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);

    return data_out;
}

#if DEBUG_VERBOSE
void
print_array(const char* header, bool is_integer, const void* arr, int32_t nq, int32_t k) {
    const int ROW = 10;
    const int COL = 10;
    assert(ROW <= nq);
    assert(COL <= k);
    printf("%s\n", header);
    printf("==============================================\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (is_integer) {
                printf("%7ld ", ((int64_t*)arr)[i * k + j]);
            } else {
                printf("%.6f ", ((float*)arr)[i * k + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
#endif

/************************************************************************************
 * https://github.com/erikbern/ann-benchmarks
 *
 * Dataset  Dimensions  Train_size  Test_size   Neighbors   Distance    Download
 * Fashion-
    MNIST   784         60,000      10,000 	100         Euclidean   HDF5 (217MB)
 * GIST     960         1,000,000   1,000       100         Euclidean   HDF5 (3.6GB)
 * GloVe    100         1,183,514   10,000      100         Angular     HDF5 (463MB)
 * GloVe    200         1,183,514   10,000      100         Angular     HDF5 (918MB)
 * MNIST    784         60,000 	    10,000      100         Euclidean   HDF5 (217MB)
 * NYTimes  256         290,000     10,000      100         Angular     HDF5 (301MB)
 * SIFT     128         1,000,000   10,000      100         Euclidean   HDF5 (501MB)
 *************************************************************************************/

class Benchmark_faiss : public ::testing::Test {
 public:
    bool parse_ann_test_name() {
        size_t pos1, pos2;

        if (ann_test_name_.empty()) {
            return false;
        }

        pos1 = ann_test_name_.find_first_of('-', 0);
        if (pos1 == std::string::npos) {
            return false;
        }
        pos2 = ann_test_name_.find_first_of('-', pos1 + 1);
        if (pos2 == std::string::npos) {
            return false;
        }

        dim_ = std::stoi(ann_test_name_.substr(pos1 + 1, pos2 - pos1 - 1));
        std::string metric_str = ann_test_name_.substr(pos2 + 1);
        if (metric_str == "angular") {
            metric_type_ = faiss::METRIC_INNER_PRODUCT;
        } else if (metric_str == "euclidean") {
            metric_type_ = faiss::METRIC_L2;
        } else {
            return false;
        }

        return true;
    }

    int32_t CalcRecall(const faiss::Index::idx_t* ids, int32_t nq, int32_t k) {
        int32_t min_k = std::min(gt_k_, k);
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::set<faiss::Index::idx_t> ground(gt_ids_ + i * gt_k_, gt_ids_ + i * gt_k_ + min_k);
            for (int32_t j = 0; j < min_k; j++) {
                faiss::Index::idx_t id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }

    void load_base_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        printf("[%.3f s] Loading HDF5 file: %s\n", elapsed() - T0_, ann_file_name.c_str());
        xb_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
        assert(dim == dim_ || !"dataset does not have correct dimension");

        if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
            printf("[%.3f s] Normalizing base data set \n", elapsed() - T0_);
            normalize(xb_, nb_, dim_);
        }
    }

    void load_query_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        xq_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
        assert(dim == dim_ || !"query does not have same dimension as train set");

        if (metric_type_ == faiss::METRIC_INNER_PRODUCT) {
            printf("[%.3f s] Normalizing query data \n", elapsed() - T0_);
            normalize(xq_, nq_, dim_);
        }
    }

    void load_ground_truth() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        // load ground-truth and convert int to long
        int32_t gt_nq;
        int* gt_int = (int*)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, gt_k_, gt_nq);
        assert(gt_nq == nq_ || !"incorrect nb of ground truth index");

        gt_ids_ = new faiss::Index::idx_t[gt_k_ * nq_];
        for (int32_t i = 0; i < gt_k_ * nq_; i++) {
            gt_ids_[i] = gt_int[i];
        }
        delete[] gt_int;

#if DEBUG_VERBOSE
        faiss::Index::distance_t* gt_dist;  // nq * k matrix of ground-truth nearest-neighbors distances
        gt_dist = (float*)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, k, nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth distance");

        std::string str;
        str = ann_test_name + " ground truth index";
        print_array(str.c_str(), true, gt, nq, k);
        str = ann_test_name + " ground truth distance";
        print_array(str.c_str(), false, gt_dist, nq, k);

        delete gt_dist;
#endif
    }

    void write_index(const std::string& filename) {
        faiss::write_index(index_, filename.c_str());
    }

    void read_index(const std::string& filename) {
        index_ = faiss::read_index(filename.c_str());
    }

    void create_cpu_index(const std::string index_file_name) {

        try {
            printf("[%.3f s] Reading index file: %s\n", elapsed() - T0_, index_file_name.c_str());
            read_index(index_file_name);
        } catch (...) {
            printf("[%.3f s] Creating CPU index \"%s\" d=%d\n", elapsed() - T0_, index_key_.c_str(), dim_);
            index_ = faiss::index_factory(dim_, index_key_.c_str(), metric_type_);

            printf("[%.3f s] Training on %d vectors\n", elapsed() - T0_, nb_);
            index_->train(nb_, xb_);

            printf("[%.3f s] Indexing on %d vectors\n", elapsed() - T0_, nb_);
            index_->add(nb_, xb_);

            printf("[%.3f s] Writing index file: %s\n", elapsed() - T0_, index_file_name.c_str());
            write_index(index_file_name);
        }
    }

    void test_ivf(
        const int64_t nlist,
        const std::vector<int32_t>& nqs,
        const std::vector<int32_t>& topks,
        const std::vector<int32_t>& nprobes) {

        faiss::Index::idx_t* I = new faiss::Index::idx_t[nqs.back() * topks.back()];
        faiss::Index::distance_t* D = new faiss::Index::distance_t[nqs.back() * topks.back()];

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n",
               elapsed() - T0_, ann_test_name_.c_str(), index_key_.c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : nprobes) {
            faiss::ParameterSpace params;
            std::string nprobe_str = "nprobe=" + std::to_string(nprobe);
            params.set_index_parameters(index_, nprobe_str.c_str());
            for (auto nq : nqs) {
                for (auto k : topks) {
                    double t_start = elapsed(), t_end;
                    index_->search(nq, xq_, k, D, I);
                    t_end = elapsed();

                    int32_t hit = CalcRecall(I, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                           nprobe, nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", elapsed() - T0_, ann_test_name_.c_str(), index_key_.c_str());

        delete[] I;
        delete[] D;
    }

    void test_hnsw(
        const int64_t M,
        const int64_t efConstruction,
        const std::vector<int32_t>& nqs,
        const std::vector<int32_t>& topks,
        const std::vector<int32_t>& efs) {

        faiss::Index::idx_t* I = new faiss::Index::idx_t[nqs.back() * topks.back()];
        faiss::Index::distance_t* D = new faiss::Index::distance_t[nqs.back() * topks.back()];

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n",
               elapsed() - T0_, ann_test_name_.c_str(), index_key_.c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef: efs) {
            for (auto nq : nqs) {
                for (auto k : topks) {
                    double t_start = elapsed(), t_end;
                    index_->search(nq_, xq_, k, D, I);
                    t_end = elapsed();

                    int32_t hit = CalcRecall(I, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                           ef, nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", elapsed() - T0_, ann_test_name_.c_str(), index_key_.c_str());

        delete[] I;
        delete[] D;
    }

 protected:
    void SetUp() override {
        T0_ = elapsed();

        if (!parse_ann_test_name()) {
            assert(true);
        }

        printf("[%.3f s] Loading base data\n", elapsed() - T0_);
        load_base_data();

        printf("[%.3f s] Loading queries\n", elapsed() - T0_);
        load_query_data();

        printf("[%.3f s] Loading ground truth\n", elapsed() - T0_);
        load_ground_truth();

        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
    }

    void TearDown() override {
        delete[] xb_;
        delete[] xq_;
        delete[] gt_ids_;
    }

 protected:
    double T0_;
    std::string ann_test_name_ = "sift-128-euclidean";
    faiss::MetricType metric_type_;
    int32_t dim_;
    int32_t nb_;
    int32_t nq_;
    int32_t gt_k_;
    faiss::Index::distance_t* xb_;
    faiss::Index::distance_t* xq_;
    faiss::Index::idx_t* gt_ids_;  // ground-truth index

    std::string index_key_;
    faiss::Index* index_ = nullptr;
};

TEST_F(Benchmark_faiss, TEST_IVFFLAT) {
    const std::vector<int32_t> nlists = {256, 512};
    const std::vector<int32_t> nqs = {100};
    const std::vector<int32_t> topks = {10};
    const std::vector<int32_t> nprobes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    std::string index_type = "Flat";

    for (auto nlist : nlists) {
        index_key_ = "IVF" + std::to_string(nlist) + "," + index_type;
        std::string index_file_name = ann_test_name_ + "_IVF" + std::to_string(nlist) + "_" + index_type + ".index";

        create_cpu_index(index_file_name);

        test_ivf(nlist, nqs, topks, nprobes);
    }
}

TEST_F(Benchmark_faiss, TEST_IVFSQ8) {
    const std::vector<int32_t> nlists = {256, 512};
    const std::vector<int32_t> nqs = {100};
    const std::vector<int32_t> topks = {10};
    const std::vector<int32_t> nprobes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    std::string index_type = "SQ8";

    for (auto nlist : nlists) {
        index_key_ = "IVF" + std::to_string(nlist) + "," + index_type;
        std::string index_file_name = ann_test_name_ + "_IVF" + std::to_string(nlist) + "_" + index_type + ".index";

        create_cpu_index(index_file_name);

        test_ivf(nlist, nqs, topks, nprobes);
    }
}

TEST_F(Benchmark_faiss, TEST_HNSW) {
    const std::vector<int32_t> ms = {8, 16};
    const std::vector<int32_t> efCons = {100, 200, 300};
    const std::vector<int32_t> nqs = {100};
    const std::vector<int32_t> topks = {10};
    const std::vector<int32_t> efs = {16, 32, 64, 128, 256};

    std::string index_type = "Flat";

    for (auto M : ms) {
        index_key_ = "HNSW" + std::to_string(M) + "," + index_type;
        for (auto efc : efCons) {
            std::string index_file_name =
                ann_test_name_ + "_HNSW" + std::to_string(M) + "_" + std::to_string(efc) + "_" + index_type + ".index";

            create_cpu_index(index_file_name);

            test_hnsw(M, efc, nqs, topks, efs);
        }
    }
}