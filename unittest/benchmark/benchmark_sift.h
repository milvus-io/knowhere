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

#pragma once

#include <assert.h>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <sys/time.h>

#include <unordered_set>
#include <vector>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

/*****************************************************
 * To run this test, please download the HDF5 from
 *  https://support.hdfgroup.org/ftp/HDF5/releases/
 * and install it to /usr/local/hdf5 .
 *****************************************************/
static const char* HDF5_POSTFIX = ".hdf5";
static const char* HDF5_DATASET_TRAIN = "train";
static const char* HDF5_DATASET_TEST = "test";
static const char* HDF5_DATASET_NEIGHBORS = "neighbors";
static const char* HDF5_DATASET_DISTANCES = "distances";

static const char* METRIC_IP_STR = "angular";
static const char* METRIC_L2_STR = "euclidean";

/************************************************************************************
 * https://github.com/erikbern/ann-benchmarks
 *
 * Dataset  Dimensions  Train_size  Test_size   Neighbors   Distance    Download
 * Fashion-
    MNIST   784         60,000      10,000      100         Euclidean   HDF5 (217MB)
 * GIST     960         1,000,000   1,000       100         Euclidean   HDF5 (3.6GB)
 * GloVe    100         1,183,514   10,000      100         Angular     HDF5 (463MB)
 * GloVe    200         1,183,514   10,000      100         Angular     HDF5 (918MB)
 * MNIST    784         60,000 	    10,000      100         Euclidean   HDF5 (217MB)
 * NYTimes  256         290,000     10,000      100         Angular     HDF5 (301MB)
 * SIFT     128         1,000,000   10,000      100         Euclidean   HDF5 (501MB)
 *************************************************************************************/
using idx_t = int64_t;
using distance_t = float;

class Benchmark_sift : public ::testing::Test {
 public:
    void
    normalize(float* arr, int32_t nq, int32_t dim) {
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

    double
    elapsed() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    double
    get_time_diff() {
        return elapsed() - T0_;
    }

    void
    set_ann_test_name(const char* test_name) {
        ann_test_name_ = test_name;
    }

    float
    CalcRecall(const idx_t* ids, int32_t nq, int32_t k) {
        int32_t min_k = std::min(gt_k_, k);
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<idx_t> ground(gt_ids_ + i * gt_k_, gt_ids_ + i * gt_k_ + min_k);
            for (int32_t j = 0; j < min_k; j++) {
                idx_t id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (nq * min_k));
    }

    float
    CalcRecall(const idx_t* ids, int32_t nq_start, int32_t step, int32_t k) {
        assert(nq_start + step <= 10000);
        int32_t min_k = std::min(gt_k_, k);
        int32_t hit = 0;
        for (int32_t i = 0; i < step; i++) {
            std::unordered_set<idx_t> ground(gt_ids_ + (i + nq_start) * gt_k_,
                                             gt_ids_ + (i + nq_start) * gt_k_ + min_k);
            for (int32_t j = 0; j < min_k; j++) {
                idx_t id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (step * min_k));
    }

    void
    parse_ann_test_name() {
        size_t pos1, pos2;

        assert(!ann_test_name_.empty() || !"ann_test_name not set");
        pos1 = ann_test_name_.find_first_of('-', 0);
        assert(pos1 != std::string::npos);

        pos2 = ann_test_name_.find_first_of('-', pos1 + 1);
        assert(pos2 != std::string::npos);

        dim_ = std::stoi(ann_test_name_.substr(pos1 + 1, pos2 - pos1 - 1));
        metric_str_ = ann_test_name_.substr(pos2 + 1);
        assert(metric_str_ == METRIC_IP_STR || metric_str_ == METRIC_L2_STR);
    }

    void
    load_base_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        printf("[%.3f s] Loading HDF5 file: %s\n", get_time_diff(), ann_file_name.c_str());
        xb_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
        assert(dim == dim_ || !"dataset does not have correct dimension");

        if (metric_str_ == METRIC_IP_STR) {
            printf("[%.3f s] Normalizing base data set \n", get_time_diff());
            normalize(xb_, nb_, dim_);
        }
    }

    void
    load_query_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        xq_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
        assert(dim == dim_ || !"query does not have same dimension as train set");

        if (metric_str_ == METRIC_IP_STR) {
            printf("[%.3f s] Normalizing query data \n", get_time_diff());
            normalize(xq_, nq_, dim_);
        }
    }

    void
    load_ground_truth() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        // load ground-truth and convert int to long
        int32_t gt_nq;
        int* gt_int = (int*)hdf5_read(ann_file_name, HDF5_DATASET_NEIGHBORS, H5T_INTEGER, gt_k_, gt_nq);
        assert(gt_nq == nq_ || !"incorrect nb of ground truth index");

        gt_ids_ = new idx_t[gt_k_ * nq_];
        for (int32_t i = 0; i < gt_k_ * nq_; i++) {
            gt_ids_[i] = gt_int[i];
        }
        delete[] gt_int;

#if DEBUG_VERBOSE
        distance_t* gt_dist = (float*)hdf5_read(ann_file_name, HDF5_DATASET_DISTANCES, H5T_FLOAT, k, nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth distance");
#endif
    }

 private:
    void*
    hdf5_read(const std::string& file_name, const std::string& dataset_name, H5T_class_t dataset_class, int32_t& d_out,
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

 protected:
    void
    SetUp() override {
        T0_ = elapsed();

        parse_ann_test_name();

        printf("[%.3f s] Loading base data\n", get_time_diff());
        load_base_data();

        printf("[%.3f s] Loading queries\n", get_time_diff());
        load_query_data();

        printf("[%.3f s] Loading ground truth\n", get_time_diff());
        load_ground_truth();
    }

    void
    TearDown() override {
        delete[] xb_;
        delete[] xq_;
        delete[] gt_ids_;
    }

 protected:
    double T0_;
    std::string ann_test_name_ = "";
    std::string metric_str_;
    int32_t dim_;
    distance_t* xb_;
    distance_t* xq_;
    int32_t nb_;
    int32_t nq_;
    int32_t gt_k_;
    idx_t* gt_ids_;  // ground-truth index
};
