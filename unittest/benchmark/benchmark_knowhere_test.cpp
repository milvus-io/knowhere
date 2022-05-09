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
#include <unordered_set>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
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

using idx_t = int64_t;
using distance_t = float;

class Benchmark_knowhere : public ::testing::Test {
 public:
    double get_time_diff() {
        return elapsed() - T0_;
    }

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
            metric_type_ = knowhere::metric::IP;
        } else if (metric_str == "euclidean") {
            metric_type_ = knowhere::metric::L2;
        } else {
            return false;
        }

        return true;
    }

    int32_t CalcRecall(const idx_t* ids, int32_t nq, int32_t k) {
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
        return hit;
    }

    void load_base_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        printf("[%.3f s] Loading HDF5 file: %s\n", get_time_diff(), ann_file_name.c_str());
        xb_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TRAIN, H5T_FLOAT, dim, nb_);
        assert(dim == dim_ || !"dataset does not have correct dimension");

        if (metric_type_ == knowhere::metric::IP) {
            printf("[%.3f s] Normalizing base data set \n", get_time_diff());
            normalize(xb_, nb_, dim_);
        }
    }

    void load_query_data() {
        const std::string ann_file_name = ann_test_name_ + HDF5_POSTFIX;

        int32_t dim;
        xq_ = (float*)hdf5_read(ann_file_name, HDF5_DATASET_TEST, H5T_FLOAT, dim, nq_);
        assert(dim == dim_ || !"query does not have same dimension as train set");

        if (metric_type_ == knowhere::metric::IP) {
            printf("[%.3f s] Normalizing query data \n", get_time_diff());
            normalize(xq_, nq_, dim_);
        }
    }

    void load_ground_truth() {
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

    void write_index(
        const std::string& filename,
        const knowhere::Config& conf) {

        binary_set_.clear();

        FileIOWriter writer(filename);
        binary_set_ = index_->Serialize(conf);

        const auto& m = binary_set_.binary_map_;
        for (auto it = m.begin(); it != m.end(); ++it) {
            const std::string& name = it->first;
            size_t name_size = name.length();
            const knowhere::BinaryPtr data = it->second;
            size_t data_size = data->size;

            writer(&name_size, sizeof(size_t));
            writer(&data->size, sizeof(data->size));
            writer((void*)name.c_str(), name_size);
            writer(data->data.get(), data->size);
        }
    }

    void read_index(const std::string& filename) {
        binary_set_.clear();

        FileIOReader reader(filename);
        int64_t file_size = reader.size();
        if (file_size < 0) {
            throw knowhere::KnowhereException(filename + " not exist");
        }

        int64_t offset = 0;
        while (offset < file_size) {
            size_t name_size, data_size;
            reader(&name_size, sizeof(size_t));
            offset += sizeof(size_t);
            reader(&data_size, sizeof(size_t));
            offset += sizeof(size_t);

            std::string name;
            name.resize(name_size);
            reader(name.data(), name_size);
            offset += name_size;
            auto data = new uint8_t[data_size];
            reader(data, data_size);
            offset += data_size;

            std::shared_ptr<uint8_t[]> data_ptr(data);
            binary_set_.Append(name, data_ptr, data_size);
        }
    }

    std::string get_index_name(const std::vector<int32_t>& params) {
        std::string params_str = "";
        for (size_t i = 0; i < params.size(); i++) {
            params_str += "_" + std::to_string(params[i]);
        }
        return ann_test_name_ + "_" + std::string(index_type_) + params_str + ".index";
    }

    void create_cpu_index(
        const std::string& index_file_name,
        const knowhere::Config& conf) {

        printf("[%.3f s] Creating CPU index \"%s\"\n", get_time_diff(), std::string(index_type_).c_str());
        auto& factory = knowhere::VecIndexFactory::GetInstance();
        index_ = factory.CreateVecIndex(index_type_);

        try {
            printf("[%.3f s] Reading index file: %s\n", get_time_diff(), index_file_name.c_str());
            read_index(index_file_name);
        } catch (...) {
            printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nb_, dim_, xb_);
            index_->BuildAll(ds_ptr, conf);

            printf("[%.3f s] Writing index file: %s\n", get_time_diff(), index_file_name.c_str());
            write_index(index_file_name, conf);
        }
    }

    void test_idmap(const knowhere::Config& cfg) {
        auto conf = cfg;

        printf("\n[%0.3f s] %s | %s \n",
               get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
            for (auto k : TOPKs_) {
                knowhere::SetMetaTopk(conf, k);

                double t_start = elapsed(), t_end;
                auto result = index_->Query(ds_ptr, conf, nullptr);
                t_end = elapsed();

                auto ids = knowhere::GetDatasetIDs(result);
                int32_t hit = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                       nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void test_ivf(
        const knowhere::Config& cfg,
        const std::vector<int32_t>& nprobes) {

        auto conf = cfg;
        auto nlist = knowhere::GetIndexParamNlist(conf);

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n",
               get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str(), nlist);
        printf("================================================================================\n");
        for (auto nprobe : nprobes) {
            knowhere::SetIndexParamNprobe(conf, nprobe);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);

                    double t_start = elapsed(), t_end;
                    auto result = index_->Query(ds_ptr, conf, nullptr);
                    t_end = elapsed();

                    auto ids = knowhere::GetDatasetIDs(result);
                    int32_t hit = CalcRecall(ids, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                           nprobe, nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void test_hnsw(
        const knowhere::Config& cfg,
        const std::vector<int32_t>& efs) {

        auto conf = cfg;
        auto M = knowhere::GetIndexParamM(conf);
        auto efConstruction = knowhere::GetIndexParamEfConstruction(conf);

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n",
               get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str(), M, efConstruction);
        printf("================================================================================\n");
        for (auto ef: efs) {
            knowhere::SetIndexParamEf(conf, ef);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);

                    double t_start = elapsed(), t_end;
                    auto result = index_->Query(ds_ptr, conf, nullptr);
                    t_end = elapsed();

                    auto ids = knowhere::GetDatasetIDs(result);
                    int32_t hit = CalcRecall(ids, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                           ef, nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

    void test_annoy(
        const knowhere::Config& cfg,
        const std::vector<int32_t>& search_ks) {

        auto conf = cfg;
        auto n_trees = knowhere::GetIndexParamNtrees(conf);

        printf("\n[%0.3f s] %s | %s | n_trees=%ld \n",
               get_time_diff(), ann_test_name_.c_str(), std::string(index_type_).c_str(), n_trees);
        printf("================================================================================\n");
        for (auto sk: search_ks) {
            knowhere::SetIndexParamSearchK(conf, sk);
            for (auto nq : NQs_) {
                knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    knowhere::SetMetaTopk(conf, k);

                    double t_start = elapsed(), t_end;
                    auto result = index_->Query(ds_ptr, conf, nullptr);
                    t_end = elapsed();

                    auto ids = knowhere::GetDatasetIDs(result);
                    int32_t hit = CalcRecall(ids, nq, k);
                    printf("  search_k = %4d, nq = %4d, k = %4d, elapse = %.4fs, R@ = %.4f\n",
                           sk, nq, k, (t_end - t_start), (hit / float(nq * std::min(gt_k_, k))));
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(),
               std::string(index_type_).c_str());
    }

 protected:
    void SetUp() override {
        T0_ = elapsed();

        if (!parse_ann_test_name()) {
            assert(true);
        }

        printf("[%.3f s] Loading base data\n", get_time_diff());
        load_base_data();

        printf("[%.3f s] Loading queries\n", get_time_diff());
        load_query_data();

        printf("[%.3f s] Loading ground truth\n", get_time_diff());
        load_ground_truth();

        knowhere::SetMetaMetricType(cfg_, metric_type_);
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
    knowhere::MetricType metric_type_;
    int32_t dim_;
    int32_t nb_;
    int32_t nq_;
    int32_t gt_k_;
    distance_t* xb_;
    distance_t* xq_;
    idx_t* gt_ids_;  // ground-truth index

    knowhere::BinarySet binary_set_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
    knowhere::Config cfg_;

    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {10};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};

    // HNSW index params
    const std::vector<int32_t> Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {100};

    // ANNOY index params
    const std::vector<int32_t> N_TREEs_ = {32};
};

TEST_F(Benchmark_knowhere, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_cpu_index(index_file_name, conf);
    index_->Load(binary_set_);
    test_idmap(conf);
}

TEST_F(Benchmark_knowhere, TEST_IVFFLAT_NM) {
    const std::vector<int32_t> nprobes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_cpu_index(index_file_name, conf);

        // IVFFLAT_NM should load raw data
        knowhere::BinaryPtr bin = std::make_shared<knowhere::Binary>();
        bin->data = std::shared_ptr<uint8_t[]>((uint8_t*)xb_, [&](uint8_t*) {});
        bin->size = dim_ * nb_ * sizeof(float);
        binary_set_.Append(RAW_DATA, bin);

        index_->Load(binary_set_);
        test_ivf(conf, nprobes);
    }
}

TEST_F(Benchmark_knowhere, TEST_IVFSQ8) {
    const std::vector<int32_t> nprobes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Config conf = cfg_;
    for (auto nlist : NLISTs_) {
        std::string index_file_name = get_index_name({nlist});
        knowhere::SetIndexParamNlist(conf, nlist);
        create_cpu_index(index_file_name, conf);
        index_->Load(binary_set_);
        test_ivf(conf, nprobes);
    }
}

TEST_F(Benchmark_knowhere, TEST_HNSW) {
    const std::vector<int32_t> efs = {16, 32, 64, 128, 256};

    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Config conf = cfg_;
    for (auto M : Ms_) {
        knowhere::SetIndexParamM(conf, M);
        for (auto efc : EFCONs_) {
            std::string index_file_name = get_index_name({M, efc});
            knowhere::SetIndexParamEfConstruction(conf, efc);
            create_cpu_index(index_file_name, conf);
            index_->Load(binary_set_);
            test_hnsw(conf, efs);
        }
    }
}

TEST_F(Benchmark_knowhere, TEST_ANNOY) {
    const std::vector<int32_t> search_ks = {16, 32, 64, 128, 256};

    index_type_ = knowhere::IndexEnum::INDEX_ANNOY;

    knowhere::Config conf = cfg_;
    for (auto n : N_TREEs_) {
        knowhere::SetIndexParamNtrees(conf, n);
        std::string index_file_name = get_index_name({n});
        create_cpu_index(index_file_name, conf);
        index_->Load(binary_set_);
        test_annoy(conf, search_ks);
    }
}
