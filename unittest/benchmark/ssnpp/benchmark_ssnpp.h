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
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "unittest/benchmark/benchmark_base.h"

class Benchmark_ssnpp : public Benchmark_base {
 public:
    void
    set_ann_test_name(const char* test_name) {
        ann_test_name_ = test_name;
    }

    template<typename T>
    void load_bin(const std::string& bin_file, void*& data, int32_t& rows, int32_t& dim) {
        std::ifstream reader;
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            printf("[%.3f s] Opening bin file %s ...\n", get_time_diff(), bin_file.c_str());
            reader.open(bin_file, std::ios::binary | std::ios::ate);
            size_t fsize = reader.tellg();
            reader.seekg(0);

            reader.read((char*)&rows, sizeof(int));
            reader.read((char*)&dim, sizeof(int));

            printf("Metadata: #rows = %d, #dims = %d ...\n", rows, dim);

            size_t expected_actual_file_size = (size_t)rows * dim * sizeof(T) + 2 * sizeof(int32_t);
            assert(fsize == expected_actual_file_size);

            int64_t elem_cnt = (int64_t)rows * dim;
            data = new T[elem_cnt];
            reader.read((char*)data, elem_cnt * sizeof(T));
        } catch (std::system_error& e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    void
    load_range_truthset(const std::string& bin_file, int32_t*& gt_ids, int32_t*& gt_lims, int32_t& gt_num) {
        std::ifstream reader;
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        reader.open(bin_file, std::ios::binary | std::ios::ate);
        printf("[%.3f s] Reading ground truth file %s ...\n", get_time_diff(), bin_file.c_str());
        size_t actual_file_size = reader.tellg();
        reader.seekg(0);

        int total_num;
        reader.read((char*)&gt_num, sizeof(int));
        reader.read((char*)&total_num, sizeof(int));

        printf("Metadata: #gt_num = %d, #total_results = %d ...\n", gt_num, total_num);

        size_t expected_file_size = 2 * sizeof(int32_t) + gt_num * sizeof(int32_t) + total_num * sizeof(int32_t);

        assert(actual_file_size == expected_file_size);

        std::vector<int32_t> elem_count(gt_num);
        reader.read((char*)elem_count.data(), sizeof(int32_t) * gt_num);

        gt_lims = new int32_t[gt_num + 1];
        std::partial_sum(elem_count.begin(), elem_count.end(), gt_lims + 1);

        int32_t total_elem = gt_lims[gt_num];
        gt_ids = new int32_t[total_elem];
        reader.read((char*)gt_ids, total_elem * sizeof(int32_t));
    }


    void
    load_range_radius(const std::string& bin_file, float*& gt_radius, int32_t& gt_num) {
        std::ifstream reader;
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        reader.open(bin_file, std::ios::binary | std::ios::ate);
        printf("[%.3f s] Reading ground truth radius file %s ...\n", get_time_diff(), bin_file.c_str());
        size_t actual_file_size = reader.tellg();
        reader.seekg(0);

        int32_t dim;
        reader.read((char*)&gt_num, sizeof(int));
        reader.read((char*)&dim, sizeof(int));

        printf("Metadata: #gt_num = %d, #dim = %d ...\n", gt_num, dim);

        size_t expected_file_size = 2 * sizeof(int32_t) + gt_num * sizeof(float);

        assert(actual_file_size == expected_file_size);

        gt_radius = new float[gt_num];
        reader.read((char*)gt_radius, sizeof(float) * gt_num);
    }

 protected:
    std::string ann_test_name_ = "";
};
