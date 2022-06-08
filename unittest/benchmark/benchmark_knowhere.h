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

#include <vector>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/benchmark/benchmark_sift.h"
#include "unittest/utils.h"

class Benchmark_knowhere : public Benchmark_sift {
 public:
    void
    write_index(const std::string& filename, const knowhere::Config& conf) {
        binary_set_.clear();

        FileIOWriter writer(filename);
        binary_set_ = index_->Serialize(conf);

        const auto& m = binary_set_.binary_map_;
        for (auto it = m.begin(); it != m.end(); ++it) {
            const std::string& name = it->first;
            size_t name_size = name.length();
            const knowhere::BinaryPtr data = it->second;
            size_t data_size = data->size;

            writer(&name_size, sizeof(name_size));
            writer(&data_size, sizeof(data_size));
            writer((void*)name.c_str(), name_size);
            writer(data->data.get(), data_size);
        }
    }

    void
    read_index(const std::string& filename) {
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

    std::string
    get_index_name(const std::vector<int32_t>& params) {
        std::string params_str = "";
        for (size_t i = 0; i < params.size(); i++) {
            params_str += "_" + std::to_string(params[i]);
        }
        return ann_test_name_ + "_" + std::string(index_type_) + params_str + ".index";
    }

    void
    create_cpu_index(const std::string& index_file_name, const knowhere::Config& conf) {
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

 protected:
    knowhere::MetricType metric_type_;
    knowhere::BinarySet binary_set_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
    knowhere::Config cfg_;
};
