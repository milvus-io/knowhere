// Copyright (C) 2019-2023 Zilliz. All rights reserved.
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

#include <exception>
#include <vector>

#include "benchmark/utils.h"
#include "benchmark_hdf5.h"
#include "knowhere/binaryset.h"
#include "knowhere/config.h"
#include "knowhere/factory.h"
#include "knowhere/index.h"

class Benchmark_knowhere : public Benchmark_hdf5 {
 public:
    void
    write_index(const std::string& filename, const knowhere::Json& conf) {
        binary_set_.clear();

        FileIOWriter writer(filename);
        index_.Serialize(binary_set_);

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
            throw std::exception();
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
        return ann_test_name_ + "_" + index_type_ + params_str + ".index";
    }

    knowhere::Index<knowhere::IndexNode>
    create_index(const std::string& index_file_name, const knowhere::Json& conf) {
        printf("[%.3f s] Creating index \"%s\"\n", get_time_diff(), index_type_.c_str());
        index_ = knowhere::IndexFactory::Instance().Create(index_type_);

        try {
            printf("[%.3f s] Reading index file: %s\n", get_time_diff(), index_file_name.c_str());
            read_index(index_file_name);
        } catch (...) {
            printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
            knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nb_, dim_, xb_);
            index_.Build(*ds_ptr, conf);

            printf("[%.3f s] Writing index file: %s\n", get_time_diff(), index_file_name.c_str());
            write_index(index_file_name, conf);
        }
        return index_;
    }

 protected:
    std::string metric_type_;
    std::string index_type_;
    knowhere::BinarySet binary_set_;
    knowhere::Json cfg_;
    knowhere::Index<knowhere::IndexNode> index_;
};
