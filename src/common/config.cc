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

#include "knowhere/config.h"

#include "knowhere/log.h"
namespace knowhere {

static const std::unordered_set<std::string> ext_legal_json_keys = {"metric_type",
                                                                    "dim",
                                                                    "nlist",
                                                                    "nprobe",
                                                                    "level",
                                                                    "index_type",
                                                                    "index_mode",
                                                                    "collection_id",
                                                                    "partition_id",
                                                                    "segment_id",
                                                                    "field_id",
                                                                    "index_build_id",
                                                                    "index_id",
                                                                    "index_version",
                                                                    "pq_code_budget_gb_ratio",
                                                                    "num_build_thread_ratio",
                                                                    "search_cache_budget_gb_ratio",
                                                                    "num_load_thread_ratio",
                                                                    "beamwidth_ratio",
                                                                    "search_list",
                                                                    "num_build_thread",
                                                                    "num_load_thread",
                                                                    "index_files",
                                                                    "gpu_id",
                                                                    "nbits",
                                                                    "m",
                                                                    "num_threads"};

Status
Config::FormatAndCheck(const Config& cfg, Json& json) {
    for (auto& it : json.items()) {
        bool status = true;
        {
            auto it_ = cfg.__DICT__.find(it.key());
            if (it_ == cfg.__DICT__.end()) {
                status = false;
            }
        }
        {
            auto it_ = ext_legal_json_keys.find(it.key());
            if (it_ == ext_legal_json_keys.end()) {
                status |= false;
            } else {
                status |= true;
            }
        }
        if (!status) {
            LOG_KNOWHERE_ERROR_ << "invalid json key: " << it.key();
            return Status::invalid_param_in_json;
        }
    }

    try {
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;
            if (json.find(it.first) != json.end() && json[it.first].is_string()) {
                if (std::get_if<Entry<CFG_INT>>(&var)) {
                    std::string::size_type sz;
                    auto value_str = json[it.first].get<std::string>();
                    CFG_INT v = std::stoi(value_str.c_str(), &sz);
                    if (sz < value_str.length()) {
                        throw KnowhereException("wrong data type in json");
                    }
                    json[it.first] = v;
                }
                if (std::get_if<Entry<CFG_FLOAT>>(&var)) {
                    CFG_FLOAT v = std::stof(json[it.first].get<std::string>().c_str());
                    json[it.first] = v;
                }

                if (std::get_if<Entry<CFG_BOOL>>(&var)) {
                    if (json[it.first] == "true") {
                        json[it.first] = true;
                    }
                    if (json[it.first] == "false") {
                        json[it.first] = false;
                    }
                }
            }
        }
    } catch (std::exception&) {
        return Status::invalid_value_in_json;
    }
    return Status::success;
}
}  // namespace knowhere
