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

#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <variant>

#include "expected.h"
#include "nlohmann/json.hpp"

namespace knowhere {

typedef nlohmann::json Json;

#ifndef CFG_INT
#define CFG_INT int32_t
#endif

#ifndef CFG_STRING
#define CFG_STRING std::string
#endif

#ifndef CFG_FLOAT
#define CFG_FLOAT float
#endif

#ifndef CFG_LIST
#define CFG_LIST std::list<int>
#endif

#ifndef CFG_BOOL
#define CFG_BOOL bool
#endif

template <typename T>
struct Entry {};

enum PARAM_TYPE {
    TRAIN = 0x1,
    SEARCH = 0x2,
    RANGE_SEARCH = 0x4,
    FEDER = 0x8,
};

template <>
struct Entry<CFG_STRING> {
    explicit Entry<CFG_STRING>(CFG_STRING* v) {
        val = v;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_STRING>() {
        val = nullptr;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    CFG_STRING* val;
    uint32_t type;
    std::optional<CFG_STRING> default_val;
    std::optional<CFG_STRING> desc;
};

template <>
struct Entry<CFG_FLOAT> {
    explicit Entry<CFG_FLOAT>(CFG_FLOAT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_FLOAT>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_FLOAT* val;
    std::optional<CFG_FLOAT> default_val;
    uint32_t type;
    std::optional<std::pair<CFG_FLOAT, CFG_FLOAT>> range;
    std::optional<CFG_STRING> desc;
};

template <>
struct Entry<CFG_INT> {
    explicit Entry<CFG_INT>(CFG_INT* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<CFG_INT>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    CFG_INT* val;
    std::optional<CFG_INT> default_val;
    uint32_t type;
    std::optional<std::pair<CFG_INT, CFG_INT>> range;
    std::optional<CFG_STRING> desc;
};

template <>
struct Entry<CFG_LIST> {
    explicit Entry<CFG_LIST>(CFG_LIST* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_LIST>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_LIST* val;
    std::optional<CFG_LIST> default_val;
    uint32_t type;
    std::optional<CFG_STRING> desc;
};

template <>
struct Entry<CFG_BOOL> {
    explicit Entry<CFG_BOOL>(CFG_BOOL* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    Entry<CFG_BOOL>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        desc = std::nullopt;
    }

    CFG_BOOL* val;
    std::optional<CFG_BOOL> default_val;
    uint32_t type;
    std::optional<CFG_STRING> desc;
};

template <typename T>
class EntryAccess {
 public:
    EntryAccess(Entry<T>* entry) : entry(entry){};

    EntryAccess&
    set_default(const T dft) {
        entry->default_val = dft;
        *entry->val = dft;
        return *this;
    }

    EntryAccess&
    set_range(T a, T b) {
        entry->range = std::make_pair(a, b);
        return *this;
    }

    EntryAccess&
    description(const CFG_STRING& desc) {
        entry->desc = desc;
        return *this;
    }

    EntryAccess&
    for_train() {
        entry->type |= PARAM_TYPE::TRAIN;
        return *this;
    }

    EntryAccess&
    for_search() {
        entry->type |= PARAM_TYPE::SEARCH;
        return *this;
    }

    EntryAccess&
    for_range_search() {
        entry->type |= PARAM_TYPE::RANGE_SEARCH;
        return *this;
    }

    EntryAccess&
    for_feder() {
        entry->type |= PARAM_TYPE::FEDER;
        return *this;
    }

    EntryAccess&
    for_all() {
        entry->type |= PARAM_TYPE::TRAIN;
        entry->type |= PARAM_TYPE::SEARCH;
        entry->type |= PARAM_TYPE::RANGE_SEARCH;
        return *this;
    }

 private:
    Entry<T>* entry;
};

class Config {
 public:
    static Json
    Save(const Config& cfg) {
        Json json;
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;
            if (const Entry<CFG_INT>* ptr = std::get_if<Entry<CFG_INT>>(&var)) {
                json[it.first] = *ptr->val;
            }

            if (const Entry<CFG_STRING>* ptr = std::get_if<Entry<CFG_STRING>>(&var)) {
                json[it.first] = *ptr->val;
            }

            if (const Entry<CFG_FLOAT>* ptr = std::get_if<Entry<CFG_FLOAT>>(&var)) {
                json[it.first] = *ptr->val;
            }
        }

        return json;
    }

    static Status
    Format(const Config& cfg, Json& json) {
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;
            if (json.find(it.first) != json.end() && json[it.first].is_string()) {
                if (std::get_if<Entry<CFG_INT>>(&var)) {
                    std::stringstream ss;
                    CFG_INT v;
                    ss.str(json[it.first]);
                    ss >> v;
                    json[it.first] = v;
                }
                if (std::get_if<Entry<CFG_FLOAT>>(&var)) {
                    std::stringstream ss;
                    CFG_FLOAT v;
                    ss << json[it.first];
                    ss >> v;
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
        return Status::success;
    }

    static Status
    Load(Config& cfg, const Json& json, PARAM_TYPE type) {
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;

            if (const Entry<CFG_INT>* ptr = std::get_if<Entry<CFG_INT>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it.first].is_number_integer()) {
                    return Status::type_conflict_in_json;
                }
                if (ptr->range.has_value()) {
                    auto v = json[it.first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        return Status::out_of_range_in_json;
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_FLOAT>* ptr = std::get_if<Entry<CFG_FLOAT>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it.first].is_number_float()) {
                    return Status::type_conflict_in_json;
                }
                if (ptr->range.has_value()) {
                    auto v = json[it.first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        return Status::out_of_range_in_json;
                    }
                } else {
                    *ptr->val = json[it.first];
                }
            }

            if (const Entry<CFG_STRING>* ptr = std::get_if<Entry<CFG_STRING>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it.first].is_string()) {
                    return Status::type_conflict_in_json;
                }
                *ptr->val = json[it.first];
            }

            if (const Entry<CFG_LIST>* ptr = std::get_if<Entry<CFG_LIST>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it.first].is_array()) {
                    return Status::type_conflict_in_json;
                }
                for (auto&& i : json[it.first]) ptr->val->push_back(i);
            }

            if (const Entry<CFG_BOOL>* ptr = std::get_if<Entry<CFG_BOOL>>(&var)) {
                if (!(type & ptr->type)) {
                    continue;
                }
                if (json.find(it.first) == json.end() && !ptr->default_val.has_value()) {
                    return Status::invalid_param_in_json;
                }
                if (json.find(it.first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it.first].is_boolean()) {
                    return Status::type_conflict_in_json;
                }
                *ptr->val = json[it.first];
            }
        }

        return Status::success;
    }

    virtual ~Config() {
    }

    using VarEntry =
        std::variant<Entry<CFG_STRING>, Entry<CFG_FLOAT>, Entry<CFG_INT>, Entry<CFG_LIST>, Entry<CFG_BOOL>>;
    std::unordered_map<CFG_STRING, VarEntry> __DICT__;
};

#define KNOHWERE_DECLARE_CONFIG(CONFIG) CONFIG()

#define KNOWHERE_CONFIG_DECLARE_FIELD(PARAM)                                                             \
    __DICT__[#PARAM] = knowhere::Config::VarEntry(std::in_place_type<Entry<decltype(PARAM)>>, &PARAM);   \
    EntryAccess<decltype(PARAM)> PARAM##_access(std::get_if<Entry<decltype(PARAM)>>(&__DICT__[#PARAM])); \
    PARAM##_access

class BaseConfig : public Config {
 public:
    CFG_STRING metric_type;
    CFG_INT k;
    CFG_FLOAT radius;
    CFG_FLOAT range_filter;
    CFG_BOOL range_filter_exist;
    CFG_BOOL trace_visit;
    KNOHWERE_DECLARE_CONFIG(BaseConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(metric_type).set_default("L2").description("metric type").for_all();
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, std::numeric_limits<CFG_INT>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(radius)
            .set_default(0.0)
            .description("radius for range search")
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(range_filter)
            .set_default(0.0)
            .description("result filter for range search")
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(range_filter_exist)
            .set_default(false)
            .description("range filter exist or not")
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(trace_visit)
            .set_default(false)
            .description("trace visit for feder")
            .for_search()
            .for_range_search();
    }
};

}  // namespace knowhere

#endif /* CONFIG_H */
