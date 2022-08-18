#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <optional>
#include <unordered_map>
#include <variant>

#include "errorstat.h"
#include "nlohmann/json.hpp"

namespace knowhere {

typedef nlohmann::json Json;

template <typename T>
struct Entry {};

enum PARAM_TYPE {
    QUERY = 0x1,
    RANGE = 0x2,
    TRAIN = 0x4,
};

template <>
struct Entry<std::string> {
    Entry<std::string>(std::string* v) {
        val = v;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    Entry<std::string>() {
        val = nullptr;
        type = 0x0;
        default_val = std::nullopt;
        desc = std::nullopt;
    }
    std::string* val;
    uint32_t type;
    std::optional<std::string> default_val;
    std::optional<std::string> desc;
};

template <>
struct Entry<float> {
    Entry<float>(float* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<float>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    float* val;
    std::optional<float> default_val;
    uint32_t type;
    std::optional<std::pair<float, float>> range;
    std::optional<std::string> desc;
};

template <>
struct Entry<int> {
    Entry<int>(int* v) {
        val = v;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }
    Entry<int>() {
        val = nullptr;
        default_val = std::nullopt;
        type = 0x0;
        range = std::nullopt;
        desc = std::nullopt;
    }

    int* val;
    std::optional<int> default_val;
    uint32_t type;
    std::optional<std::pair<int, int>> range;
    std::optional<std::string> desc;
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
    description(const std::string& desc) {
        entry->desc = desc;
        return *this;
    }
    EntryAccess&
    for_query() {
        entry->type |= PARAM_TYPE::QUERY;
        return *this;
    }
    EntryAccess&
    for_train() {
        entry->type |= PARAM_TYPE::TRAIN;
        return *this;
    }
    EntryAccess&
    for_range() {
        entry->type |= PARAM_TYPE::RANGE;
        return *this;
    }
    EntryAccess&
    for_all() {
        entry->type |= PARAM_TYPE::QUERY;
        entry->type |= PARAM_TYPE::RANGE;
        entry->type |= PARAM_TYPE::TRAIN;
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
        for (auto it = cfg.__DICT__.begin(); it != cfg.__DICT__.end(); ++it) {
            const auto& var = it->second;

            if (const Entry<int>* ptr = std::get_if<Entry<int>>(&var)) {
                json[it->first] = *ptr->val;
            }

            if (const Entry<std::string>* ptr = std::get_if<Entry<std::string>>(&var)) {
                json[it->first] = *ptr->val;
            }

            if (const Entry<float>* ptr = std::get_if<Entry<float>>(&var)) {
                json[it->first] = *ptr->val;
            }
        }

        return json;
    }

    static int
    Load(Config& cfg, const Json& json, PARAM_TYPE type) {
        auto cfg_bak = cfg;
        for (auto it = cfg.__DICT__.begin(); it != cfg.__DICT__.end(); ++it) {
            const auto& var = it->second;

            if (const Entry<int>* ptr = std::get_if<Entry<int>>(&var)) {
                if (!(type & ptr->type))
                    continue;
                if (json.find(it->first) == json.end() && !ptr->default_val.has_value()) {
                    cfg = cfg_bak;
                    return -1;
                }
                if (json.find(it->first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it->first].is_number_integer()) {
                    cfg = cfg_bak;
                    return -2;
                }
                if (ptr->range.has_value()) {
                    auto v = json[it->first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        cfg = cfg_bak;
                        return -3;
                    }
                } else {
                    *ptr->val = json[it->first];
                }
            }

            if (const Entry<float>* ptr = std::get_if<Entry<float>>(&var)) {
                if (!(type & ptr->type))
                    continue;
                if (json.find(it->first) == json.end() && !ptr->default_val.has_value()) {
                    cfg = cfg_bak;
                    return -4;
                }
                if (json.find(it->first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it->first].is_number_float()) {
                    cfg = cfg_bak;
                    return -5;
                }
                if (ptr->range.has_value()) {
                    auto v = json[it->first];
                    if (ptr->range.value().first <= v && v <= ptr->range.value().second) {
                        *ptr->val = v;
                    } else {
                        cfg = cfg_bak;
                        return -6;
                    }
                } else {
                    *ptr->val = json[it->first];
                }
            }

            if (const Entry<std::string>* ptr = std::get_if<Entry<std::string>>(&var)) {
                if (!(type & ptr->type))
                    continue;
                if (json.find(it->first) == json.end() && !ptr->default_val.has_value()) {
                    cfg = cfg_bak;
                    return -7;
                }
                if (json.find(it->first) == json.end()) {
                    *ptr->val = ptr->default_val.value();
                    continue;
                }
                if (!json[it->first].is_string()) {
                    cfg = cfg_bak;
                    return -8;
                }
                *ptr->val = json[it->first];
            }
        }

        return 0;
    }

    typedef std::variant<Entry<std::string>, Entry<float>, Entry<int>> VarEntry;
    std::unordered_map<std::string, VarEntry> __DICT__;
};

#define KNOHWERE_DECLARE_CONFIG(CONFIG) CONFIG()

#define KNOWHERE_CONFIG_DECLARE_FIELD(PARAM)                                                             \
    __DICT__[#PARAM] = knowhere::Config::VarEntry(std::in_place_type<Entry<decltype(PARAM)>>, &PARAM);   \
    EntryAccess<decltype(PARAM)> PARAM##_access(std::get_if<Entry<decltype(PARAM)>>(&__DICT__[#PARAM])); \
    PARAM##_access

}  // namespace knowhere

#endif /* CONFIG_H */
