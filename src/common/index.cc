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

#include "knowhere/index.h"

#include "knowhere/log.h"

#ifdef NOT_COMPILE_FOR_SWIG
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

inline Status
LoadConfig(BaseConfig* cfg, const Json& json, knowhere::PARAM_TYPE param_type, const std::string& method,
           std::string* const msg = nullptr) {
    Json json_(json);
    auto res = Config::FormatAndCheck(*cfg, json_, msg);
    LOG_KNOWHERE_DEBUG_ << method << " config dump: " << json_.dump();
    RETURN_IF_ERROR(res);
    return Config::Load(*cfg, json_, param_type, msg);
}

template <typename T>
inline Status
Index<T>::Build(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));
    RETURN_IF_ERROR(cfg->CheckAndAdjustForBuild());

#ifdef NOT_COMPILE_FOR_SWIG
    knowhere_build_count.Increment();
#endif
    return this->node->Build(dataset, *cfg);
}

template <typename T>
inline Status
Index<T>::Train(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Train"));
    return this->node->Train(dataset, *cfg);
}

template <typename T>
inline Status
Index<T>::Add(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Add"));
    return this->node->Add(dataset, *cfg);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::Search(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    const Status load_status = LoadConfig(cfg.get(), json, knowhere::SEARCH, "Search", &msg);
    if (load_status != Status::success) {
        expected<DataSetPtr> ret(load_status);
        ret << msg;
        return ret;
    }
    const Status search_status = cfg->CheckAndAdjustForSearch(&msg);
    if (search_status != Status::success) {
        expected<DataSetPtr> ret(search_status);
        ret << msg;
        return ret;
    }

#ifdef NOT_COMPILE_FOR_SWIG
    knowhere_search_count.Increment();
#endif
    return this->node->Search(dataset, *cfg, bitset);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::RangeSearch(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::RANGE_SEARCH, "RangeSearch"));
    RETURN_IF_ERROR(cfg->CheckAndAdjustForRangeSearch());

#ifdef NOT_COMPILE_FOR_SWIG
    knowhere_range_search_count.Increment();
#endif
    return this->node->RangeSearch(dataset, *cfg, bitset);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetVectorByIds(const DataSet& dataset) const {
    return this->node->GetVectorByIds(dataset);
}

template <typename T>
inline bool
Index<T>::HasRawData(const std::string& metric_type) const {
    return this->node->HasRawData(metric_type);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetIndexMeta(const Json& json) const {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::FEDER, "GetIndexMeta"));
    return this->node->GetIndexMeta(*cfg);
}

template <typename T>
inline Status
Index<T>::Serialize(BinarySet& binset) const {
    return this->node->Serialize(binset);
}

template <typename T>
inline Status
Index<T>::Deserialize(const BinarySet& binset, const Json& json) {
    Json json_(json);
    auto cfg = this->node->CreateConfig();
    {
        auto res = Config::FormatAndCheck(*cfg, json_);
        LOG_KNOWHERE_DEBUG_ << "Deserialize config dump: " << json_.dump();
        if (res != Status::success) {
            return res;
        }
    }
    auto res = Config::Load(*cfg, json_, knowhere::DESERIALIZE);
    if (res != Status::success) {
        return res;
    }
    return this->node->Deserialize(binset, *cfg);
}

template <typename T>
inline Status
Index<T>::DeserializeFromFile(const std::string& filename, const Json& json) {
    Json json_(json);
    auto cfg = this->node->CreateConfig();
    {
        auto res = Config::FormatAndCheck(*cfg, json_);
        LOG_KNOWHERE_DEBUG_ << "DeserializeFromFile config dump: " << json_.dump();
        if (res != Status::success) {
            return res;
        }
    }
    auto res = Config::Load(*cfg, json_, knowhere::DESERIALIZE_FROM_FILE);
    if (res != Status::success) {
        return res;
    }
    return this->node->DeserializeFromFile(filename, *cfg);
}

template <typename T>
inline int64_t
Index<T>::Dim() const {
    return this->node->Dim();
}

template <typename T>
inline int64_t
Index<T>::Size() const {
    return this->node->Size();
}

template <typename T>
inline int64_t
Index<T>::Count() const {
    return this->node->Count();
}

template <typename T>
inline std::string
Index<T>::Type() const {
    return this->node->Type();
}

template class Index<IndexNode>;

}  // namespace knowhere
