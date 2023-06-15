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

#ifndef INDEX_H
#define INDEX_H

#include "knowhere/config.h"
#include "knowhere/index_node.h"
#include "knowhere/log.h"

#ifdef NOT_COMPILE_FOR_SWIG
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

template <typename T1>
class Index {
 public:
    template <typename T2>
    friend class Index;

    Index() : node(nullptr) {
    }

    template <typename... Args>
    static Index<T1>
    Create(Args&&... args) {
        return Index(new (std::nothrow) T1(std::forward<Args>(args)...));
    }

    Index(const Index<T1>& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    Index(Index<T1>&& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    template <typename T2>
    Index(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    template <typename T2>
    Index(Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    template <typename T2>
    Index<T1>&
    operator=(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (idx.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = idx.node;
        node->IncRef();
        return *this;
    }

    template <typename T2>
    Index<T1>&
    operator=(Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = idx.node;
        idx.node = nullptr;
        return *this;
    }

    template <typename T2>
    Index<T2>
    Cast() {
        static_assert(std::is_base_of<T1, T2>::value);
        node->IncRef();
        return Index(dynamic_cast<T2>(node));
    }

    Status
    Build(const DataSet& dataset, const Json& json) {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_STATUS(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "Build config dump: " << json_.dump();
        KNOWHERE_CHECK_STATUS(Config::Load(*cfg, json_, knowhere::TRAIN));

#ifdef NOT_COMPILE_FOR_SWIG
        TimeRecorder rc("Build");
        KNOWHERE_CHECK_STATUS(this->node->Build(dataset, *cfg));
        auto span = rc.ElapseFromBegin("done");
        span *= 0.000001;  // convert to s
        kw_build_latency.Observe(span);
#else
        KNOWHERE_CHECK_STATUS(this->node->Build(dataset, *cfg));
#endif
        return Status::success;
    }

    Status
    Train(const DataSet& dataset, const Json& json) {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_STATUS(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "Train config dump: " << json_.dump();
        KNOWHERE_CHECK_STATUS(Config::Load(*cfg, json_, knowhere::TRAIN));

        KNOWHERE_CHECK_STATUS(this->node->Train(dataset, *cfg));
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Json& json) {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_STATUS(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "Add config dump: " << json_.dump();
        KNOWHERE_CHECK_STATUS(Config::Load(*cfg, json_, knowhere::TRAIN));

        KNOWHERE_CHECK_STATUS(this->node->Add(dataset, *cfg));
        return Status::success;
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_EXPECTED(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "Search config dump: " << json_.dump();
        KNOWHERE_CHECK_EXPECTED(Config::Load(*cfg, json_, knowhere::SEARCH));
        KNOWHERE_CHECK_EXPECTED(cfg->CheckAndAdjustConfig());

#ifdef NOT_COMPILE_FOR_SWIG
        TimeRecorder rc("Search");
        auto res = this->node->Search(dataset, *cfg, bitset);
        auto span = rc.ElapseFromBegin("done");
        span *= 0.001;  // convert to ms
        kw_search_latency.Observe(span);
#else
        auto res = this->node->Search(dataset, *cfg, bitset);
#endif
        return res;
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_EXPECTED(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "RangeSearch config dump: " << json_.dump();
        KNOWHERE_CHECK_EXPECTED(Config::Load(*cfg, json_, knowhere::RANGE_SEARCH));

#ifdef NOT_COMPILE_FOR_SWIG
        TimeRecorder rc("Range Search");
        auto res = this->node->RangeSearch(dataset, *cfg, bitset);
        auto span = rc.ElapseFromBegin("done");
        span *= 0.001;  // convert to ms
        kw_range_search_latency.Observe(span);
#else
        auto res = this->node->RangeSearch(dataset, *cfg, bitset);
#endif
        return res;
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset) const {
        return this->node->GetVectorByIds(dataset);
    }

    bool
    HasRawData(const std::string& metric_type) const {
        return this->node->HasRawData(metric_type);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Json& json) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        KNOWHERE_CHECK_EXPECTED(Config::FormatAndCheck(*cfg, json_));
        LOG_KNOWHERE_DEBUG_ << "GetIndexMeta config dump: " << json_.dump();
        KNOWHERE_CHECK_EXPECTED(Config::Load(*cfg, json_, knowhere::FEDER));

        return this->node->GetIndexMeta(*cfg);
    }

    Status
    Serialize(BinarySet& binset) const {
        return this->node->Serialize(binset);
    }

    Status
    Deserialize(const BinarySet& binset, const Json& json = {}) {
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

    Status
    DeserializeFromFile(const std::string& filename, const Config& config = {}) {
        return this->node->Deserialize(filename, config);
    }

    int64_t
    Dim() const {
        return this->node->Dim();
    }

    int64_t
    Size() const {
        return this->node->Size();
    }

    int64_t
    Count() const {
        return this->node->Count();
    }

    std::string
    Type() {
        return this->node->Type();
    }

    ~Index() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    Index(T1* node) : node(node) {
        static_assert(std::is_base_of<IndexNode, T1>::value);
    }

    T1* node;
};

}  // namespace knowhere

#endif /* INDEX_H */
