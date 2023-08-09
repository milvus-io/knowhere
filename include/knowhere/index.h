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

#include "knowhere/binaryset.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index_node.h"

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
    Build(const DataSet& dataset, const Json& json);

    Status
    Train(const DataSet& dataset, const Json& json);

    Status
    Add(const DataSet& dataset, const Json& json);

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Json& json, const BitsetView& bitset) const;

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Json& json, const BitsetView& bitset) const;

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const;

    bool
    HasRawData(const std::string& metric_type) const;

    expected<DataSetPtr>
    GetIndexMeta(const Json& json) const;

    Status
    Serialize(BinarySet& binset) const;

    Status
    Deserialize(const BinarySet& binset, const Json& json = {});

    Status
    DeserializeFromFile(const std::string& filename, const Json& json = {});

    int64_t
    Dim() const;

    int64_t
    Size() const;

    int64_t
    Count() const;

    std::string
    Type() const;

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
