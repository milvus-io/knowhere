#ifndef KNOWHERE_H
#define KNOWHERE_H

#include <atomic>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/file_manager.h"
#include "knowhere/log.h"
namespace knowhere {

class Object {
 public:
    Object() = default;
    Object(const std::nullptr_t value) {
        assert(value == nullptr);
    }
    inline uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    }
    inline void
    DecRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    }
    inline void
    IncRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    }
    virtual ~Object() {
    }

 private:
    mutable std::atomic_uint32_t ref_counts_ = 1;
};

template <typename T>
class Pack : public Object {
    static_assert(std::is_same_v<T, std::shared_ptr<knowhere::FileManager>>,
                  "IndexPack only support std::shared_ptr<knowhere::FileManager> by far.");

 public:
    Pack() {
    }
    Pack(T package) : package_(package) {
    }
    T
    GetPack() const {
        return package_;
    }
    ~Pack() {
    }

 private:
    T package_;
};

class IndexNode : public Object {
 public:
    virtual Status
    Build(const DataSet& dataset, const Config& cfg) = 0;
    virtual Status
    Train(const DataSet& dataset, const Config& cfg) = 0;
    virtual Status
    Add(const DataSet& dataset, const Config& cfg) = 0;
    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;
    virtual expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;
    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const = 0;
    virtual expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const = 0;
    virtual Status
    Serialization(BinarySet& binset) const = 0;
    virtual Status
    Deserialization(const BinarySet& binset) = 0;
    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const = 0;
    virtual int64_t
    Dims() const = 0;
    virtual int64_t
    Size() const = 0;
    virtual int64_t
    Count() const = 0;
    virtual std::string
    Type() const = 0;
    virtual ~IndexNode() {
    }
};

template <typename T1>
class Index {
 public:
    template <typename T2>
    friend class Index;

    Index() : node(nullptr) {
    }

    static Index<T1>
    Create(const Object& object) {
        return Index(new (std::nothrow) T1(object));
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
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::TRAIN);
        if (res != Status::success) {
            return res;
        }
        return this->node->Build(dataset, *cfg);
    }

    Status
    Train(const DataSet& dataset, const Json& json) {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::TRAIN);
        if (res != Status::success) {
            return res;
        }
        return this->node->Train(dataset, *cfg);
    }

    Status
    Add(const DataSet& dataset, const Json& json) {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::TRAIN);
        if (res != Status::success) {
            return res;
        }
        return this->node->Add(dataset, *cfg);
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::SEARCH);
        if (res != Status::success) {
            return unexpected(res);
        }
        return this->node->Search(dataset, *cfg, bitset);
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::RANGE_SEARCH);
        if (res != Status::success) {
            return unexpected(res);
        }
        return this->node->RangeSearch(dataset, *cfg, bitset);
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Json& json) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::SEARCH);
        if (res != Status::success) {
            return unexpected(res);
        }
        return this->node->GetVectorByIds(dataset, *cfg);
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Json& json) const {
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        Config::Format(*cfg, json_);
        LOG_KNOWHERE_INFO_ << json_.dump();
        auto res = Config::Load(*cfg, json_, knowhere::FEDER);
        if (res != Status::success) {
            return unexpected(res);
        }
        return this->node->GetIndexMeta(*cfg);
    }

    Status
    Serialization(BinarySet& binset) const {
        return this->node->Serialization(binset);
    }

    Status
    Deserialization(const BinarySet& binset) {
        return this->node->Deserialization(binset);
    }

    int64_t
    Dims() const {
        return this->node->Dims();
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

class IndexFactory {
 public:
    Index<IndexNode>
    Create(const std::string& name, const Object& object = nullptr);
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const Object&)> func);
    static IndexFactory&
    Instance();

 private:
    typedef std::map<std::string, std::function<Index<IndexNode>(const Object&)>> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CONCAT(x, y) x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func) \
    const IndexFactory& KNOWHERE_CONCAT(index_factory_ref_, name) = IndexFactory::Instance().Register(#name, func)

}  // namespace knowhere

#endif /* KNOWHERE_H */
