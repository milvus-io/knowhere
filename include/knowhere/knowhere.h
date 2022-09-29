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
namespace knowhere {

class Object {
 public:
    inline uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    };
    inline void
    DecRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    };
    inline void
    IncRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    };

 private:
    mutable std::atomic_uint32_t ref_counts_ = 1;
};

class IndexNode : public Object {
 public:
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) = 0;
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) = 0;
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) = 0;
    virtual expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;
    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const = 0;
    virtual Error
    Serialization(BinarySet& binset) const = 0;
    virtual Error
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
    virtual ~IndexNode(){};
};

template <typename T1>
class Index {
 public:
    template <typename T2>
    friend class Index;

    Index() : node(nullptr) {
    }
    static Index<T1>
    Create() {
        return Index(new (std::nothrow) T1());
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

    Error
    Build(const DataSet& dataset, const Json& json) {
        auto cfg = this->node->CreateConfig();
        auto res = Config::Load(*cfg, json, knowhere::TRAIN);
        if (res != Error::success)
            return res;
        return this->node->Build(dataset, *cfg);
    }

    Error
    Train(const DataSet& dataset, const Json& json) {
        auto cfg = this->node->CreateConfig();
        auto res = Config::Load(*cfg, json, knowhere::TRAIN);
        if (res != Error::success)
            return res;
        return this->node->Train(dataset, *cfg);
    }

    Error
    Add(const DataSet& dataset, const Json& json) {
        auto cfg = this->node->CreateConfig();
        auto res = Config::Load(*cfg, json, knowhere::TRAIN);
        if (res != Error::success)
            return res;
        return this->node->Add(dataset, *cfg);
    };

    expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Json& json, const BitsetView& bitset) const {
        auto cfg = this->node->CreateConfig();
        auto res = Config::Load(*cfg, json, knowhere::SEARCH);
        if (res != Error::success)
            return unexpected(res);
        return this->node->Search(dataset, *cfg, bitset);
    };

    expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Json& json) const {
        auto cfg = this->node->CreateConfig();
        auto res = Config::Load(*cfg, json, knowhere::SEARCH);
        if (res != Error::success)
            return unexpected(res);
        return this->node->GetVectorByIds(dataset, *cfg);
    };

    Error
    Serialization(BinarySet& binset) const {
        return this->node->Serialization(binset);
    };

    Error
    Deserialization(const BinarySet& binset) {
        return this->node->Deserialization(binset);
    };

    int64_t
    Dims() const {
        return this->node->Dims();
    };
    int64_t
    Size() const {
        return this->node->Size();
    };
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
    Create(const std::string& name);
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>()> func);

    static IndexFactory&
    Instance();

 private:
    typedef std::map<std::string, std::function<Index<IndexNode>()>> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CONCAT(x, y) x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func) \
    const IndexFactory& KNOWHERE_CONCAT(index_factory_ref_, name) = IndexFactory::Instance().Register(#name, func)

}  // namespace knowhere

#endif /* KNOWHERE_H */
