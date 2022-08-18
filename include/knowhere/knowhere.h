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
    uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    };
    void
    DecRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    };
    void
    IncRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    };

 private:
    mutable std::atomic_uint32_t ref_counts_ = 1;
};

class IndexNode : public Object {
 public:
    virtual int
    Build(const DataSet& dataset, const Config& cfg) = 0;
    virtual int
    Train(const DataSet& dataset, const Config& cfg) = 0;
    virtual int
    Add(const DataSet& dataset, const Config& cfg) = 0;
    virtual DataSetPtr
    Qeury(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;
    virtual DataSetPtr
    QueryByRange(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const = 0;
    virtual DataSetPtr
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const = 0;
    virtual int
    Serialization(BinarySet& binset) const = 0;
    virtual int
    Deserialization(const BinarySet& binset) = 0;
    virtual std::unique_ptr<Config>
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

    static Index<T1>
    Create() {
        return Index(new T1());
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

    T1*
    operator->() {
        return node;
    }

    template <typename T2>
    Index<T2>
    Cast() {
        static_assert(std::is_base_of<T1, T2>::value);
        node->IncRef();
        return Index(dynamic_cast<T2>(node));
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

#define KNOWHERE_REGISTER_GLOBAL(name, func) \
    const IndexFactory& index_factory_ref_##__COUNTER__ = IndexFactory::Instance().Register(name, func)

}  // namespace knowhere

#endif /* KNOWHERE_H */
