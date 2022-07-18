#include <atomic>
#include <iostream>

namespace knowhere {

class DataSet;

class Binary;

class Object {
 public:
    virtual std::string
    Type() const = 0;
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
    std::atomic_uint32_t ref_counts_ = 1;
};

class IndexProxy : public Object {
 public:
    virtual int
    Train(const DataSet& dataset) = 0;
    virtual int
    AddWithOutIds(const DataSet& dataset) = 0;
    virtual DataSet
    Qeury(const DataSet& dataset, const BitSet& bitset) const = 0;
    virtual DataSet
    QueryByRange(const DataSet& dataset, const BitSet& bitset) const = 0;
    virtual DataSet
    GetVectorByIds(const Binary& ids, Binary& x) const = 0;
    virtual int
    Serialization(Binary& bin) const = 0;
    virtual int
    Deserialization(const Binary& bin) = 0;
    virtual int64_t
    Dims() const = 0;
    virtual int64_t
    Size() const = 0;
    virtual int64_t
    Count() const = 0;
    virtual const string
    Type() {
        return "IndexProxy";
    }
};

template <typename T1>
class Index {
 public:
    template <typename T2>
    friend class Index;

    Index(T1 const* node) : node(node) {
        static_assert(std::is_base_of<IndexProxy, T1>::value);
    }
    template <typename T2>
    Index(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        node = idx.node;
        idx.node->IncRef();
    }

    template <typename T2>
    Index(const Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        node = idx.node;
    }

    template <typename T2>
    Index<T1>&
    operator=(const Index<T2>& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        node = idx.node;
        idx.node->IncRef();
        return *this;
    }
    template <typename T2>
    Index<T1>&
    operator=(const Index<T2>&& idx) {
        static_assert(std::is_base_of<T1, T2>::value);
        node = idx.node;
        return *this;
    }

    Index<T1>
    Clone() {
        return Index<T1>(node);
    }

    T1*
    operator*() {
        return node;
    }
    template <typename T2>
    Index<T2>
    Cast() {
        static_assert(std::is_base_of<T1, T2>::value);
        return Index(dynamic_cast<T2>(node));
    }

    ~Index() {
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    T1* node;
};

}  // namespace knowhere
