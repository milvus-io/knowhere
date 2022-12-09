#ifndef OBJECT_H
#define OBJECT_H

#include <atomic>
#include <iostream>

#include "knowhere/file_manager.h"

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

}  // namespace knowhere
#endif /* OBJECT_H */
