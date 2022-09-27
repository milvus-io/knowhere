#ifndef EXPECTED_H
#define EXPECTED_H

#include <cassert>
#include <iostream>
#include <string>
namespace knowhere {

enum class Error {
    success = 0,
    invalid_args = 1,
    invalid_param_in_json = 2,
    out_of_range_in_json = 3,
    type_conflict_in_json = 4,
    invalid_metric_type = 5,
    empty_index = 6,
    not_implemented = 7,
    index_not_trained = 8,
    index_already_trained = 9,
    faiss_inner_error = 10,
    annoy_inner_error = 11,
    hnsw_inner_error = 12,
};

template <typename E>
class unexpected {
 public:
    constexpr unexpected(const E& err) : err(err) {
        static_assert(std::is_same<E, Error>::value);
    }
    constexpr unexpected(E&& err) : err(err) {
        static_assert(std::is_same<E, Error>::value);
    }
    ~unexpected() = default;

    E err;
};

template <typename T, typename E>
class expected {
 public:
    expected(const T& t) : has_val(true), val(t) {
    }
    expected(T&& t) : has_val(true), val(std::move(t)) {
    }
    template <typename... Args>
    expected(Args... args) : has_val(true), val(std::forward<Args...>(args...)) {
    }

    constexpr expected(const unexpected<E>& unexp) {
        has_val = false;
        err = unexp.err;
    }

    expected(const expected<T, E>& other) {
        this->has_val = other.has_val;
        if (other.has_val) {
            this->val = other.val;
        } else {
            this->err = other.err;
        }
    };
    expected(expected<T, E>&&) = default;
    bool
    has_value() {
        return has_val;
    }
    E
    error() const {
        assert(has_val == false);
        return err;
    }
    T&
    value() {
        assert(has_val == true);
        return val;
    }

    expected<T, E>&
    operator=(const unexpected<E>& unexp) {
        has_val = false;
        err = unexp.err;
    }
    ~expected() {
        if (has_val) {
            val.~T();
        } else {
            err.~E();
        }
    }

 private:
    bool has_val;
    union {
        T val;
        E err;
    };
};

}  // namespace knowhere

#endif /* EXPECTED_H */
