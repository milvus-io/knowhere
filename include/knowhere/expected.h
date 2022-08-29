#ifndef EXPECTED_H
#define EXPECTED_H

#include <cassert>
#include <iostream>
#include <string>
namespace knowhere {

enum class Error {
    success = 0,
    invalid_args,
    invalid_param_in_json,
    out_of_range_in_json,
    type_conflict_in_json,
    invalid_metric_type,
    empty_index,
    not_implemented,
    index_not_trained,
    index_already_trained,
    faiss_inner_error,
    annoy_inner_error,
    hnsw_inner_error,
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

    expected(const expected&) = default;
    expected(expected&&) = default;
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
