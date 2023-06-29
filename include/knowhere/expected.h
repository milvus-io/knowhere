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

#ifndef EXPECTED_H
#define EXPECTED_H

#include <cassert>
#include <iostream>
#include <optional>
#include <string>

namespace knowhere {

enum class Status {
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
    hnsw_inner_error = 12,
    malloc_error = 13,
    diskann_inner_error = 14,
    diskann_file_error = 15,
    invalid_value_in_json = 16,
    arithmetic_overflow = 17,
    raft_inner_error = 18,
    invalid_binary_set = 19,
};

template <typename T>
class expected {
 public:
    template <typename... Args>
    expected(Args&&... args) : val(std::make_optional<T>(std::forward<Args>(args)...)) {
    }

    expected(const Status& err) : err(err) {
        assert(err != Status::success);
    }

    expected(Status&& err) : err(err) {
        assert(err != Status::success);
    }

    expected(const expected<T>&) = default;

    expected(expected<T>&&) noexcept = default;

    expected&
    operator=(const expected<T>&) = default;

    expected&
    operator=(expected<T>&&) noexcept = default;

    bool
    has_value() const {
        return val.has_value();
    }

    Status
    error() const {
        assert(val.has_value() == false);
        return err.value();
    }

    const T&
    value() const {
        assert(val.has_value() == true);
        return val.value();
    }

    const std::string&
    what() const {
        return msg;
    }

    void
    operator<<(const std::string& msg) {
        this->msg += msg;
    }

    expected<T>&
    operator=(const Status& err) {
        assert(err != Status::success);
        this->err = err;
        return *this;
    }

 private:
    std::optional<T> val = std::nullopt;
    std::optional<Status> err = std::nullopt;
    std::string msg;
};

// Evaluates expr that returns a Status. Does nothing if the returned Status is
// a Status::success, otherwise returns the Status from the current function.
#define RETURN_IF_ERROR(expr)            \
    do {                                 \
        auto status = (expr);            \
        if (status != Status::success) { \
            return status;               \
        }                                \
    } while (0)

template <typename T>
Status
DoAssignOrReturn(T& lhs, const expected<T>& exp) {
    if (exp.has_value()) {
        lhs = exp.value();
        return Status::success;
    }
    return exp.error();
}

#define STATUS_INTERNAL_CONCAT_NAME_INNER(x, y) x##y
#define STATUS_INTERNAL_CONCAT_NAME(x, y) STATUS_INTERNAL_CONCAT_NAME_INNER(x, y)

#define STATUS_INTERNAL_DEPAREN(X) STATUS_INTERNAL_ESC(STATUS_INTERNAL_ISH X)
#define STATUS_INTERNAL_ISH(...) STATUS_INTERNAL_ISH __VA_ARGS__
#define STATUS_INTERNAL_ESC(...) STATUS_INTERNAL_ESC_(__VA_ARGS__)
#define STATUS_INTERNAL_ESC_(...) STATUS_INTERNAL_VAN_STATUS_INTERNAL_##__VA_ARGS__
#define STATUS_INTERNAL_VAN_STATUS_INTERNAL_STATUS_INTERNAL_ISH

#define STATUS_INTERNAL_ASSIGN_OR_RETURN_IMPL(status, lhs, rexpr) \
    Status status = knowhere::DoAssignOrReturn(lhs, rexpr);       \
    if (status != Status::success) {                              \
        return status;                                            \
    }

// Evaluates an expression that returns an `expected`. If the expected has a value, assigns
// the value to var. Otherwise returns the error from the current function.
//
// Example: ASSIGN_OR_RETURN(int, i, MaybeInt());
//
// If the type parameter has comma not wrapped by paired parenthesis/double quotes, wrap
// the comma in parenthesis properly.
//
// Examples:
//    ASSIGN_OR_RETURN(std::pair<int, int>, pair, MaybePair());  // Not OK
//    ASSIGN_OR_RETURN((std::pair<int, int>), pair, MaybePair());  // OK
//    ASSIGN_OR_RETURN(std::function<void(int, int)>), fn, MaybeFunction());  // OK
//
// Note that this macro expands into multiple statements and thus cannot be used in a single statement
// such as the body of an if statement without {}.
#define ASSIGN_OR_RETURN(type, var, rexpr) \
    STATUS_INTERNAL_DEPAREN(type) var;     \
    STATUS_INTERNAL_ASSIGN_OR_RETURN_IMPL(STATUS_INTERNAL_CONCAT_NAME(_excepted_, __COUNTER__), var, rexpr)

}  // namespace knowhere

#endif /* EXPECTED_H */
