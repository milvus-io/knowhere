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
    annoy_inner_error = 11,
    hnsw_inner_error = 12,
    malloc_error = 13,
    diskann_inner_error = 14,
    diskann_file_error = 15,
};

template <typename E>
class unexpected {
 public:
    constexpr explicit unexpected(const E& err) : err(err) {
        static_assert(std::is_same<E, Status>::value);
    }

    constexpr explicit unexpected(E&& err) : err(err) {
        static_assert(std::is_same<E, Status>::value);
    }

    ~unexpected() = default;

    E err;
};

template <typename T, typename E>
class expected {
 public:
    template <typename... Args>
    expected(Args&&... args) : val(std::forward<Args...>(args...)) {
    }

    expected(const unexpected<E>& unexp) {
        err = unexp.err;
    }

    expected(unexpected<E>&& unexp) {
        err = std::move(unexp.err);
    }

    expected(const expected<T, E>&) = default;

    expected(expected<T, E>&&) noexcept = default;

    expected&
    operator=(const expected<T, E>&) = default;

    expected&
    operator=(expected<T, E>&&) noexcept = default;

    bool
    has_value() {
        return val.has_value();
    }

    E
    error() const {
        assert(val.has_value() == false);
        return err.value();
    }

    T&
    value() {
        assert(val.has_value() == true);
        return val.value();
    }

    expected<T, E>&
    operator=(const unexpected<E>& unexp) {
        err = unexp.err;
        return *this;
    }

 private:
    std::optional<T> val = std::nullopt;
    std::optional<E> err = std::nullopt;
};

}  // namespace knowhere

#endif /* EXPECTED_H */
