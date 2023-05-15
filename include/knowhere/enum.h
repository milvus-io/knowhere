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

#ifndef ENUM_H
#define ENUM_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace knowhere {

namespace details {

template <typename E>
using enable_enum_t = typename std::enable_if<std::is_enum<E>::value, typename std::underlying_type<E>::type>::type;

}  // namespace details

template <typename E>
constexpr inline details::enable_enum_t<E>
underlying_value(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

template <typename E, typename T>
constexpr inline typename std::enable_if<std::is_enum<E>::value && std::is_integral<T>::value, E>::type
to_enum(T value) noexcept {
    return static_cast<E>(value);
}

}  // namespace knowhere

#endif
