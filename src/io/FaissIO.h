// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <faiss/impl/io.h>

namespace knowhere {

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

inline uint16_t
SwapByteOrder_16(uint16_t value) {
    uint16_t Hi = value << 8;
    uint16_t Lo = value >> 8;
    return Hi | Lo;
}
inline uint32_t
SwapByteOrder_32(uint32_t value) {
    uint32_t Byte0 = value & 0x000000FF;
    uint32_t Byte1 = value & 0x0000FF00;
    uint32_t Byte2 = value & 0x00FF0000;
    uint32_t Byte3 = value & 0xFF000000;
    return (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
}

inline uint64_t
SwapByteOrder_64(uint64_t value) {
    uint64_t Hi = SwapByteOrder_32(uint32_t(value));
    uint32_t Lo = SwapByteOrder_32(uint32_t(value >> 32));
    return (Hi << 32) | Lo;
}

inline float
getSwappedBytes(float C) {
    union {
        uint32_t i;
        float f;
    } in, out;
    in.f = C;
    out.i = SwapByteOrder_32(in.i);
    return out.f;
}

inline float
getSwappedBytes(uint32_t C) {
    return SwapByteOrder_32(C);
}

inline size_t
getSwappedBytes(size_t C) {
    if constexpr (sizeof(size_t) == 4)
        return SwapByteOrder_32(C);
    if constexpr (sizeof(size_t) == 8)
        return SwapByteOrder_64(C);
    static_assert(true, "size_t size error.");
}

inline char
getSwappedBytes(char C) {
    return C;
}

#endif

struct MemoryIOWriter : public faiss::IOWriter {
    uint8_t* data_ = nullptr;
    size_t total = 0;
    size_t rp = 0;

    size_t
    operator()(const void* ptr, size_t size, size_t nitems) override;

    template <typename T>
    size_t
    write(T* ptr, size_t size, size_t nitems = 1) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (size_t i = 0; i < nitems; ++i) {
            *(ptr + i) = getSwappedBytes(*(ptr + i));
        }

#endif
        return operator()((const void*)ptr, size, nitems);
    }
};

struct MemoryIOReader : public faiss::IOReader {
    uint8_t* data_;
    size_t rp = 0;
    size_t total = 0;

    size_t
    operator()(void* ptr, size_t size, size_t nitems) override;

    template <typename T>
    size_t
    read(T* ptr, size_t size, size_t nitems = 1) {
        auto res = operator()((void*)ptr, size, nitems);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (size_t i = 0; i < nitems; ++i) {
            *(ptr + i) = getSwappedBytes(*(ptr + i));
        }
#endif

        return res;
    }
};

}  // namespace knowhere
