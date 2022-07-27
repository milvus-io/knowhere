// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "knowhere/archive/KnowhereConfig.h"
#include "knowhere/common/Dataset.h"
#include "knowhere/common/Log.h"

class DataGen {
 public:
    DataGen() {
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
    }

 protected:
    void
    Init_with_default(const bool is_binary = false);

    void
    Generate(const int dim, const int nb, const int nq, const bool is_binary = false);

 protected:
    int nb = 10000;
    int nq = 10;
    int dim = 128;
    int k = 10;
    int buffer_size = 16384;
    float radius = 2.8;
    std::vector<float> xb;
    std::vector<float> xq;
    std::vector<uint8_t> xb_bin;
    std::vector<uint8_t> xq_bin;
    std::vector<int64_t> ids;
    knowhere::DatasetPtr base_dataset = nullptr;
    knowhere::DatasetPtr query_dataset = nullptr;
    knowhere::DatasetPtr id_dataset = nullptr;

    std::vector<uint8_t> bitset_data;
    faiss::BitsetViewPtr bitset;
};

extern void
GenAll(const int64_t dim,
       const int64_t nb,
       std::vector<float>& xb,
       std::vector<int64_t>& ids,
       const int64_t nq,
       std::vector<float>& xq);

extern void
GenAll(const int64_t dim,
       const int64_t nb,
       std::vector<uint8_t>& xb,
       std::vector<int64_t>& ids,
       const int64_t nq,
       std::vector<uint8_t>& xq);

extern void
GenBase(const int64_t dim,
        const int64_t nb,
        const void* xb,
        int64_t* ids,
        const int64_t nq,
        const void* xq,
        const bool is_binary);

enum class CheckMode {
    CHECK_EQUAL = 0,
    CHECK_NOT_EQUAL = 1,
    CHECK_APPROXIMATE_EQUAL = 2,
};

void
AssertAnns(const knowhere::DatasetPtr& result,
           const int nq,
           const int k,
           const CheckMode check_mode = CheckMode::CHECK_EQUAL);

void
AssertDist(const knowhere::DatasetPtr& result,
           const knowhere::MetricType& metric,
           const int nq,
           const int k);

void
AssertVec(const knowhere::DatasetPtr& result,
          const knowhere::DatasetPtr& base_dataset,
          const knowhere::DatasetPtr& id_dataset,
          const int n,
          const int dim);

void
AssertBinVec(const knowhere::DatasetPtr& result,
             const knowhere::DatasetPtr& base_dataset,
             const knowhere::DatasetPtr& id_dataset,
             const int n,
             const int dim);

void
normalize(float* vec, int64_t n, int64_t dim);

void
PrintResult(const knowhere::DatasetPtr& result, const int& nq, const int& k);

struct FileIOWriter {
    std::fstream fs;
    std::string name;

    explicit FileIOWriter(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::out | std::ios::binary);
    }

    ~FileIOWriter() {
        fs.close();
    }

    size_t operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct FileIOReader {
    std::fstream fs;
    std::string name;

    explicit FileIOReader(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::in | std::ios::binary);
    }

    ~FileIOReader() {
        fs.close();
    }

    size_t operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return size;
    }

    size_t size() {
        fs.seekg(0, fs.end);
        size_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};

inline void
set_bit(uint8_t* data, size_t idx) {
    data[idx >> 3] |= 0x1 << (idx & 0x7);
}

inline void
clear_bit(uint8_t* data, size_t idx) {
    data[idx >> 3] &= ~(0x1 << (idx & 0x7));
}

template <typename T_>
struct CMin {
    typedef T_ T;
    inline static bool cmp(T a, T b) {
        return a < b;
    }
};

template <typename T_>
struct CMax {
    typedef T_ T;
    inline static bool cmp(T a, T b) {
        return a > b;
    }
};

std::string
temp_path(const char* path);

#ifdef __MINGW64__

uint32_t
lrand48();

float
drand48();

int64_t
random();

#endif
