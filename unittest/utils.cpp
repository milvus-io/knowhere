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

#include <gtest/gtest.h>
#include <math.h>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/utils/BitsetView.h"
#include "unittest/utils.h"

void
DataGen::Init_with_default(const bool is_binary) {
    Generate(dim, nb, nq, is_binary);
}

void
DataGen::Generate(const int dim, const int nb, const int nq, const bool is_binary) {
    this->dim = dim;
    this->nb = nb;
    this->nq = nq;

    if (!is_binary) {
        GenAll(dim, nb, xb, ids, nq, xq);
        assert(xb.size() == (size_t)dim * nb);
        assert(xq.size() == (size_t)dim * nq);

        base_dataset = knowhere::GenDataset(nb, dim, xb.data());
        query_dataset = knowhere::GenDataset(nq, dim, xq.data());
    } else {
        int64_t dim_x = dim / 8;
        GenAll(dim_x, nb, xb_bin, ids, nq, xq_bin);
        assert(xb_bin.size() == (size_t)dim_x * nb);
        assert(xq_bin.size() == (size_t)dim_x * nq);

        base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
    }

    // used to test GetVectorById [0, nq-1]
    id_dataset = knowhere::GenDatasetWithIds(nq, dim, ids.data());

    bitset_data.resize(nb/8);
    for (int64_t i = 0; i < nq; ++i) {
        set_bit(bitset_data.data(), i);
    }
    bitset = std::make_shared<faiss::BitsetView>(bitset_data.data(), nb);
}

void
GenAll(const int64_t dim,
       const int64_t nb,
       std::vector<float>& xb,
       std::vector<int64_t>& ids,
       const int64_t nq,
       std::vector<float>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), false);
}

void
GenAll(const int64_t dim,
       const int64_t nb,
       std::vector<uint8_t>& xb,
       std::vector<int64_t>& ids,
       const int64_t nq,
       std::vector<uint8_t>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), true);
}

void
GenBase(const int64_t dim,
        const int64_t nb,
        const void* xb,
        int64_t* ids,
        const int64_t nq,
        const void* xq,
        bool is_binary) {
    if (!is_binary) {
        float* xb_f = (float*)xb;
        float* xq_f = (float*)xq;
        for (auto i = 0; i < nb; ++i) {
            for (auto j = 0; j < dim; ++j) {
                xb_f[i * dim + j] = drand48();
            }
            xb_f[dim * i] += i / 1000.;
            ids[i] = i;
        }
        for (int64_t i = 0; i < nq * dim; ++i) {
            xq_f[i] = xb_f[i];
        }
    } else {
        uint8_t* xb_u = (uint8_t*)xb;
        uint8_t* xq_u = (uint8_t*)xq;
        for (auto i = 0; i < nb; ++i) {
            for (auto j = 0; j < dim; ++j) {
                xb_u[i * dim + j] = (uint8_t)lrand48();
            }
            xb_u[dim * i] += i / 1000.;
            ids[i] = i;
        }
        for (int64_t i = 0; i < nq * dim; ++i) {
            xq_u[i] = xb_u[i];
        }
    }
}

void
AssertAnns(const knowhere::DatasetPtr& result,
           const int nq,
           const int k,
           const CheckMode check_mode) {
    auto ids = knowhere::GetDatasetIDs(result);
    for (auto i = 0; i < nq; i++) {
        auto id = ids[i * k];
        switch (check_mode) {
            case CheckMode::CHECK_EQUAL:
                ASSERT_EQ(i, id);
                break;
            case CheckMode::CHECK_NOT_EQUAL:
                ASSERT_NE(i, id);
                break;
            default:
                ASSERT_TRUE(false);
                break;
        }
    }
}

void
AssertDist(const knowhere::DatasetPtr& result,
           const knowhere::MetricType& metric,
           const int nq,
           const int k) {
    auto distance = knowhere::GetDatasetDistance(result);
    bool is_ip = (metric == knowhere::metric::IP);
    for (auto i = 0; i < nq; i++) {
        for (auto j = 1; j < k; j++) {
            auto va = distance[i * k + j - 1];
            auto vb = distance[i * k + j];
            if (is_ip) {
                ASSERT_GE(va, vb);  // descending order
            } else {
                ASSERT_LE(va, vb);  // ascending order
            }
        }
    }
}

void
AssertVec(const knowhere::DatasetPtr& result,
          const knowhere::DatasetPtr& base_dataset,
          const knowhere::DatasetPtr& id_dataset,
          const int n,
          const int dim) {
    float* base = (float*)knowhere::GetDatasetTensor(base_dataset);
    auto ids = knowhere::GetDatasetInputIDs(id_dataset);
    auto x = (float*)knowhere::GetDatasetTensor(result);
    for (auto i = 0; i < n; i++) {
        auto id = ids[i];
        for (auto j = 0; j < dim; j++) {
            float va = *(base + id * dim + j);
            float vb = *(x + i * dim + j);
            ASSERT_EQ(va, vb);
        }
    }
}

void
AssertBinVec(const knowhere::DatasetPtr& result,
             const knowhere::DatasetPtr& base_dataset,
             const knowhere::DatasetPtr& id_dataset,
             const int n,
             const int dim) {
    auto base = (uint8_t*)knowhere::GetDatasetTensor(base_dataset);
    auto ids = knowhere::GetDatasetInputIDs(id_dataset);
    auto x = (uint8_t*)knowhere::GetDatasetTensor(result);
    int dim_uint8 = dim / 8;
    for (auto i = 0; i < n; i++) {
        auto id = ids[i];
        for (auto j = 0; j < dim_uint8; j++) {
            uint8_t va = *(base + id * dim_uint8 + j);
            uint8_t vb = *(x + i * dim_uint8 + j);
            ASSERT_EQ(va, vb);
        }
    }
}

void
normalize(float* vec, int64_t n, int64_t dim) {
    for (int64_t i = 0; i < n; i++) {
        double vecLen = 0.0, inv_vecLen = 0.0;
        for (int64_t j = 0; j < dim; j++) {
            double val = vec[i * dim + j];
            vecLen += val * val;
        }
        inv_vecLen = 1.0 / std::sqrt(vecLen);
        for (int64_t j = 0; j < dim; j++) {
            vec[i * dim + j] = (float)(vec[i * dim + j] * inv_vecLen);
        }
    }
}

void
PrintResult(const knowhere::DatasetPtr& result, const int& nq, const int& k) {
    auto ids = knowhere::GetDatasetIDs(result);
    auto dist = knowhere::GetDatasetDistance(result);

    std::stringstream ss_id;
    std::stringstream ss_dist;
    for (auto i = 0; i < nq; i++) {
        for (auto j = 0; j < k; ++j) {
            // ss_id << *(ids->data()->GetValues<int64_t>(1, i * k + j)) << " ";
            // ss_dist << *(dists->data()->GetValues<float>(1, i * k + j)) << " ";
            ss_id << *((int64_t*)(ids) + i * k + j) << " ";
            ss_dist << *((float*)(dist) + i * k + j) << " ";
        }
        ss_id << std::endl;
        ss_dist << std::endl;
    }
    std::cout << "id\n" << ss_id.str() << std::endl;
    std::cout << "dist\n" << ss_dist.str() << std::endl;
}

// path like /tmp may not work for windows
std::string
temp_path(const char* path)
{
    std::string new_path{path};
#ifdef WIN32
    for (auto &ch : new_path) {
        if (ch == '/') {
            ch = '_';
        }
    }
    new_path = std::string("tmp/") + new_path;
    mkdir("tmp/");
#endif
    return new_path;
}

#ifdef __MINGW64__

static std::random_device rd;
static std::mt19937 gen(rd());

uint32_t lrand48 () {
    std::uniform_int_distribution<uint32_t> distrib(0, (1 << 31));
    return distrib(gen);
}

float drand48 () {
    std::uniform_real_distribution<float> distrib(0.0, 1.0);
    return distrib(gen);
}

int64_t random() {
    std::uniform_int_distribution<int64_t> distrib(0, (1LL << 63));
    return distrib(gen);
}

#endif
