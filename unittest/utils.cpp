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
        GenAll(dim, nb, xb, ids, xids, nq, xq);
        assert(xb.size() == (size_t)dim * nb);
        assert(xq.size() == (size_t)dim * nq);

        base_dataset = knowhere::GenDataset(nb, dim, xb.data());
        query_dataset = knowhere::GenDataset(nq, dim, xq.data());
    } else {
        int64_t dim_x = dim / 8;
        GenAll(dim_x, nb, xb_bin, ids, xids, nq, xq_bin);
        assert(xb_bin.size() == (size_t)dim_x * nb);
        assert(xq_bin.size() == (size_t)dim_x * nq);

        base_dataset = knowhere::GenDataset(nb, dim, xb_bin.data());
        query_dataset = knowhere::GenDataset(nq, dim, xq_bin.data());
    }

    id_dataset = knowhere::GenDataset(nq, dim, nullptr);
    xid_dataset = knowhere::GenDataset(nq, dim, nullptr);

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
       std::vector<int64_t>& xids,
       const int64_t nq,
       std::vector<float>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    xids.resize(1);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), xids.data(), false);
}

void
GenAll(const int64_t dim,
       const int64_t nb,
       std::vector<uint8_t>& xb,
       std::vector<int64_t>& ids,
       std::vector<int64_t>& xids,
       const int64_t nq,
       std::vector<uint8_t>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    xids.resize(1);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), xids.data(), true);
}

void
GenBase(const int64_t dim,
        const int64_t nb,
        const void* xb,
        int64_t* ids,
        const int64_t nq,
        const void* xq,
        int64_t* xids,
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
    xids[0] = 3;  // pseudo random
}

void
AssertAnns(const knowhere::DatasetPtr& result, const int nq, const int k, const CheckMode check_mode) {
    auto ids = knowhere::GetDatasetIDs(result);
    for (auto i = 0; i < nq; i++) {
        switch (check_mode) {
            case CheckMode::CHECK_EQUAL:
                ASSERT_EQ(i, *((int64_t*)(ids) + i * k));
                break;
            case CheckMode::CHECK_NOT_EQUAL:
                ASSERT_NE(i, *((int64_t*)(ids) + i * k));
                break;
            default:
                ASSERT_TRUE(false);
                break;
        }
    }
}

#if 0
void
AssertVec(const knowhere::DatasetPtr& result, const knowhere::DatasetPtr& base_dataset,
          const knowhere::DatasetPtr& id_dataset, const int n, const int dim, const CheckMode check_mode) {
    float* base = (float*)knowhere::GetDatasetTensor(base_dataset);
    auto ids = knowhere::GetDatasetIDs(id_dataset);
    auto x = (float*)knowhere::GetDatasetTensor(result);
    for (auto i = 0; i < n; i++) {
        auto id = ids[i];
        for (auto j = 0; j < dim; j++) {
            switch (check_mode) {
                case CheckMode::CHECK_EQUAL: {
                    ASSERT_EQ(*(base + id * dim + j), *(x + i * dim + j));
                    break;
                }
                case CheckMode::CHECK_NOT_EQUAL: {
                    ASSERT_NE(*(base + id * dim + j), *(x + i * dim + j));
                    break;
                }
                case CheckMode::CHECK_APPROXIMATE_EQUAL: {
                    float a = *(base + id * dim + j);
                    float b = *(x + i * dim + j);
                    ASSERT_TRUE((std::fabs(a - b) / std::fabs(a)) < 0.1);
                    break;
                }
                default:
                    ASSERT_TRUE(false);
                    break;
            }
        }
    }
}

void
AssertBinVec(const knowhere::DatasetPtr& result, const knowhere::DatasetPtr& base_dataset,
             const knowhere::DatasetPtr& id_dataset, const int n, const int dim, const CheckMode check_mode) {
    auto base = (uint8_t*)knowhere::GetDatasetTensor(base_dataset);
    auto ids = knowhere::GetDatasetIDs(id_dataset;
    auto x = (float*)knowhere::GetDatasetTensor(result);
    for (auto i = 0; i < 1; i++) {
        auto id = ids[i];
        for (auto j = 0; j < dim; j++) {
            ASSERT_EQ(*(base + id * dim + j), *(x + i * dim + j));
        }
    }
}
#endif

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

// not used
#if 0
void
Load_nns_graph(std::vector<std::vector<int64_t>>& final_graph, const char* filename) {
    std::vector<std::vector<unsigned>> knng;

    std::ifstream in(filename, std::ios::binary);
    unsigned k;
    in.read((char*)&k, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = (size_t)(fsize / (k + 1) / 4);
    in.seekg(0, std::ios::beg);

    knng.resize(num);
    knng.reserve(num);
    int64_t kk = (k + 3) / 4 * 4;
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        knng[i].resize(k);
        knng[i].reserve(kk);
        in.read((char*)knng[i].data(), k * sizeof(unsigned));
    }
    in.close();

    final_graph.resize(knng.size());
    for (int i = 0; i < knng.size(); ++i) {
        final_graph[i].resize(knng[i].size());
        for (int j = 0; j < knng[i].size(); ++j) {
            final_graph[i][j] = knng[i][j];
        }
    }
}

float*
fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int*  // not very clean, but works as long as sizeof(int) == sizeof(float)
ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}
#endif

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
