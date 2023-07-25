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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "common/range_util.h"
#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"

constexpr int64_t kSeed = 42;
using IdDisPair = std::pair<int64_t, float>;
struct DisPairLess {
    bool
    operator()(const IdDisPair& p1, const IdDisPair& p2) {
        return p1.second < p2.second;
    }
};

inline knowhere::DataSetPtr
GenDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> distrib(0.0, 100.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) ts[i] = (float)distrib(rng);
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
CopyDataSet(knowhere::DataSetPtr dataset, const int64_t copy_rows) {
    auto rows = copy_rows;
    auto dim = dataset->GetDim();
    auto data = dataset->GetTensor();
    float* ts = new float[rows * dim];
    memcpy(ts, data, rows * dim * sizeof(float));
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenBinDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> distrib(0.0, 100.0);
    int uint8_num = dim / 8;
    uint8_t* ts = new uint8_t[rows * uint8_num];
    for (int i = 0; i < rows * uint8_num; ++i) ts[i] = (uint8_t)distrib(rng);
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
CopyBinDataSet(knowhere::DataSetPtr dataset, const int64_t copy_rows) {
    auto rows = copy_rows;
    auto dim = dataset->GetDim();
    auto data = dataset->GetTensor();
    int uint8_num = dim / 8;
    uint8_t* ts = new uint8_t[rows * uint8_num];
    memcpy(ts, data, rows * uint8_num * sizeof(uint8_t));
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenIdsDataSet(int rows, int nq, int64_t seed = 42) {
    std::mt19937 g(seed);
    int64_t* ids = new int64_t[rows];
    for (int i = 0; i < rows; ++i) ids[i] = i;
    std::shuffle(ids, ids + rows, g);
    auto ds = knowhere::GenIdsDataSet(nq, ids);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenIdsDataSet(int rows, std::vector<int64_t>& ids) {
    auto ds = knowhere::GenIdsDataSet(rows, ids.data());
    ds->SetIsOwner(false);
    return ds;
}

inline float
GetKNNRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result) {
    REQUIRE(ground_truth.GetDim() >= result.GetDim());

    auto nq = result.GetRows();
    auto gt_k = ground_truth.GetDim();
    auto res_k = result.GetDim();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();

    uint32_t matched_num = 0;
    for (auto i = 0; i < nq; ++i) {
        std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + res_k);
        std::vector<int64_t> ids_1(res_ids + i * res_k, res_ids + i * res_k + res_k);

        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());

        std::vector<int64_t> v(std::max(ids_0.size(), ids_1.size()));
        std::vector<int64_t>::iterator it;
        it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it - v.begin());
        matched_num += v.size();
    }
    return ((float)matched_num) / ((float)nq * res_k);
}

inline float
GetRangeSearchRecall(const knowhere::DataSet& gt, const knowhere::DataSet& result) {
    uint32_t nq = result.GetRows();
    auto res_ids_p = result.GetIds();
    auto res_lims_p = result.GetLims();
    auto gt_ids_p = gt.GetIds();
    auto gt_lims_p = gt.GetLims();
    uint32_t ninter = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        std::set<int64_t> inter;
        std::set<int64_t> res_ids_set(res_ids_p + res_lims_p[i], res_ids_p + res_lims_p[i + 1]);
        std::set<int64_t> gt_ids_set(gt_ids_p + gt_lims_p[i], gt_ids_p + gt_lims_p[i + 1]);
        std::set_intersection(res_ids_set.begin(), res_ids_set.end(), gt_ids_set.begin(), gt_ids_set.end(),
                              std::inserter(inter, inter.begin()));
        ninter += inter.size();
    }

    float recall = ninter * 1.0f / gt_lims_p[nq];
    float precision = ninter * 1.0f / res_lims_p[nq];

    return (1 + precision) * recall / 2;
}

// Return a n-bits bitset data with first t bits set to true
inline std::vector<uint8_t>
GenerateBitsetWithFirstTbitsSet(size_t n, size_t t) {
    assert(t >= 0 && t <= n);
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    for (size_t i = 0; i < t; ++i) {
        data[i >> 3] |= (0x1 << (i & 0x7));
    }
    return data;
}

// Return a n-bits bitset data with random t bits set to true
inline std::vector<uint8_t>
GenerateBitsetWithRandomTbitsSet(size_t n, size_t t) {
    assert(t >= 0 && t <= n);
    std::vector<bool> bits_shuffle(n, false);
    for (size_t i = 0; i < t; ++i) bits_shuffle[i] = true;
    std::mt19937 g(kSeed);
    std::shuffle(bits_shuffle.begin(), bits_shuffle.end(), g);
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    for (size_t i = 0; i < n; ++i) {
        if (bits_shuffle[i]) {
            data[i >> 3] |= (0x1 << (i & 0x7));
        }
    }
    return data;
}

// Randomly generate n (distances, id) pairs
inline std::vector<std::pair<float, size_t>>
GenerateRandomDistanceIdPair(size_t n) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<> distrib(std::numeric_limits<float>().min(), std::numeric_limits<float>().max());
    std::vector<std::pair<float, size_t>> res;
    res.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        res.emplace_back(distrib(rng), i);
    }
    return res;
}
