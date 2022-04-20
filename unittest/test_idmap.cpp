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

#include <iostream>
#include <thread>

#include "knowhere/common/Exception.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuCloner.h>
#include "knowhere/index/vector_index/gpu/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif
#include "knowhere/utils/distances_simd.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

typedef float (*fvec_func_ptr)(const float*, const float*, size_t);

class IDMAPTest : public DataGen, public TestWithParam<knowhere::IndexMode> {
 protected:
    void
    SetUp() override {
        Init_with_default();
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICEID, PINMEM, TEMPMEM, RESNUM);
#endif
        index_mode_ = GetParam();
        index_ = std::make_shared<knowhere::IDMAP>();
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

    void
    RunRangeSearchBF(
        std::vector<std::vector<bool>>& golden_result,
        std::vector<size_t>& golden_cnt,
        float radius,
        fvec_func_ptr func,
        bool check_small,
        const faiss::BitsetView bitset = nullptr) {
        for (auto i = 0; i < nq; ++i) {
            const float* pq = xq.data() + i * dim;
            for (auto j = 0; j < nb; ++j) {
                if (bitset.empty() || !bitset.test(j)) {
                    const float* pb = xb.data() + j * dim;
                    auto dist = func(pq, pb, dim);
                    if ((check_small && dist < radius) || (!check_small && dist > radius)) {
                        golden_result[i][j] = true;
                        golden_cnt[i]++;
                    }
                }
            }
        }
    };

    void
    CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const int nq,
        const std::vector<std::vector<bool>>& golden_result,
        const std::vector<size_t>& golden_cnt) {
        auto lims = result->Get<size_t*>(knowhere::meta::LIMS);
        auto ids = result->Get<int64_t*>(knowhere::meta::IDS);
        for (auto i = 0; i < nq; ++i) {
            int correct_cnt = 0;
            for (auto j = lims[i]; j < lims[i+1]; j++) {
                auto idx = ids[j];
                ASSERT_EQ(golden_result[i][idx], true);
                if (golden_result[i][idx]) {
                    correct_cnt++;
                }
            }
            ASSERT_EQ(correct_cnt, golden_cnt[i]);
        }
    }

 protected:
    knowhere::IDMAPPtr index_ = nullptr;
    knowhere::IndexMode index_mode_;
};

INSTANTIATE_TEST_CASE_P(
    IDMAPParameters,
    IDMAPTest,
    Values(
#ifdef KNOWHERE_GPU_VERSION
        knowhere::IndexMode::MODE_GPU,
#endif
        knowhere::IndexMode::MODE_CPU)
    );

TEST_P(IDMAPTest, idmap_basic) {
    ASSERT_TRUE(!xb.empty());

    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::L2}
    };

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf));
    }

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_TRUE(index_->GetRawVectors() != nullptr);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICEID, conf));
    }
#endif

    auto binaryset = index_->Serialize(conf);
    auto new_index = std::make_shared<knowhere::IDMAP>();
    new_index->Load(binaryset);
    auto result2 = new_index->Query(query_dataset, conf, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(re_result, nq, k);

#if 0
    auto result3 = new_index->QueryById(id_dataset, conf);
    AssertAnns(result3, nq, k);

    auto result4 = new_index->GetVectorById(xid_dataset, conf);
    AssertVec(result4, base_dataset, xid_dataset, 1, dim);
#endif

    std::shared_ptr<uint8_t[]> data(new uint8_t[nb/8]);
    for (int64_t i = 0; i < nq; ++i) {
        set_bit(data.get(), i);
    }
    auto bitset = faiss::BitsetView(data.get(), nb);
    auto result_bs_1 = index_->Query(query_dataset, conf, bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);

#if 0
    auto result_bs_2 = index_->QueryById(id_dataset, conf);
    AssertAnns(result_bs_2, nq, k, CheckMode::CHECK_NOT_EQUAL);

    auto result_bs_3 = index_->GetVectorById(xid_dataset, conf);
    AssertVec(result_bs_3, base_dataset, xid_dataset, 1, dim, CheckMode::CHECK_NOT_EQUAL);
#endif
}

TEST_P(IDMAPTest, idmap_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::L2}
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICEID, conf));
    }
#endif

    auto re_result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(re_result, nq, k);
    // PrintResult(re_result, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf);
    auto bin = binaryset.GetByName("IVF");

    std::string filename = temp_path("/tmp/idmap_test_serialize.bin");
    auto load_data = new uint8_t[bin->size];
    serialize(filename, bin, load_data);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> data(load_data);
    binaryset.Append("IVF", data, bin->size);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);
}

TEST_P(IDMAPTest, idmap_slice) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
        {knowhere::Metric::TYPE, knowhere::Metric::L2}
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICEID, conf));
    }
#endif

    auto re_result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(re_result, nq, k);
    // PrintResult(re_result, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);
}

TEST_P(IDMAPTest, idmap_range_search_l2) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::IndexParams::range_search_radius, radius},
        {knowhere::Metric::TYPE, knowhere::Metric::L2}
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    // test without bitset
    {
        std::vector<std::vector<bool>> golden_result(nq, std::vector<bool>(nb, false));
        std::vector<size_t> golden_cnt(nq, 0);
        RunRangeSearchBF(golden_result, golden_cnt, radius * radius, faiss::fvec_L2sqr_ref, true);

        auto result = index_->QueryByRange(qd, conf, nullptr);
        CheckRangeSearchResult(result, nq, golden_result, golden_cnt);
    }

    auto binaryset = index_->Serialize(conf);
    index_->Load(binaryset);

    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // test with bitset
    {
        std::shared_ptr<uint8_t[]> data(new uint8_t[nb / 8]);
        for (int64_t i = 0; i < nb; i += 2) {
            set_bit(data.get(), i);
        }
        auto bitset = faiss::BitsetView(data.get(), nb);

        std::vector<std::vector<bool>> golden_result(nq, std::vector<bool>(nb, false));
        std::vector<size_t> golden_cnt(nq, 0);
        RunRangeSearchBF(golden_result, golden_cnt, radius * radius, faiss::fvec_L2sqr_ref, true, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult(result, nq, golden_result, golden_cnt);
    }
}

TEST_P(IDMAPTest, idmap_range_search_ip) {
    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::IndexParams::range_search_radius, radius},
        {knowhere::Metric::TYPE, knowhere::Metric::IP}
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    // test without bitset
    {
        std::vector<std::vector<bool>> golden_result(nq, std::vector<bool>(nb, false));
        std::vector<size_t> golden_cnt(nq, 0);
        RunRangeSearchBF(golden_result, golden_cnt, radius, faiss::fvec_inner_product_ref, false);

        auto result = index_->QueryByRange(qd, conf, nullptr);
        CheckRangeSearchResult(result, nq, golden_result, golden_cnt);
    }

    auto binaryset = index_->Serialize(conf);
    index_->Load(binaryset);

    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // test with bitset
    {
        std::shared_ptr<uint8_t[]> data(new uint8_t[nb / 8]);
        for (int64_t i = 0; i < nb; i += 2) {
            set_bit(data.get(), i);
        }
        auto bitset = faiss::BitsetView(data.get(), nb);

        std::vector<std::vector<bool>> golden_result(nq, std::vector<bool>(nb, false));
        std::vector<size_t> golden_cnt(nq, 0);
        RunRangeSearchBF(golden_result, golden_cnt, radius, faiss::fvec_inner_product_ref, false, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult(result, nq, golden_result, golden_cnt);
    }
}

#ifdef KNOWHERE_GPU_VERSION
TEST_P(IDMAPTest, idmap_copy) {
    ASSERT_TRUE(!xb.empty());

    knowhere::Config conf{
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::Metric::TYPE, knowhere::Metric::L2}
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_TRUE(index_->GetRawVectors() != nullptr);
    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

    // clone
    // auto clone_index = index_->Clone();
    // auto clone_result = clone_index->Search(query_dataset, conf);
    // AssertAnns(clone_result, nq, k);

    // cpu to gpu
    ASSERT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, conf));
    auto clone_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICEID, conf);
    auto clone_result = clone_index->Query(query_dataset, conf, nullptr);

    AssertAnns(clone_result, nq, k);
    ASSERT_THROW({ std::static_pointer_cast<knowhere::GPUIDMAP>(clone_index)->GetRawVectors(); },
                 knowhere::KnowhereException);
    ASSERT_ANY_THROW(clone_index->Serialize(conf));

    auto binary = clone_index->Serialize(conf);
    clone_index->Load(binary);
    auto new_result = clone_index->Query(query_dataset, conf, nullptr);
    AssertAnns(new_result, nq, k);

    // auto clone_gpu_idx = clone_index->Clone();
    // auto clone_gpu_res = clone_gpu_idx->Search(query_dataset, conf);
    // AssertAnns(clone_gpu_res, nq, k);

    // gpu to cpu
    auto host_index = knowhere::cloner::CopyGpuToCpu(clone_index, conf);
    auto host_result = host_index->Query(query_dataset, conf, nullptr);
    AssertAnns(host_result, nq, k);
    ASSERT_TRUE(std::static_pointer_cast<knowhere::IDMAP>(host_index)->GetRawVectors() != nullptr);

    // gpu to gpu
    auto device_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICEID, conf);
    auto new_device_index =
        std::static_pointer_cast<knowhere::GPUIDMAP>(device_index)->CopyGpuToGpu(DEVICEID, conf);
    auto device_result = new_device_index->Query(query_dataset, conf, nullptr);
    AssertAnns(device_result, nq, k);
}
#endif
