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

#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuIndexIVFFlat.h>
#endif

#include "knowhere/common/Exception.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#include "knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.h"
#endif

#include "knowhere/utils/distances_simd.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class IVFNMTest : public DataGen,
                  public TestWithParam<::std::tuple<knowhere::IndexType, knowhere::IndexMode>> {
 protected:
    void
    SetUp() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICE_ID, PINMEM, TEMPMEM, RESNUM);
#endif
        std::tie(index_type_, index_mode_) = GetParam();
        Generate(DIM, NB, NQ);
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

 protected:
    knowhere::IndexType index_type_;
    knowhere::IndexMode index_mode_;
    knowhere::Config conf_;
    knowhere::VecIndexPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    IVFParameters,
    IVFNMTest,
    Values(
#ifdef KNOWHERE_GPU_VERSION
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, knowhere::IndexMode::MODE_GPU),
#endif
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, knowhere::IndexMode::MODE_CPU)));

void
LoadRawData(const knowhere::VecIndexPtr index,
            const knowhere::DatasetPtr dataset,
            const knowhere::Config& conf) {
    using namespace knowhere;
    GET_TENSOR_DATA_DIM(dataset)
    knowhere::BinarySet bs = index->Serialize(conf);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)p_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);
    index->Load(bs);
}

TEST_P(IVFNMTest, ivfnm_basic) {
    assert(!xb.empty());

    // null faiss index
    ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    LoadRawData(index_, base_dataset, conf_);

    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);

#ifdef KNOWHERE_GPU_VERSION
    // copy cpu to gpu
    if (index_mode_ == knowhere::IndexMode::MODE_CPU) {
        EXPECT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, knowhere::Config()));
        EXPECT_NO_THROW({
            auto clone_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICE_ID, conf_);
            auto clone_result = clone_index->Query(query_dataset, conf_, nullptr);
            AssertAnns(clone_result, nq, k);
            std::cout << "clone C <=> G [" << index_type_ << "] success" << std::endl;
        });
    }

    // copy gpu to cpu
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        EXPECT_NO_THROW({
            auto clone_index = knowhere::cloner::CopyGpuToCpu(index_, conf_);
            LoadRawData(clone_index, base_dataset, conf_);
            auto clone_result = clone_index->Query(query_dataset, conf_, nullptr);
            AssertAnns(clone_result, nq, k);
            std::cout << "clone G <=> C [" << index_type_ << "] success" << std::endl;
        });
    }
#endif

    auto result_bs_1 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);

#ifdef KNOWHERE_GPU_VERSION
    knowhere::FaissGpuResourceMgr::GetInstance().Dump();
#endif
}

TEST_P(IVFNMTest, ivfnm_range_search_l2) {
    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }
    knowhere::SetMetaMetricType(conf_, knowhere::metric::L2);

    index_->BuildAll(base_dataset, conf_);
    LoadRawData(index_, base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::L2,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, nq, radius * radius, golden_labels.data(), golden_lims.data(), false);
    };

    for (float radius: {4.1f, 4.2f, 4.3f}) {
        knowhere::SetMetaRadius(conf_, radius);
        test_range_search_l2(radius, nullptr);
        test_range_search_l2(radius, *bitset);
    }
}

TEST_P(IVFNMTest, ivfnm_range_search_ip) {
    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }
    knowhere::SetMetaMetricType(conf_, knowhere::metric::IP);

    index_->BuildAll(base_dataset, conf_);
    LoadRawData(index_, base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF<CMax<float>>(golden_labels, golden_distances, golden_lims, knowhere::metric::IP,
                                           xb.data(), nb, xq.data(), nq, dim, radius, bitset);

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMax<float>>(result, nq, radius, golden_labels.data(), golden_lims.data(), false);
    };

    for (float radius: {42.0f, 43.0f, 44.0f}) {
        knowhere::SetMetaRadius(conf_, radius);
        test_range_search_ip(radius, nullptr);
        test_range_search_ip(radius, *bitset);
    }
}
