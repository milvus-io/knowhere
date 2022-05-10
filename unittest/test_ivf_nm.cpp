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

#include "unittest/Helper.h"
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
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICEID, PINMEM, TEMPMEM, RESNUM);
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

    template <class C>
    void CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const float radius) {

        auto lims = knowhere::GetDatasetLims(result);
        auto distances = knowhere::GetDatasetDistance(result);

        for (auto i = 0; i < lims[nq]; ++i) {
            ASSERT_TRUE(C::cmp(distances[i], radius));
        }
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

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    index_->SetIndexSize(nq * dim * sizeof(float));

    LoadRawData(index_, base_dataset, conf_);

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);

#ifdef KNOWHERE_GPU_VERSION
    // copy cpu to gpu
    if (index_mode_ == knowhere::IndexMode::MODE_CPU) {
        EXPECT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, knowhere::Config()));
        EXPECT_NO_THROW({
            auto clone_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICEID, conf_);
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
            AssertEqual(result, clone_result);
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
    knowhere::SetMetaMetricType(conf_, knowhere::metric::L2);

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    index_->SetIndexSize(nq * dim * sizeof(float));

    LoadRawData(index_, base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        knowhere::SetMetaRadius(conf_, radius);
        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius * radius);
    };

    for (float radius: {4.0, 4.5, 5.0}) {
        test_range_search_l2(radius, nullptr);
        test_range_search_l2(radius, *bitset);
    }
}

TEST_P(IVFNMTest, ivfnm_range_search_ip) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::IP);

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());
    index_->SetIndexSize(nq * dim * sizeof(float));

    LoadRawData(index_, base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        knowhere::SetMetaRadius(conf_, radius);
        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMax<float>>(result, radius);
    };

    for (float radius: {30.0, 35.0, 40.0}) {
        test_range_search_ip(radius, nullptr);
        test_range_search_ip(radius, *bitset);
    }
}
