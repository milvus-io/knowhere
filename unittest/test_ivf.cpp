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
#include "knowhere/common/Timer.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class IVFTest : public DataGen,
                public TestWithParam<::std::tuple<knowhere::IndexType, knowhere::IndexMode>> {
 protected:
    void
    SetUp() override {
        Init_with_default();
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICE_ID, PINMEM, TEMPMEM, RESNUM);
#endif
        std::tie(index_type_, index_mode_) = GetParam();
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        // conf_->Dump();
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
    IVFTest,
    Values(
#ifdef KNOWHERE_GPU_VERSION
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::IndexMode::MODE_GPU),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, knowhere::IndexMode::MODE_GPU),
#endif
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::IndexMode::MODE_CPU),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, knowhere::IndexMode::MODE_CPU)));

TEST_P(IVFTest, ivf_basic) {
    assert(!xb.empty());

    // null faiss index
    ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);
    if (index_mode_ == knowhere::IndexMode::MODE_CPU) {
        ASSERT_ANY_THROW(index_->GetVectorById(id_dataset, conf_));
    }

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

    auto result_bs_1 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);
    // PrintResult(result, nq, k);

#ifdef KNOWHERE_GPU_VERSION
    knowhere::FaissGpuResourceMgr::GetInstance().Dump();
#endif
}

TEST_P(IVFTest, ivf_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);
    auto bin = binaryset.GetByName("IVF");

    std::string filename = temp_path("/tmp/ivf_test_serialize.bin");
    auto load_data = new uint8_t[bin->size];
    serialize(filename, bin, load_data);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> data(load_data);
    binaryset.Append("IVF", data, bin->size);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
}

TEST_P(IVFTest, ivf_slice) {
    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
}

TEST_P(IVFTest, ivf_range_search_l2) {
    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }
    knowhere::MetricType metric_type = knowhere::metric::L2;
    knowhere::SetMetaMetricType(conf_, metric_type);

    index_->BuildAll(base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::L2,
                              xb.data(), nb, xq.data(), nq, dim, low_bound, high_bound, bitset);

        auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
        ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), false, bitset);
    };

    for (std::pair<float, float> range: {
             std::make_pair<float, float>(0, 16.81f),
             std::make_pair<float, float>(16.81f, 17.64f),
             std::make_pair<float, float>(17.64f, 18.49f)}) {
        knowhere::SetMetaRadiusLowBound(conf_, range.first);
        knowhere::SetMetaRadiusHighBound(conf_, range.second);
        test_range_search_l2(range.first, range.second, nullptr);
        test_range_search_l2(range.first, range.second, *bitset);
    }
}

TEST_P(IVFTest, ivf_range_search_ip) {
    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }
    knowhere::MetricType metric_type = knowhere::metric::IP;
    knowhere::SetMetaMetricType(conf_, metric_type);

    normalize(xb.data(), nb, dim);
    normalize(xq.data(), nq, dim);

    index_->BuildAll(base_dataset, conf_);

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](const float low_bound, const float high_bound, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::IP,
                              xb.data(), nb, xq.data(), nq, dim, low_bound, high_bound, bitset);

        auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
        ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult(result, metric_type, nq, low_bound, high_bound,
                               golden_labels.data(), golden_lims.data(), false, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.70f, 0.75f),
        std::make_pair<float, float>(0.75f, 0.80f),
        std::make_pair<float, float>(0.80f, 1.0f)}) {
        knowhere::SetMetaRadiusLowBound(conf_, range.first);
        knowhere::SetMetaRadiusHighBound(conf_, range.second);
        test_range_search_ip(range.first, range.second, nullptr);
        test_range_search_ip(range.first, range.second, *bitset);
    }
}

// TODO(linxj): deprecated
#ifdef KNOWHERE_GPU_VERSION
TEST_P(IVFTest, clone_test) {
    assert(!xb.empty());

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

    auto AssertEqual = [&](knowhere::DatasetPtr p1, knowhere::DatasetPtr p2) {
        auto ids_p1 = knowhere::GetDatasetIDs(p1);
        auto ids_p2 = knowhere::GetDatasetIDs(p2);

        for (int i = 0; i < nq * k; ++i) {
            EXPECT_EQ(*((int64_t*)(ids_p2) + i), *((int64_t*)(ids_p1) + i));
            //            EXPECT_EQ(*(ids_p2->data()->GetValues<int64_t>(1, i)), *(ids_p1->data()->GetValues<int64_t>(1,
            //            i)));
        }
    };

    {
        // copy from gpu to cpu
        if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
            EXPECT_NO_THROW({
                auto clone_index = knowhere::cloner::CopyGpuToCpu(index_, knowhere::Config());
                auto clone_result = clone_index->Query(query_dataset, conf_, nullptr);
                AssertEqual(result, clone_result);
                std::cout << "clone G <=> C [" << index_type_ << "] success" << std::endl;
            });
        } else {
            EXPECT_THROW(
                {
                    std::cout << "clone G <=> C [" << index_type_ << "] failed" << std::endl;
                    auto clone_index = knowhere::cloner::CopyGpuToCpu(index_, knowhere::Config());
                },
                knowhere::KnowhereException);
        }
    }
}
#endif

#ifdef KNOWHERE_GPU_VERSION
TEST_P(IVFTest, gpu_seal_test) {
    if (index_mode_ != knowhere::IndexMode::MODE_GPU) {
        return;
    }
    assert(!xb.empty());

    ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
    //ASSERT_ANY_THROW(index_->Seal());

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);

    auto cpu_idx = knowhere::cloner::CopyGpuToCpu(index_, knowhere::Config());
    knowhere::IVFPtr ivf_idx = std::dynamic_pointer_cast<knowhere::IVF>(cpu_idx);

    knowhere::TimeRecorder tc("CopyToGpu");
    knowhere::cloner::CopyCpuToGpu(cpu_idx, DEVICE_ID, knowhere::Config());
    auto without_seal = tc.RecordSection("Without seal");
    ivf_idx->Seal();
    tc.RecordSection("seal cost");
    knowhere::cloner::CopyCpuToGpu(cpu_idx, DEVICE_ID, knowhere::Config());
    auto with_seal = tc.RecordSection("With seal");
    ASSERT_GE(without_seal, with_seal);

    // copy to GPU with invalid device id
    ASSERT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(cpu_idx, -1, knowhere::Config()));
}

TEST_P(IVFTest, invalid_gpu_source) {
    if (index_mode_ != knowhere::IndexMode::MODE_GPU) {
        return;
    }

    auto invalid_conf = ParamGenerator::GetInstance().Gen(index_type_);
    knowhere::SetMetaDeviceID(invalid_conf, -1);

    index_->Train(base_dataset, conf_);

    auto ivf_index = std::dynamic_pointer_cast<knowhere::GPUIVF>(index_);
    if (ivf_index) {
        auto gpu_index = std::dynamic_pointer_cast<knowhere::GPUIndex>(ivf_index);
        gpu_index->SetGpuDevice(-1);
        ASSERT_EQ(gpu_index->GetGpuDevice(), -1);
    }

    // ASSERT_ANY_THROW(index_->Load(binaryset));
    ASSERT_ANY_THROW(index_->Train(base_dataset, invalid_conf));
}
#endif
