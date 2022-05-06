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
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#ifdef KNOWHERE_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFSQ.h"
#include "knowhere/index/vector_index/gpu/IndexIVFSQHybrid.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

#include "unittest/Helper.h"
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
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICEID, PINMEM, TEMPMEM, RESNUM);
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

    template <class C>
    void CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const float radius) {

        auto lims = result->Get<size_t*>(knowhere::Meta::LIMS);
        auto distances = result->Get<float*>(knowhere::Meta::DISTANCE);

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
    IVFTest,
    Values(
#ifdef KNOWHERE_GPU_VERSION
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::IndexMode::MODE_GPU),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, knowhere::IndexMode::MODE_GPU),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H, knowhere::IndexMode::MODE_GPU),
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
    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
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
    AssertAnns(result, nq, conf_[knowhere::Meta::TOPK]);
}

TEST_P(IVFTest, ivf_slice) {
    // serialize index
    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, conf_[knowhere::Meta::TOPK]);
}

TEST_P(IVFTest, ivf_range_search_l2) {
    conf_[knowhere::Meta::METRIC_TYPE] = knowhere::MetricEnum::L2;

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        conf_[knowhere::Meta::RADIUS] = radius;
        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius * radius);
    };

    for (float radius: {4.0, 5.0, 6.0}) {
        test_range_search_l2(radius, nullptr);
        test_range_search_l2(radius, *bitset);
    }
}

TEST_P(IVFTest, ivf_range_search_ip) {
    conf_[knowhere::Meta::METRIC_TYPE] = knowhere::MetricEnum::IP;

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        conf_[knowhere::Meta::RADIUS] = radius;
        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult<CMax<float>>(result, radius);
    };

    for (float radius: {30.0, 35.0, 40.0}) {
        test_range_search_ip(radius, nullptr);
        test_range_search_ip(radius, *bitset);
    }
}

// TODO(linxj): deprecated
#ifdef KNOWHERE_GPU_VERSION
TEST_P(IVFTest, clone_test) {
    assert(!xb.empty());

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    /* set peseodo index size, avoid throw exception */
    index_->SetIndexSize(nq * dim * sizeof(float));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, conf_[knowhere::Meta::TOPK]);
    // PrintResult(result, nq, k);

    auto AssertEqual = [&](knowhere::DatasetPtr p1, knowhere::DatasetPtr p2) {
        auto ids_p1 = p1->Get<int64_t*>(knowhere::Meta::IDS);
        auto ids_p2 = p2->Get<int64_t*>(knowhere::Meta::IDS);

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

    {
        // copy to gpu
        if (index_type_ != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H) {
            EXPECT_NO_THROW({
                auto clone_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICEID, knowhere::Config());
                auto clone_result = clone_index->Query(query_dataset, conf_, nullptr);
                AssertEqual(result, clone_result);
                std::cout << "clone C <=> G [" << index_type_ << "] success" << std::endl;
            });
            EXPECT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, knowhere::Config()));
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
    ASSERT_ANY_THROW(index_->Seal());

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    /* set peseodo index size, avoid throw exception */
    index_->SetIndexSize(nq * dim * sizeof(float));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, conf_[knowhere::Meta::TOPK]);
    ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
    ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));

    auto cpu_idx = knowhere::cloner::CopyGpuToCpu(index_, knowhere::Config());
    knowhere::IVFPtr ivf_idx = std::dynamic_pointer_cast<knowhere::IVF>(cpu_idx);

    knowhere::TimeRecorder tc("CopyToGpu");
    knowhere::cloner::CopyCpuToGpu(cpu_idx, DEVICEID, knowhere::Config());
    auto without_seal = tc.RecordSection("Without seal");
    ivf_idx->Seal();
    tc.RecordSection("seal cost");
    knowhere::cloner::CopyCpuToGpu(cpu_idx, DEVICEID, knowhere::Config());
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
    invalid_conf[knowhere::Meta::DEVICEID] = -1;

    // if (index_type_ == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
    //     null faiss index
    //     index_->SetIndexSize(0);
    //     knowhere::cloner::CopyGpuToCpu(index_, knowhere::Config());
    // }

    index_->Train(base_dataset, conf_);
    ASSERT_ANY_THROW(index_->Serialize(conf_));
    ASSERT_ANY_THROW(index_->Query(base_dataset, invalid_conf, nullptr));

    auto ivf_index = std::dynamic_pointer_cast<knowhere::GPUIVF>(index_);
    if (ivf_index) {
        auto gpu_index = std::dynamic_pointer_cast<knowhere::GPUIndex>(ivf_index);
        gpu_index->SetGpuDevice(-1);
        ASSERT_EQ(gpu_index->GetGpuDevice(), -1);
    }

    // ASSERT_ANY_THROW(index_->Load(binaryset));
    ASSERT_ANY_THROW(index_->Train(base_dataset, invalid_conf));
}

TEST_P(IVFTest, IVFSQHybrid_test) {
    if (index_type_ != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H) {
        return;
    }

    index_->SetIndexSize(0);
    knowhere::cloner::CopyGpuToCpu(index_, conf_);
    ASSERT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, conf_));
    ASSERT_ANY_THROW(index_->Train(base_dataset, conf_));
    ASSERT_ANY_THROW(index_->CopyCpuToGpu(DEVICEID, conf_));

    index_->Train(base_dataset, conf_);
    auto index = std::dynamic_pointer_cast<knowhere::IVFSQHybrid>(index_);
    ASSERT_TRUE(index != nullptr);
    ASSERT_ANY_THROW(index->UnsetQuantizer());

    ASSERT_ANY_THROW(index->SetQuantizer(nullptr));
}
#endif
