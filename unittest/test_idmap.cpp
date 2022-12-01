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
#include <thread>
#include <unordered_set>

#include "knowhere/common/Exception.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#ifdef KNOWHERE_GPU_VERSION
#include <faiss/gpu/GpuCloner.h>
#include "knowhere/index/vector_index/gpu/IndexGPUIDMAP.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif
#include "unittest/range_utils.h"
#include "unittest/utils.h"
#include "unittest/Helper.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class IDMAPTest : public DataGen, public TestWithParam<knowhere::IndexMode> {
 protected:
    void
    SetUp() override {
        Init_with_default();
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICE_ID, PINMEM, TEMPMEM, RESNUM);
#endif
        conf_ = knowhere::Config{
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::meta::DIM, dim},
            {knowhere::meta::TOPK, k},
            {knowhere::meta::QUERY_OMP_NUM, QUERY_OMP_NUM},
        };
        index_mode_ = GetParam();
        index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;
        index_ = std::make_shared<knowhere::IDMAP>();
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

 protected:
    knowhere::Config conf_;
    knowhere::IDMAPPtr index_ = nullptr;
    knowhere::IndexMode index_mode_;
    knowhere::IndexType index_type_;
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

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf_));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
    }

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, k);
    // PrintResult(result1, nq, k);

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICE_ID, conf_));
    }
#endif

    auto binaryset = index_->Serialize(conf_);
    auto new_index = std::make_shared<knowhere::IDMAP>();
    new_index->Load(binaryset);
    auto result2 = new_index->Query(query_dataset, conf_, nullptr);
    AssertAnns(result2, nq, k);
    // PrintResult(re_result, nq, k);

    // query with bitset
    auto result_bs_1 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs_1, nq, k, CheckMode::CHECK_NOT_EQUAL);
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

    index_->BuildAll(base_dataset, conf_);

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICE_ID, conf_));
    }
#endif

    auto re_result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(re_result, nq, k);
    // PrintResult(re_result, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf_);
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
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);
}

TEST_P(IDMAPTest, idmap_slice) {
    index_->BuildAll(base_dataset, conf_);

#ifdef KNOWHERE_GPU_VERSION
    if (index_mode_ == knowhere::IndexMode::MODE_GPU) {
        // cpu to gpu
        index_ = std::dynamic_pointer_cast<knowhere::IDMAP>(index_->CopyCpuToGpu(DEVICE_ID, conf_));
    }
#endif

    auto re_result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(re_result, nq, k);
    // PrintResult(re_result, nq, k);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto binaryset = index_->Serialize(conf_);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);
}

TEST_P(IDMAPTest, idmap_range_search_l2) {
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
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
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
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(IDMAPTest, idmap_range_search_ip) {
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
                               golden_labels.data(), golden_lims.data(), true, bitset);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (std::pair<float, float> range: {
            std::make_pair<float, float>(0.70f, 0.75f),
            std::make_pair<float, float>(0.75f, 0.80f),
            std::make_pair<float, float>(0.80f, 1.01f)}) {
            knowhere::SetMetaRadiusLowBound(conf_, range.first);
            knowhere::SetMetaRadiusHighBound(conf_, range.second);
            test_range_search_ip(range.first, range.second, nullptr);
            test_range_search_ip(range.first, range.second, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

#ifdef KNOWHERE_GPU_VERSION
TEST_P(IDMAPTest, idmap_copy) {
    ASSERT_TRUE(!xb.empty());

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    // PrintResult(result, nq, k);

    // clone
    // auto clone_index = index_->Clone();
    // auto clone_result = clone_index->Search(query_dataset, conf_);
    // AssertAnns(clone_result, nq, k);

    // cpu to gpu
    ASSERT_ANY_THROW(knowhere::cloner::CopyCpuToGpu(index_, -1, conf_));
    auto clone_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICE_ID, conf_);
    auto clone_result = clone_index->Query(query_dataset, conf_, nullptr);

    AssertAnns(clone_result, nq, k);

    auto binary = clone_index->Serialize(conf_);
    clone_index->Load(binary);
    auto new_result = clone_index->Query(query_dataset, conf_, nullptr);
    AssertAnns(new_result, nq, k);

    // auto clone_gpu_idx = clone_index->Clone();
    // auto clone_gpu_res = clone_gpu_idx->Search(query_dataset, conf_);
    // AssertAnns(clone_gpu_res, nq, k);

    // gpu to cpu
    auto host_index = knowhere::cloner::CopyGpuToCpu(clone_index, conf_);
    auto host_result = host_index->Query(query_dataset, conf_, nullptr);
    AssertAnns(host_result, nq, k);

    // gpu to gpu
    auto device_index = knowhere::cloner::CopyCpuToGpu(index_, DEVICE_ID, conf_);
    auto new_device_index =
        std::static_pointer_cast<knowhere::GPUIDMAP>(device_index)->CopyGpuToGpu(DEVICE_ID, conf_);
    auto device_result = new_device_index->Query(query_dataset, conf_, nullptr);
    AssertAnns(device_result, nq, k);
}
#endif
