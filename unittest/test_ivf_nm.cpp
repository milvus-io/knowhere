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
#include "knowhere/feder/IVFFlat.h"
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
    
    if (index_mode_ == knowhere::IndexMode::MODE_CPU) {
        auto result = index_->GetVectorById(id_dataset, conf_);
        AssertVec(result, base_dataset, id_dataset, nq, dim);

        std::vector<int64_t> ids_invalid(nq, nb);
        auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
        ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));
    }
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
    knowhere::MetricType metric_type = knowhere::metric::L2;
    knowhere::SetMetaMetricType(conf_, metric_type);

    index_->BuildAll(base_dataset, conf_);
    LoadRawData(index_, base_dataset, conf_);

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

TEST_P(IVFNMTest, ivfnm_range_search_ip) {
    if (index_mode_ != knowhere::IndexMode::MODE_CPU) {
        return;
    }
    knowhere::MetricType metric_type = knowhere::metric::IP;
    knowhere::SetMetaMetricType(conf_, metric_type);

    normalize(xb.data(), nb, dim);
    normalize(xq.data(), nq, dim);

    index_->BuildAll(base_dataset, conf_);
    LoadRawData(index_, base_dataset, conf_);

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
        std::make_pair<float, float>(0.80f, 1.01f)}) {
        knowhere::SetMetaRadiusLowBound(conf_, range.first);
        knowhere::SetMetaRadiusHighBound(conf_, range.second);
        test_range_search_ip(range.first, range.second, nullptr);
        test_range_search_ip(range.first, range.second, *bitset);
    }
}

TEST_P(IVFNMTest, ivfnm_get_meta) {
    assert(!xb.empty());

    index_->BuildAll(base_dataset, conf_);
    LoadRawData(index_, base_dataset, conf_);

    auto qd = knowhere::GenDataset(1, dim, xq.data());
    index_->Query(qd, conf_, nullptr);

    auto result = index_->GetIndexMeta(conf_);

    auto json_info = knowhere::GetDatasetJsonInfo(result);
    auto json_id_set = knowhere::GetDatasetJsonIdSet(result);
    //std::cout << json_info << std::endl;
    std::cout << "json_info size = " << json_info.size() << std::endl;
    std::cout << "json_id_set size = " << json_id_set.size() << std::endl;

    // check IVFFlatMeta
    knowhere::feder::ivfflat::IVFFlatMeta meta;
    knowhere::Config j1 = nlohmann::json::parse(json_info);
    ASSERT_NO_THROW(nlohmann::from_json(j1, meta));

    ASSERT_EQ(meta.GetNlist(), knowhere::GetIndexParamNlist(conf_));
    ASSERT_EQ(meta.GetDim(), knowhere::GetMetaDim(conf_));
    ASSERT_EQ(meta.GetNtotal(), nb);

    // sum of all cluster nodes should be equal to nb
    auto& clusters = meta.GetClusters();
    std::unordered_set<int64_t> all_id_set;
    ASSERT_EQ(clusters.size(), knowhere::GetIndexParamNlist(conf_));
    for (auto& cluster : clusters) {
        for (auto id : cluster.node_ids_) {
            ASSERT_GE(id, 0);
            ASSERT_LT(id, nb);
            all_id_set.insert(id);
        }
    }
    ASSERT_EQ(all_id_set.size(), nb);

    // check IDSet
    std::unordered_set<int64_t> id_set;
    knowhere::Config j2 = nlohmann::json::parse(json_id_set);
    ASSERT_NO_THROW(nlohmann::from_json(j2, id_set));
    std::cout << "id_set num = " << id_set.size() << std::endl;
}