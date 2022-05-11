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
        index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;
        index_ = std::make_shared<knowhere::IDMAP>();
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

    template <class C>
    void RunRangeSearchBF(
        std::vector<int64_t>& golden_labels,
        std::vector<float>& golden_distances,
        std::vector<size_t>& golden_lims,
        float radius,
        fvec_func_ptr func,
        const faiss::BitsetView bitset) {

        golden_lims.push_back(0);
        for (auto i = 0; i < nq; ++i) {
            const float* pq = xq.data() + i * dim;
            for (auto j = 0; j < nb; ++j) {
                if (bitset.empty() || !bitset.test(j)) {
                    const float* pb = xb.data() + j * dim;
                    auto dis = func(pq, pb, dim);
                    if (C::cmp(dis, radius)) {
                        golden_labels.push_back(j);
                        golden_distances.push_back(dis);
                    }
                }
            }
            golden_lims.push_back(golden_labels.size());
        }
    }

    void CompareRangeSearchResult(
        const int64_t* golden_labels,
        const float* golden_distances,
        const size_t golden_size,
        const int64_t* labels,
        const float* distances,
        const size_t size) {

        if (size == golden_size) {
            return;
        } else if (size > golden_size) {
            std::unordered_set<int64_t> golden_set(golden_labels, golden_labels + golden_size);
            size_t cnt = 0;
            for (auto i = 0; i < size; i++) {
                if (golden_set.find(labels[i]) == golden_set.end()) {
                    std::cout << "No." << cnt++ << " [" << labels[i] << ", " << distances[i] << "] "
                              << "not in GOLDEN result" << std::endl;
                }
            }
        } else {
            std::unordered_set<int64_t> test_set(labels, labels + size);
            size_t cnt = 0;
            for (auto i = 0; i < golden_size; i++) {
                if (test_set.find(golden_labels[i]) == test_set.end()) {
                    std::cout << "No." << cnt++ << " [" << golden_labels[i] << ", " << golden_distances[i] << "] "
                              << "not in TEST result" << std::endl;
                }
            }
        }
    }

    template <class C>
    void CheckRangeSearchResult(
        const knowhere::DatasetPtr& result,
        const float radius,
        const std::vector<int64_t>& golden_labels,
        const std::vector<float>& golden_distances,
        const std::vector<size_t>& golden_lims) {

        auto lims = knowhere::GetDatasetLims(result);
        auto ids = knowhere::GetDatasetIDs(result);
        auto distances = knowhere::GetDatasetDistance(result);

        for (int64_t i = 0; i < nq; i++) {
            if (golden_lims[i+1] != lims[i+1]) {
                std::cout << "No." << i << " range search fail" << std::endl;
                CompareRangeSearchResult(
                    golden_labels.data() + golden_lims[i],
                    golden_distances.data() + golden_lims[i],
                    golden_lims[i+1] - golden_lims[i],
                    ids + lims[i],
                    distances + lims[i],
                    lims[i+1] - lims[i]);
            }
            ASSERT_EQ(golden_lims[i+1], lims[i+1]);
            for (size_t j = lims[i]; j < lims[i+1]; j++) {
                ASSERT_EQ(golden_labels[j], ids[j]);
                ASSERT_TRUE(C::cmp(distances[j], radius));
            }
        }
    }

 protected:
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

    knowhere::Config conf{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
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
    ASSERT_GT(index_->Size(), 0);

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf, index_type_, index_mode_));

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

    // query with bitset
    auto result_bs_1 = index_->Query(query_dataset, conf, *bitset);
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

    knowhere::Config conf{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
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
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::DIM, dim},
        {knowhere::meta::TOPK, k},
        {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, knowhere::index_file_slice_size},
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
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::DIM, dim},
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_l2 = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, radius * radius,
                                      faiss::fvec_L2sqr_ref, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult<CMin<float>>(result, radius * radius, golden_labels, golden_distances, golden_lims);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (float radius: {4.0, 4.5, 5.0}) {
            knowhere::SetMetaRadius(conf, radius);
            test_range_search_l2(radius, nullptr);
            test_range_search_l2(radius, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
}

TEST_P(IDMAPTest, idmap_range_search_ip) {
    knowhere::Config conf{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::DIM, dim},
    };

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, knowhere::Config());

    auto qd = knowhere::GenDataset(nq, dim, xq.data());

    auto test_range_search_ip = [&](float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunRangeSearchBF<CMax<float>>(golden_labels, golden_distances, golden_lims, radius,
                                      faiss::fvec_inner_product_ref, bitset);

        auto result = index_->QueryByRange(qd, conf, bitset);
        CheckRangeSearchResult<CMax<float>>(result, radius, golden_labels, golden_distances, golden_lims);
    };

    auto old_blas_threshold = knowhere::KnowhereConfig::GetBlasThreshold();
    for (int64_t blas_threshold : {0, 20}) {
        knowhere::KnowhereConfig::SetBlasThreshold(blas_threshold);
        for (float radius: {30.0, 40.0, 45.0}) {
            knowhere::SetMetaRadius(conf, radius);
            test_range_search_ip(radius, nullptr);
            test_range_search_ip(radius, *bitset);
        }
    }
    knowhere::KnowhereConfig::SetBlasThreshold(old_blas_threshold);
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
