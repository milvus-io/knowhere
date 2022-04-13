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

#include "easyloggingpp/easylogging++.h"

#ifdef KNOWHERE_GPU_VERSION

#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#include "wrapper/WrapperException.h"

#endif

#include <gtest/gtest.h>

#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "wrapper/VecIndex.h"
#include "wrapper/utils.h"

INITIALIZE_EASYLOGGINGPP

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class KnowhereWrapperTest
    : public DataGenBase,
      public TestWithParam<::std::tuple<milvus::engine::IndexType, std::string, int, int, int, int>> {
 protected:
    void
    SetUp() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().InitDevice(DEVICEID, PINMEM, TEMPMEM, RESNUM);
#endif
        std::string generator_type;
        std::tie(index_type, generator_type, dim, nb, nq, k) = GetParam();
        GenData(dim, nb, nq, xb, xq, ids, k, gt_ids, gt_dis);

        knowhere::Config tempconf{{knowhere::Metric::TYPE, knowhere::Metric::L2},
                                  {knowhere::meta::ROWS, nb},
                                  {knowhere::meta::DIM, dim},
                                  {knowhere::meta::TOPK, k},
                                  {knowhere::meta::DEVICEID, DEVICEID}};

        index_ = GetVecIndexFactory(index_type);
        conf = ParamGenerator::GetInstance().GenBuild(index_type, tempconf);
        searchconf = ParamGenerator::GetInstance().GenSearchConf(index_type, tempconf);
    }

    void
    TearDown() override {
#ifdef KNOWHERE_GPU_VERSION
        knowhere::FaissGpuResourceMgr::GetInstance().Free();
#endif
    }

 protected:
    milvus::engine::IndexType index_type;
    milvus::engine::VecIndexPtr index_ = nullptr;
    knowhere::Config conf;
    knowhere::Config searchconf;
};

INSTANTIATE_TEST_CASE_P(
    WrapperParam,
    KnowhereWrapperTest,
    Values(
//["Index type", "Generator type", "dim", "nb", "nq", "k", "build config", "search config"]
#ifdef KNOWHERE_GPU_VERSION
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFFLAT_GPU, "Default", DIM, NB, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFFLAT_MIX, "Default", 64, 1000, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFSQ8_GPU, "Default", DIM, NB, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFSQ8_MIX, "Default", DIM, NB, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFPQ_MIX, "Default", 64, 1000, 10, 10),
// std::make_tuple(milvus::engine::IndexType::NSG_MIX, "Default", 128, 250000, 10, 10),
#endif
        // std::make_tuple(milvus::engine::IndexType::SPTAG_KDT_RNT_CPU, "Default", 128, 100, 10, 10),
        // std::make_tuple(milvus::engine::IndexType::SPTAG_BKT_RNT_CPU, "Default", 126, 100, 10, 10),
        std::make_tuple(milvus::engine::IndexType::HNSW, "Default", 64, 10000, 5, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IDMAP, "Default", 64, 1000, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFFLAT_CPU, "Default", 64, 1000, 10, 10),
        std::make_tuple(milvus::engine::IndexType::FAISS_IVFSQ8_CPU, "Default", DIM, NB, 10, 10)));

#ifdef KNOWHERE_GPU_VERSION
TEST_P(KnowhereWrapperTest, WRAPPER_EXCEPTION_TEST) {
    std::string err_msg = "failed";
    milvus::engine::WrapperException ex(err_msg);

    std::string msg = ex.what();
    EXPECT_EQ(msg, err_msg);
}

#endif

TEST_P(KnowhereWrapperTest, BASE_TEST) {
    EXPECT_EQ(index_->GetType(), index_type);

    auto elems = nq * k;
    std::vector<int64_t> res_ids(elems);
    std::vector<float> res_dis(elems);

    index_->BuildAll(nb, xb.data(), ids.data(), conf);
    index_->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
    AssertResult(res_ids, res_dis);

    {
        index_->GetDeviceId();
        index_->BuildAll(nb, xb.data(), ids.data(), conf);
        index_->Add(nb, xb.data(), ids.data());
        index_->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
    }
}

#ifdef KNOWHERE_GPU_VERSION
TEST_P(KnowhereWrapperTest, TO_GPU_TEST) {
    if (index_type == milvus::engine::IndexType::HNSW) {
        return;
    }
    EXPECT_EQ(index_->GetType(), index_type);

    auto elems = nq * k;
    std::vector<int64_t> res_ids(elems);
    std::vector<float> res_dis(elems);

    index_->BuildAll(nb, xb.data(), ids.data(), conf);
    index_->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
    AssertResult(res_ids, res_dis);

    {
        auto dev_idx = index_->CopyToGpu(DEVICEID);
        for (int i = 0; i < 10; ++i) {
            dev_idx->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
        }
        AssertResult(res_ids, res_dis);
    }

    {
        std::string file_location = temp_path("/tmp/knowhere_gpu_file");
        write_index(index_, file_location);
        auto new_index = milvus::engine::read_index(file_location);

        auto dev_idx = new_index->CopyToGpu(DEVICEID);
        for (int i = 0; i < 10; ++i) {
            dev_idx->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
        }
        AssertResult(res_ids, res_dis);
    }
}

#endif

TEST_P(KnowhereWrapperTest, SERIALIZE_TEST) {
    std::cout << "type: " << static_cast<int>(index_type) << std::endl;
    EXPECT_EQ(index_->GetType(), index_type);

    auto elems = nq * k;
    std::vector<int64_t> res_ids(elems);
    std::vector<float> res_dis(elems);
    index_->BuildAll(nb, xb.data(), ids.data(), conf);
    index_->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
    AssertResult(res_ids, res_dis);

    {
        auto binary = index_->Serialize();
        auto type = index_->GetType();
        auto new_index = GetVecIndexFactory(type);
        new_index->Load(binary);
        EXPECT_EQ(new_index->Dimension(), index_->Dimension());
        EXPECT_EQ(new_index->Count(), index_->Count());

        std::vector<int64_t> res_ids(elems);
        std::vector<float> res_dis(elems);
        new_index->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
        AssertResult(res_ids, res_dis);
    }

    {
        std::string file_location = temp_path("/tmp/knowhere");
        write_index(index_, file_location);
        auto new_index = milvus::engine::read_index(file_location);
        EXPECT_EQ(new_index->GetType(), ConvertToCpuIndexType(index_type));
        EXPECT_EQ(new_index->Dimension(), index_->Dimension());
        EXPECT_EQ(new_index->Count(), index_->Count());

        std::vector<int64_t> res_ids(elems);
        std::vector<float> res_dis(elems);
        new_index->Search(nq, xq.data(), res_dis.data(), res_ids.data(), searchconf);
        AssertResult(res_ids, res_dis);
    }

}
