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
#include <tuple>

#include "knowhere/index/IndexType.h"
#include "knowhere/index/VecIndex.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"

#include "unittest/Helper.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class VecIndexTest : public DataGen,
                     public TestWithParam<std::tuple<knowhere::IndexType, knowhere::MetricType>> {
 protected:
    void
    SetUp() override {
        Init_with_default();
        std::tie(index_type_, metric_type_) = GetParam();
        if (metric_type_ == knowhere::metric::IP) {
            normalize(xb.data(), nb, dim);
            normalize(xq.data(), nq, dim);
        }
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
        knowhere::SetMetaMetricType(conf_, metric_type_);
    }

    void
    TearDown() override {
    }

 protected:
    knowhere::IndexType index_type_;
    knowhere::MetricType metric_type_;
    knowhere::Config conf_;
    knowhere::VecIndexPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    IVFParameters,
    VecIndexTest,
    Values(
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFHNSW, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFHNSW, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_HNSW, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_HNSW, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_ANNOY, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_ANNOY, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWFlat, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWFlat, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWPQ, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWPQ, knowhere::metric::IP),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWSQ, knowhere::metric::L2),
        std::make_tuple(knowhere::IndexEnum::INDEX_RHNSWSQ, knowhere::metric::IP)));

TEST_P(VecIndexTest, basic) {
    assert(!xb.empty());

    // null faiss index
    ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->BuildAll(base_dataset, conf_);
    ASSERT_EQ(index_->index_type(), index_type_);
    ASSERT_EQ(index_->Dim(), dim);
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_GT(index_->Size(), 0);

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, knowhere::IndexMode::MODE_CPU));

    auto result = index_->Query(query_dataset, conf_, nullptr);
    if (index_type_ != knowhere::IndexEnum::INDEX_FAISS_IVFPQ &&
        index_type_ != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 &&
        index_type_ != knowhere::IndexEnum::INDEX_RHNSWPQ &&
        index_type_ != knowhere::IndexEnum::INDEX_RHNSWSQ) {
        AssertAnns(result, nq, k);
    }
    AssertDist(result, metric_type_, nq, k);

    auto result_bs = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result_bs, nq, k, CheckMode::CHECK_NOT_EQUAL);
    AssertDist(result_bs, metric_type_, nq, k);
}
