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
#include "knowhere/common/Config.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include <iostream>
#include <random>
#include "knowhere/common/Exception.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        IndexType = GetParam();
        std::cout << "IndexType from GetParam() is: " << IndexType << std::endl;
        Generate(64, 10000, 10);  // dim = 64, nb = 10000, nq = 10
        index_ = std::make_shared<knowhere::IndexHNSW>();
        conf = knowhere::Config{
            {knowhere::meta::DIM, 64},        {knowhere::meta::TOPK, 10},
            {knowhere::IndexParams::M, 16},   {knowhere::IndexParams::efConstruction, 200},
            {knowhere::IndexParams::ef, 200}, {knowhere::Metric::TYPE, knowhere::Metric::L2},
        };
    }

 protected:
    knowhere::Config conf;
    std::shared_ptr<knowhere::IndexHNSW> index_ = nullptr;
    std::string IndexType;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, HNSWTest, Values("HNSW"));

TEST_P(HNSWTest, HNSW_basic) {
    assert(!xb.empty());

    // null faiss index
    /*
    {
        ASSERT_ANY_THROW(index_->Serialize());
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf));
        ASSERT_ANY_THROW(index_->Count());
        ASSERT_ANY_THROW(index_->Dim());
    }
    */

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = base_dataset->Get<int64_t>(knowhere::meta::DIM);
    int64_t rows = base_dataset->Get<int64_t>(knowhere::meta::ROWS);
    auto raw_data = base_dataset->Get<const void*>(knowhere::meta::TENSOR);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);

    // case: k > nb
    const int64_t new_rows = 6;
    base_dataset->Set(knowhere::meta::ROWS, new_rows);
    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    auto result2 = index_->Query(query_dataset, conf, nullptr);
    auto res_ids = result2->Get<int64_t*>(knowhere::meta::IDS);
    for (int64_t i = 0; i < nq; i++) {
        for (int64_t j = new_rows; j < k; j++) {
            ASSERT_EQ(res_ids[i * k + j], -1);
        }
    }
}

TEST_P(HNSWTest, HNSW_delete) {
    assert(!xb.empty());

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = base_dataset->Get<int64_t>(knowhere::meta::DIM);
    int64_t rows = base_dataset->Get<int64_t>(knowhere::meta::ROWS);
    auto raw_data = base_dataset->Get<const void*>(knowhere::meta::TENSOR);
    knowhere::BinaryPtr bptr = std::make_shared<knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf, *bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);

    /*
     * delete result checked by eyes
    auto ids1 = result1->Get<int64_t*>(knowhere::meta::IDS);
    auto ids2 = result2->Get<int64_t*>(knowhere::meta::IDS);
    std::cout << std::endl;
    for (int i = 0; i < nq; ++ i) {
        std::cout << "ids1: ";
        for (int j = 0; j < k; ++ j) {
            std::cout << *(ids1 + i * k + j) << " ";
        }
        std::cout << "ids2: ";
        for (int j = 0; j < k; ++ j) {
            std::cout << *(ids2 + i * k + j) << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < std::min(5, k>>1); ++ j) {
            ASSERT_EQ(*(ids1 + i * k + j + 1), *(ids2 + i * k + j));
        }
    }
    */
}

/*
TEST_P(HNSWTest, HNSW_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    {
        index_->Train(base_dataset, conf);
        index_->AddWithoutIds(base_dataset, conf);
        auto binaryset = index_->Serialize();
        auto bin = binaryset.GetByName("HNSW");

        std::string filename = temp_path("/tmp/HNSW_test_serialize.bin");
        auto load_data = new uint8_t[bin->size];
        serialize(filename, bin, load_data);

        binaryset.clear();
        std::shared_ptr<uint8_t[]> data(load_data);
        binaryset.Append("HNSW", data, bin->size);

        index_->Load(binaryset);
        EXPECT_EQ(index_->Count(), nb);
        EXPECT_EQ(index_->Dim(), dim);
        auto result = index_->Query(query_dataset, conf);
        AssertAnns(result, nq, conf[knowhere::meta::TOPK]);
    }
}*/

/*
 * faiss style test
 * keep it
int
main() {
    int64_t d = 64;      // dimension
    int64_t nb = 10000;  // database size
    int64_t nq = 10;     // 10000;                        // nb of queries
    faiss::ConcurrentBitsetPtr bitset = std::make_shared<faiss::ConcurrentBitset>(nb);

    int64_t* ids = new int64_t[nb];
    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    //    int64_t *ids = (int64_t*)malloc(nb * sizeof(int64_t));
    //    float* xb = (float*)malloc(d * nb * sizeof(float));
    //    float* xq = (float*)malloc(d * nq * sizeof(float));

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
        ids[i] = i;
    }
//    printf("gen xb and ids done! \n");

    //    srand((unsigned)time(nullptr));
    auto random_seed = (unsigned)time(nullptr);
//    printf("delete ids: \n");
    for (int i = 0; i < nq; i++) {
        auto tmp = rand_r(&random_seed) % nb;
//        printf("%ld\n", tmp);
        //        std::cout << "before delete, test result: " << bitset->test(tmp) << std::endl;
        bitset->set(tmp);
        //        std::cout << "after delete, test result: " << bitset->test(tmp) << std::endl;
        for (int j = 0; j < d; j++) xq[d * i + j] = xb[d * tmp + j];
        //        xq[d * i] += i / 1000.;
    }
//    printf("\n");

    int k = 4;
    int m = 16;
    int ef = 200;
    knowhere::IndexHNSW_NM index;
    knowhere::DatasetPtr base_dataset = generate_dataset(nb, d, (const void*)xb, ids);
//    base_dataset->Set(knowhere::meta::ROWS, nb);
//    base_dataset->Set(knowhere::meta::DIM, d);
//    base_dataset->Set(knowhere::meta::TENSOR, (const void*)xb);
//    base_dataset->Set(knowhere::meta::IDS, (const int64_t*)ids);

    knowhere::Config base_conf{
        {knowhere::meta::DIM, d},
        {knowhere::meta::TOPK, k},
        {knowhere::IndexParams::M, m},
        {knowhere::IndexParams::efConstruction, ef},
        {knowhere::Metric::TYPE, knowhere::Metric::L2},
    };
    knowhere::DatasetPtr query_dataset = generate_query_dataset(nq, d, (const void*)xq);
    knowhere::Config query_conf{
        {knowhere::meta::DIM, d},
        {knowhere::meta::TOPK, k},
        {knowhere::IndexParams::M, m},
        {knowhere::IndexParams::ef, ef},
        {knowhere::Metric::TYPE, knowhere::Metric::L2},
    };

    index.Train(base_dataset, base_conf);
    index.Add(base_dataset, base_conf);

//    printf("------------sanity check----------------\n");
    {  // sanity check
        auto res = index.Query(query_dataset, query_conf);
//        printf("Query done!\n");
        const int64_t* I = res->Get<int64_t*>(knowhere::meta::IDS);
//        float* D = res->Get<float*>(knowhere::meta::DISTANCE);

//        printf("I=\n");
//        for (int i = 0; i < 5; i++) {
//            for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
//            printf("\n");
//        }

//        printf("D=\n");
//        for (int i = 0; i < 5; i++) {
//            for (int j = 0; j < k; j++) printf("%7g ", D[i * k + j]);
//            printf("\n");
//        }
    }

//    printf("---------------search xq-------------\n");
    {  // search xq
        auto res = index.Query(query_dataset, query_conf);
        const int64_t* I = res->Get<int64_t*>(knowhere::meta::IDS);

        printf("I=\n");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
            printf("\n");
        }
    }

    printf("----------------search xq with delete------------\n");
    {  // search xq with delete
        auto res = index.Query(query_dataset, query_conf, bitset);
        auto I = res->Get<int64_t*>(knowhere::meta::IDS);

        printf("I=\n");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
            printf("\n");
        }
    }

    delete[] xb;
    delete[] xq;
    delete[] ids;

    return 0;
}
*/
