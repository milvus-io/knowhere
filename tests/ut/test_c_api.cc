// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/knowhere_c.h"
#include "utils.h"

using namespace knowhere;

TEST_CASE("Test C API", "[all]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 10;
    int64_t dim = 128;
    int64_t seed = 42;

    CKnowhereConfig config{"auto", 10};
    knowhere_init(&config);

    auto train_ds = GenDataSet(nb, dim, seed);
    KV params[] = {{indexparam::NLIST, "16"}};
    CBuildParams build_params = {
        metric::L2, params, 1, nb, dim, train_ds->GetTensor(),
    };
    CIndexCtx index;
    auto res = knowhere_build_index(IndexEnum::INDEX_FAISS_IVFPQ, &build_params, &index);
    REQUIRE(0 == res);

    CBinarySet bs;
    res = knowhere_serialize_index(&index, &bs);
    REQUIRE(0 == res);

    CIndexCtx index2;
    res = knowhere_deserialize_index(IndexEnum::INDEX_FAISS_IVFPQ, &bs, &index2);
    REQUIRE(0 == res);

    KV search_kvs[] = {{meta::METRIC_TYPE, metric::L2}, {meta::TOPK, "16"}};
    CSearchParams search_params = {
        search_kvs, 2, nq, dim, train_ds->GetTensor(), nullptr, 0,
    };
    CSearchResult out;
    res = knowhere_search_index(&index2, &search_params, &out);
    REQUIRE(out.ids[1] > 0);
    REQUIRE(0 == res);

    knowhere_destroy_index(&index);
    knowhere_destroy_index(&index2);
    knowhere_destroy_binary_set(&bs);
    knowhere_destroy_search_result(&out);
}
