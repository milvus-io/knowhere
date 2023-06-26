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

#pragma once

#include <memory>
#include <string>

#include "annoy/src/annoylib.h"
#include "annoy/src/kissrandom.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/ThreadPool.h"
#include "knowhere/index/VecIndex.h"

namespace knowhere {

using ThreadedBuildPolicy = AnnoyIndexSingleThreadedBuildPolicy;

class IndexAnnoy : public VecIndex {
 public:
    IndexAnnoy() {
        index_type_ = IndexEnum::INDEX_ANNOY;
        pool_ = ThreadPool::GetGlobalThreadPool();
    }

    BinarySet
    Serialize(const Config&) override;

    void
    Load(const BinarySet&) override;

    void
    Train(const DatasetPtr&, const Config&) override;

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    GetVectorById(const DatasetPtr&, const Config&) override;

    bool
    HasRawData(const std::string& /*metric_type*/) const override {
        return false;
    }

    DatasetPtr
    Query(const DatasetPtr&, const Config&, const faiss::BitsetView) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    int64_t
    Size() override;

 private:
    bool is_build_ = false;
    std::string metric_type_;
    std::shared_ptr<ThreadPool> pool_;
    std::shared_ptr<AnnoyIndexInterface<int64_t, float>> index_ = nullptr;
};

}  // namespace knowhere
