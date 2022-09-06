#pragma once

#include <future>
#include <vector>

#include "knowhere/common/Dataset.h"
#include "knowhere/index/VecIndex.h"

namespace knowhere {

namespace mthreadtest {
std::vector<std::future<DatasetPtr>>&
GetFeatureQueue() {
    static std::vector<std::future<DatasetPtr>> qu;
    return qu;
}

DatasetPtr
Async(VecIndexPtr& idx, DatasetPtr& dataset, Config& cfg, faiss::BitsetView bitset) {
    std::vector<std::future<DatasetPtr>>& qu = GetFeatureQueue();
    qu.emplace_back(std::async(std::launch::async,
                               [&idx, &dataset, &cfg, &bitset]() { return idx->Query(dataset, cfg, bitset); }));
    assert(!qu.empty());
    auto res = qu.front().get();
    qu.erase(qu.begin());
    return res;
}
}  // namespace mthreadtest
}  // namespace knowhere
