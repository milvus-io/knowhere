#include "catch2/catch_test_macros.hpp"
#include "common/range_util.h"
#include "knowhere/knowhere.h"
#include "utils.h"

TEST_CASE("Test distance_in_range", "[range search]") {
    const float low_bound = 0.0f;
    const float high_bound = 1.0f;

    bool is_ip = false;
    REQUIRE(knowhere::distance_in_range(0.0, low_bound, high_bound, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(1.0, low_bound, high_bound, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(0.314, low_bound, high_bound, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(-1.0, low_bound, high_bound, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.23, low_bound, high_bound, is_ip) == false);

    is_ip = true;
    REQUIRE(knowhere::distance_in_range(0.0, low_bound, high_bound, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.0, low_bound, high_bound, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(0.314, low_bound, high_bound, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(-1.0, low_bound, high_bound, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.23, low_bound, high_bound, is_ip) == false);
}

namespace {
void
GenRangeSearchResult(std::vector<std::vector<int64_t>>& labels, std::vector<std::vector<float>>& distances,
                     const int64_t nq, const int64_t label_min, const int64_t label_max, const float distance_min,
                     const float distances_max, int seed = 42) {
    std::mt19937 e(seed);
    std::uniform_int_distribution<> uniform_num(0, 10);
    std::uniform_int_distribution<> uniform_label(label_min, label_max);
    std::uniform_real_distribution<> uniform_distance(distance_min, distances_max);

    labels.resize(nq);
    distances.resize(nq);
    for (int64_t i = 0; i < nq; i++) {
        int64_t num = uniform_num(e);
        for (int64_t j = 0; j < num; j++) {
            auto id = uniform_label(e);
            auto dis = uniform_distance(e);
            labels[i].push_back(id);
            distances[i].push_back(dis);
        }
    }
}

size_t
CountValidRangeSearchResult(const std::vector<std::vector<float>>& distances, const float low_bound,
                            const float high_bound, const bool is_ip) {
    int64_t valid = 0;
    for (size_t i = 0; i < distances.size(); i++) {
        for (size_t j = 0; j < distances[i].size(); j++) {
            if (knowhere::distance_in_range(distances[i][j], low_bound, high_bound, is_ip)) {
                valid++;
            }
        }
    }
    return valid;
}
}  // namespace

TEST_CASE("Test GetRangeSearchResult for HNSW/DiskANN", "[range search]") {
    const int64_t nq = 10;
    const int64_t label_min = 0, label_max = 10000;
    const float dist_min = 0.0, dist_max = 100.0;
    std::vector<std::vector<int64_t>> gen_labels;
    std::vector<std::vector<float>> gen_distances;

    GenRangeSearchResult(gen_labels, gen_distances, nq, label_min, label_max, dist_min, dist_max);

    float* distances;
    int64_t* labels;
    size_t* lims;

    auto GetRangeSearchResult = [](std::vector<std::vector<float>>& result_distances,
                                   std::vector<std::vector<int64_t>>& result_labels, const bool is_ip, const int64_t nq,
                                   const float low_bound, const float high_bound, float*& distances, int64_t*& labels,
                                   size_t*& lims) {
        for (auto i = 0; i < nq; i++) {
            knowhere::FilterRangeSearchResultForOneNq(result_distances[i], result_labels[i], is_ip, low_bound,
                                                      high_bound);
        }
        knowhere::GetRangeSearchResult(result_distances, result_labels, is_ip, nq, low_bound, high_bound, distances,
                                       labels, lims);
    };

    std::vector<std::tuple<bool, float, float>> test_sets = {
        std::make_tuple(false, -10.0, -1.0), std::make_tuple(false, -10.0, 0.0),   std::make_tuple(false, -10.0, 50.0),
        std::make_tuple(false, 0.0, 50.0),   std::make_tuple(false, 0.0, 100.0),   std::make_tuple(false, 50.0, 100.0),
        std::make_tuple(false, 50.0, 200.0), std::make_tuple(false, 100.0, 200.0),

        std::make_tuple(true, -10.0, -1.0),  std::make_tuple(true, -10.0, 0.0),    std::make_tuple(true, -10.0, 50.0),
        std::make_tuple(true, 0.0, 50.0),    std::make_tuple(true, 0.0, 100.0),    std::make_tuple(true, 50.0, 100.0),
        std::make_tuple(true, 50.0, 200.0),  std::make_tuple(true, 100.0, 200.0),
    };

    for (auto& item : test_sets) {
        bool is_ip = std::get<0>(item);
        float low_bound = std::get<1>(item);
        float high_bound = std::get<2>(item);
        GetRangeSearchResult(gen_distances, gen_labels, is_ip, nq, low_bound, high_bound, distances, labels, lims);
        REQUIRE(lims[nq] == CountValidRangeSearchResult(gen_distances, low_bound, high_bound, is_ip));
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace {
void
GenRangeSearchResult(faiss::RangeSearchResult& res, const int64_t nq, const int64_t label_min, const int64_t label_max,
                     const float distance_min, const float distances_max, int seed = 42) {
    std::mt19937 e(seed);
    std::uniform_int_distribution<> uniform_num(0, 10);
    std::uniform_int_distribution<> uniform_label(label_min, label_max);
    std::uniform_real_distribution<> uniform_distance(distance_min, distances_max);

    assert(res.nq == (size_t)nq);
    for (int64_t i = 0; i < nq; i++) {
        int64_t num = uniform_num(e);
        res.lims[i + 1] = res.lims[i] + num;
    }

    res.labels = new int64_t[res.lims[nq]];
    res.distances = new float[res.lims[nq]];
    for (int64_t i = 0; i < nq; i++) {
        for (size_t j = res.lims[i]; j < res.lims[i + 1]; j++) {
            auto id = uniform_label(e);
            auto dis = uniform_distance(e);
            res.labels[j] = id;
            res.distances[j] = dis;
        }
    }
}

size_t
CountValidRangeSearchResult(const float* distances, const size_t* lims, const int64_t nq, const float low_bound,
                            const float high_bound, const bool is_ip) {
    int64_t valid = 0;
    for (int64_t i = 0; i < nq; i++) {
        for (size_t j = lims[i]; j < lims[i + 1]; j++) {
            if (knowhere::distance_in_range(distances[j], low_bound, high_bound, is_ip)) {
                valid++;
            }
        }
    }
    return valid;
}
}  // namespace

TEST_CASE("Test GetRangeSearchResult for Faiss", "[range search]") {
    const int64_t nq = 10;
    const int64_t label_min = 0, label_max = 10000;
    const float dist_min = 0.0, dist_max = 100.0;

    faiss::RangeSearchResult res(nq);
    GenRangeSearchResult(res, nq, label_min, label_max, dist_min, dist_max);

    float* distances;
    int64_t* labels;
    size_t* lims;

    std::vector<std::tuple<bool, float, float>> test_sets = {
        std::make_tuple(false, -10.0, -1.0), std::make_tuple(false, -10.0, 0.0),   std::make_tuple(false, -10.0, 50.0),
        std::make_tuple(false, 0.0, 50.0),   std::make_tuple(false, 0.0, 100.0),   std::make_tuple(false, 50.0, 100.0),
        std::make_tuple(false, 50.0, 200.0), std::make_tuple(false, 100.0, 200.0),

        std::make_tuple(true, -10.0, -1.0),  std::make_tuple(true, -10.0, 0.0),    std::make_tuple(true, -10.0, 50.0),
        std::make_tuple(true, 0.0, 50.0),    std::make_tuple(true, 0.0, 100.0),    std::make_tuple(true, 50.0, 100.0),
        std::make_tuple(true, 50.0, 200.0),  std::make_tuple(true, 100.0, 200.0),
    };

    for (auto& item : test_sets) {
        bool is_ip = std::get<0>(item);
        float low_bound = std::get<1>(item);
        float high_bound = std::get<2>(item);
        knowhere::GetRangeSearchResult(res, is_ip, nq, low_bound, high_bound, distances, labels, lims, nullptr);
        REQUIRE(lims[nq] == CountValidRangeSearchResult(res.distances, res.lims, nq, low_bound, high_bound, is_ip));
    }
}
