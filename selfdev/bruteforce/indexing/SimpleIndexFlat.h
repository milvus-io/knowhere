#pragma once
/**
 *  Simplified version of Faiss's IndexFlat struct with minimal functions.
 *
 *  This library does *not* dependent on Faiss.
 */
#include <cstdio>
#include <typeinfo>
#include <string>
#include <vector>
#include "../include/BitsetView.h"

namespace knowhere {

// MetricType is migrated from Faiss metric type.
enum MetricType {
    METRIC_INNER_PRODUCT = 0,  ///< maximum inner product search
    METRIC_L2 = 1,             ///< squared L2 search
    METRIC_L1,                 ///< L1 (aka cityblock)
    METRIC_Linf,               ///< infinity distance
    METRIC_Lp,                 ///< L_p distance, p is given by a faiss::Index
    /// metric_arg
    METRIC_Jaccard,
    METRIC_Tanimoto,
    METRIC_Hamming,
    METRIC_Substructure,       ///< Tversky case alpha = 0, beta = 1
    METRIC_Superstructure,     ///< Tversky case alpha = 1, beta = 0

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
};

/** Index that stores the full vectors and performs exhaustive search */
struct SimpleIndexFlat {
    using idx_t = int64_t;  ///< all indices are this type
    using component_t = float;
    using distance_t = float;

    /// Database vectors of size ntotal * d.
    std::vector<float> xb;

    int d;                 ///< vector dimension
    idx_t ntotal;          ///< total nb of indexed vectors
    bool verbose;          ///< verbosity level

    /// set if the Index does not require training, or if training is
    /// done already
    bool is_trained;

    /// type of metric this index uses for search
    MetricType metric_type;
    float metric_arg;     ///< argument of the metric type

    explicit SimpleIndexFlat(idx_t d, MetricType metric = METRIC_L2);

    void add(idx_t n, const float *x);

    void search(
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels,
            const BitsetView bitset = nullptr) const;

    void train(idx_t n, const float* x);

};

}  // namespace knowhere
