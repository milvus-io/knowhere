#ifndef COMP_KNOWHERE_CONFIG_H
#define COMP_KNOWHERE_CONFIG_H

#include <string>
#include <vector>

namespace knowhere {

class KnowhereConfig {
 public:
    /**
     * set SIMD type
     */
    enum SimdType {
        AUTO = 0,  // enable all and depend on the system
        AVX512,    // only enable AVX512
        AVX2,      // only enable AVX2
        SSE4_2,    // only enable SSE4_2
        GENERIC,   // use arithmetic instead of SIMD
    };

    static std::string
    SetSimdType(const SimdType simd_type);

    /**
     * Set openblas threshold
     *   if nq < use_blas_threshold, calculated by omp
     *   else, calculated by openblas
     */
    static void
    SetBlasThreshold(const int64_t use_blas_threshold);

    static int64_t
    GetBlasThreshold();

    /**
     * set Clustering early stop [0, 100]
     *   It is to reduce the number of iterations of K-means.
     *   Between each two iterations, if the optimization rate < early_stop_threshold, stop
     *   And if early_stop_threshold = 0, won't early stop
     */
    static void
    SetEarlyStopThreshold(const double early_stop_threshold);

    static double
    GetEarlyStopThreshold();

    /**
     * set Clustering type
     */
    enum ClusteringType {
        K_MEANS = 0,        // k-means (default)
        K_MEANS_PLUS_PLUS,  // k-means++
    };

    static void
    SetClusteringType(const ClusteringType clustering_type);

    /**
     * set Statistics Level [0, 3]
     */
    static void
    SetStatisticsLevel(const int32_t stat_level);

    // todo: add log level?
    /**
     * set Log handler
     */
    static void
    SetLogHandler();
};

}  // namespace knowhere

#endif /* COMP_KNOWHERE_CONFIG_H */
