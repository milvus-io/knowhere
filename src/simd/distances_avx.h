#ifndef DISTANCES_AVX_H
#define DISTANCES_AVX_H

#include <cstddef>
#include <cstdint>

namespace faiss {

/// Squared L2 distance between two vectors
float
fvec_L2sqr_avx(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_avx(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_avx(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_avx(const float* x, const float* y, size_t d);

}  // namespace faiss

#endif /* DISTANCES_AVX_H */
