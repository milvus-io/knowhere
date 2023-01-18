#ifndef DISTANCES_AVX512_H
#define DISTANCES_AVX512_H

#include <stddef.h>
#include <stdint.h>

namespace faiss {

float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_avx512(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_avx512(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_avx512(const float* x, const float* y, size_t d);

}  // namespace faiss

#endif /* DISTANCES_AVX512_H */
