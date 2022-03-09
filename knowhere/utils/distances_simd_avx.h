
// -*- c++ -*-

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace faiss {

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/// Squared L2 distance between two vectors
float fvec_L2sqr_avx(const float* x, const float* y, size_t d);

/// inner product
float fvec_inner_product_avx(const float* x, const float* y, size_t d);

/// L1 distance
float fvec_L1_avx(const float* x, const float* y, size_t d);

/// infinity distance
float fvec_Linf_avx(const float* x, const float* y, size_t d);

/// binary distance
int xor_popcnt_AVX2_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n);

int or_popcnt_AVX2_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n);

int and_popcnt_AVX2_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n);

/// popcnt
int popcnt_AVX2_lookup(const uint8_t* data, const size_t n);

float jaccard_AVX2(const uint8_t* a, const uint8_t* b, size_t n);

}  // namespace faiss
