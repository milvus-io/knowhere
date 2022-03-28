/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

namespace faiss {

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/// Squared L2 distance between two vectors
float fvec_L2sqr_sse(
        const float* x,
        const float* y,
        size_t d);

/// inner product
float fvec_inner_product_sse(
        const float* x,
        const float* y,
        size_t d);

/// L1 distance
float fvec_L1_sse(
        const float* x,
        const float* y,
        size_t d);

/// infinity distance
float fvec_Linf_sse(
        const float* x,
        const float* y,
        size_t d);

float fvec_norm_L2sqr_sse(
        const float* x,
        size_t d);

void fvec_L2sqr_ny_sse(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

void fvec_inner_products_ny_sse(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny);

void fvec_madd_sse(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c);

int fvec_madd_and_argmin_sse(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c);

} // namespace faiss
