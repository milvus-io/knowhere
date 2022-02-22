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

float fvec_L2sqr_ref(
        const float* x,
        const float* y,
        size_t d);

float fvec_inner_product_ref(
        const float* x,
        const float* y,
        size_t d);

float fvec_L1_ref(
        const float* x,
        const float* y,
        size_t d);

float fvec_Linf_ref(
        const float* x,
        const float* y,
        size_t d);


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

} // namespace faiss
