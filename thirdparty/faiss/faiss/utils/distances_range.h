/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once

#include <stdint.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <knowhere/utils/BitsetView.h>

namespace faiss {

/***************************************************************************
 * Range search
 ***************************************************************************/

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x      query vectors, size nx * d
 * @param y      database vectors, size ny * d
 * @param radius search radius around the x vectors
 * @param result result structure
 */
void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*> &result,
        size_t buffer_size,
        const BitsetView bitset = nullptr);

/// same as range_search_L2sqr for the inner product similarity
void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*>& result,
        size_t buffer_size,
        const BitsetView bitset = nullptr);

} // namespace faiss
