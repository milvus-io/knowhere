/** BruteForce provides brute-force search as an option that the data is just
 *  copied to the index without further encoding or organization.
 *
 *  Note that this class does not depend on Faiss or other third-party vector
 *  search libraries (But with some dangling faiss quantifier).
 */
#pragma once

#include <cstdio>
#include "knowhere/common/Heap.h"
#include "knowhere/utils/BitsetView.h"
#include "knowhere/utils/distances_simd.h"

namespace knowhere {

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_L2sqr_sse(
        const float *x,
        const float *y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t *res,
        const faiss::BitsetView bitset = nullptr);

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_inner_product_sse(const float *x,
                           const float *y,
                           size_t d, size_t nx, size_t ny,
                           float_minheap_array_t *res,
                           const faiss::BitsetView bitset = nullptr);

}  // namespace knowhere
