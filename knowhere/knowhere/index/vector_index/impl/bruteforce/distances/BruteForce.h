/** BruteForce provides brute-force search as an option that the data is just
 *  copied to the index without further encoding or organization.
 *
 *  Note that this class does not depend on Faiss or other third-party vector
 *  search libraries (But with some dangling faiss quantifier).
 */
#pragma once

#include <cstdio>
#include "../include//Heap.h"
#include "../include/BitsetView.h"

namespace knowhere {

/** Copied from Faiss. **/
float fvec_L2sqr_ref(const float *x,
                     const float *y,
                     size_t d);

/** Copied from Faiss. **/
float fvec_inner_product_ref(const float *x,
                             const float *y,
                             size_t d);

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_L2sqr_sse(
        const float *x,
        const float *y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t *res,
        const BitsetView bitset = nullptr);

/** Partly copied from <knn_L2sqr_sse> in knowhere project. **/
void knn_inner_product_sse(const float *x,
                           const float *y,
                           size_t d, size_t nx, size_t ny,
                           float_minheap_array_t *res,
                           const BitsetView bitset = nullptr);

}  // namespace knowhere
