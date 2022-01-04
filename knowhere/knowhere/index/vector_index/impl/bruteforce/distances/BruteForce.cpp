#include "BruteForce.h"
#include <omp.h>

namespace knowhere {

float fvec_L2sqr_ref(const float *x,
                                 const float *y,
                                 size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float fvec_inner_product_ref(const float *x,
                                         const float *y,
                                         size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * y[i];
    return res;
}

void knn_L2sqr_sse(
        const float *x,
        const float *y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t *res,
        const BitsetView bitset) {
    size_t k = res->k;

    float *value = res->val;
    int64_t *labels = res->ids;

#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        const float *x_i = x + i * d;
        const float *y_j = y;

        float *__restrict val_ = value + i * k;
        int64_t *__restrict ids_ = labels + i * k;

        for (size_t j = 0; j < k; j++) {
            val_[j] = 1.0 / 0.0;
            ids_[j] = -1;
        }

        for (size_t j = 0; j < ny; j++) {
            if (!bitset || !bitset.test(j)) {
                float disij = fvec_L2sqr_ref(x_i, y_j, d);
                if (disij < val_[0]) {
                    maxheap_swap_top(k, val_, ids_, disij, j);
                }
            }
            y_j += d;
        }

        maxheap_reorder(k, val_, ids_);
    }
}

void knn_inner_product_sse(const float * x,
                                   const float * y,
                                   size_t d, size_t nx, size_t ny,
                                   float_minheap_array_t * res,
                                   const BitsetView bitset) {
    size_t k = res->k;
    float * value = res->val;
    int64_t * labels = res->ids;

#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        const float *x_i = x + i * d;
        const float *y_j = y;

        float * __restrict val_ = value  + i * k;
        int64_t * __restrict ids_ = labels  + i * k;

        for (size_t j = 0; j < k; j++) {
            val_[j] = -1.0 / 0.0;
            ids_[j] = -1;
        }

        for (size_t j = 0; j < ny; j++) {
            if (!bitset || !bitset.test(j)) {
                float disij = fvec_inner_product_ref (x_i, y_j, d);
                if (disij > val_[0]) {
                    minheap_swap_top (k, val_, ids_, disij, j);
                }
            }
            y_j += d;
        }

        minheap_reorder (k, val_, ids_);
    }
}

}  // namespace knowhere
