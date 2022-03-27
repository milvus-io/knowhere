/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <cmath>
#include <cstring>

//#ifdef __aarch64__
//#include <arm_neon.h>
//#endif

#include "distances_simd.h"

namespace faiss {

/*********************************************************
 * Optimized distance computations
 *********************************************************/

/* Functions to compute:
   - L2 distance between 2 vectors
   - inner product between 2 vectors
   - L2 norm of a vector

   The functions should probably not be invoked when a large number of
   vectors are be processed in batch (in which case Matrix multiply
   is faster), but may be useful for comparing vectors isolated in
   memory.

   Works with any vectors of any dimension, even unaligned (in which
   case they are slower).

*/

/*********************************************************
 * Reference implementations
 */

float fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float fvec_L1_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += fabs(tmp);
    }
    return res;
}

float fvec_Linf_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        res = fmax(res, fabs(x[i] - y[i]));
    }
    return res;
}

float fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * y[i];
    return res;
}

float fvec_norm_L2sqr_ref(const float* x, size_t d) {
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++)
        res += x[i] * x[i];
    return res;
}

void fvec_L2sqr_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref(x, y, d);
        y += d;
    }
}

void fvec_inner_products_ny_ref(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    // BLAS slower for the use cases here
#if 0
    {
        FINTEGER di = d;
        FINTEGER nyi = ny;
        float one = 1.0, zero = 0.0;
        FINTEGER onei = 1;
        sgemv_ ("T", &di, &nyi, &one, y, &di, x, &onei, &zero, ip, &onei);
    }
#endif
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_ref(x, y, d);
        y += d;
    }
}

//#if defined(__aarch64__)
//
//float fvec_L2sqr(const float* x, const float* y, size_t d) {
//    float32x4_t accux4 = vdupq_n_f32(0);
//    const size_t d_simd = d - (d & 3);
//    size_t i;
//    for (i = 0; i < d_simd; i += 4) {
//        float32x4_t xi = vld1q_f32(x + i);
//        float32x4_t yi = vld1q_f32(y + i);
//        float32x4_t sq = vsubq_f32(xi, yi);
//        accux4 = vfmaq_f32(accux4, sq, sq);
//    }
//    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
//    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
//    for (; i < d; ++i) {
//        float32_t xi = x[i];
//        float32_t yi = y[i];
//        float32_t sq = xi - yi;
//        accux1 += sq * sq;
//    }
//    return accux1;
//}
//
//float fvec_inner_product(const float* x, const float* y, size_t d) {
//    float32x4_t accux4 = vdupq_n_f32(0);
//    const size_t d_simd = d - (d & 3);
//    size_t i;
//    for (i = 0; i < d_simd; i += 4) {
//        float32x4_t xi = vld1q_f32(x + i);
//        float32x4_t yi = vld1q_f32(y + i);
//        accux4 = vfmaq_f32(accux4, xi, yi);
//    }
//    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
//    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
//    for (; i < d; ++i) {
//        float32_t xi = x[i];
//        float32_t yi = y[i];
//        accux1 += xi * yi;
//    }
//    return accux1;
//}
//
//float fvec_norm_L2sqr(const float* x, size_t d) {
//    float32x4_t accux4 = vdupq_n_f32(0);
//    const size_t d_simd = d - (d & 3);
//    size_t i;
//    for (i = 0; i < d_simd; i += 4) {
//        float32x4_t xi = vld1q_f32(x + i);
//        accux4 = vfmaq_f32(accux4, xi, xi);
//    }
//    float32x4_t accux2 = vpaddq_f32(accux4, accux4);
//    float32_t accux1 = vdups_laneq_f32(accux2, 0) + vdups_laneq_f32(accux2, 1);
//    for (; i < d; ++i) {
//        float32_t xi = x[i];
//        accux1 += xi * xi;
//    }
//    return accux1;
//}
//
//// not optimized for ARM
//void fvec_L2sqr_ny(
//        float* dis,
//        const float* x,
//        const float* y,
//        size_t d,
//        size_t ny) {
//    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
//}
//
//float fvec_L1(const float* x, const float* y, size_t d) {
//    return fvec_L1_ref(x, y, d);
//}
//
//float fvec_Linf(const float* x, const float* y, size_t d) {
//    return fvec_Linf_ref(x, y, d);
//}
//
//void fvec_inner_products_ny(
//        float* dis,
//        const float* x,
//        const float* y,
//        size_t d,
//        size_t ny) {
//    fvec_inner_products_ny_ref(dis, x, y, d, ny);
//}
//
//#else
//// scalar implementation
//
//float fvec_L2sqr(const float* x, const float* y, size_t d) {
//    return fvec_L2sqr_ref(x, y, d);
//}
//
//float fvec_L1(const float* x, const float* y, size_t d) {
//    return fvec_L1_ref(x, y, d);
//}
//
//float fvec_Linf(const float* x, const float* y, size_t d) {
//    return fvec_Linf_ref(x, y, d);
//}
//
//float fvec_inner_product(const float* x, const float* y, size_t d) {
//    return fvec_inner_product_ref(x, y, d);
//}
//
//float fvec_norm_L2sqr(const float* x, size_t d) {
//    return fvec_norm_L2sqr_ref(x, d);
//}
//
//void fvec_L2sqr_ny(
//        float* dis,
//        const float* x,
//        const float* y,
//        size_t d,
//        size_t ny) {
//    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
//}
//
//void fvec_inner_products_ny(
//        float* dis,
//        const float* x,
//        const float* y,
//        size_t d,
//        size_t ny) {
//    fvec_inner_products_ny_ref(dis, x, y, d, ny);
//}
//
//#endif

/***************************************************************************
 * heavily optimized table computations
 ***************************************************************************/

void fvec_madd_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
}

int fvec_madd_and_argmin_ref(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

} // namespace faiss
