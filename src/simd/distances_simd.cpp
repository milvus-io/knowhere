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

uint8_t lookup8bit[256] = {
    /*  0 */ 0, /*  1 */ 1, /*  2 */ 1, /*  3 */ 2, /*  4 */ 1, /*  5 */ 2, /*  6 */ 2, /*  7 */ 3,
    /*  8 */ 1, /*  9 */ 2, /*  a */ 2, /*  b */ 3, /*  c */ 2, /*  d */ 3, /*  e */ 3, /*  f */ 4,
    /* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3, /* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
    /* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4, /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
    /* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3, /* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
    /* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4, /* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
    /* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4, /* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
    /* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5, /* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
    /* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3, /* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
    /* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4, /* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
    /* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4, /* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
    /* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5, /* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
    /* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4, /* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
    /* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5, /* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
    /* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5, /* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
    /* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6, /* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
    /* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3, /* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
    /* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4, /* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
    /* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4, /* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
    /* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5, /* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
    /* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4, /* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
    /* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5, /* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
    /* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5, /* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
    /* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6, /* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
    /* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4, /* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
    /* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5, /* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
    /* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5, /* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
    /* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6, /* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
    /* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5, /* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
    /* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6, /* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
    /* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6, /* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
    /* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7, /* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8};

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

float
fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fvec_L1_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += fabs(tmp);
    }
    return res;
}

float
fvec_Linf_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        res = fmax(res, fabs(x[i] - y[i]));
    }
    return res;
}

float
fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) res += x[i] * y[i];
    return res;
}

float
fvec_norm_L2sqr_ref(const float* x, size_t d) {
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++) res += x[i] * x[i];
    return res;
}

void
fvec_L2sqr_ny_ref(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_ref(float* ip, const float* x, const float* y, size_t d, size_t ny) {
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
// float fvec_L2sqr(const float* x, const float* y, size_t d) {
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
// float fvec_inner_product(const float* x, const float* y, size_t d) {
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
// float fvec_norm_L2sqr(const float* x, size_t d) {
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
// void fvec_L2sqr_ny(
//         float* dis,
//         const float* x,
//         const float* y,
//         size_t d,
//         size_t ny) {
//     fvec_L2sqr_ny_ref(dis, x, y, d, ny);
// }
//
// float fvec_L1(const float* x, const float* y, size_t d) {
//     return fvec_L1_ref(x, y, d);
// }
//
// float fvec_Linf(const float* x, const float* y, size_t d) {
//     return fvec_Linf_ref(x, y, d);
// }
//
// void fvec_inner_products_ny(
//         float* dis,
//         const float* x,
//         const float* y,
//         size_t d,
//         size_t ny) {
//     fvec_inner_products_ny_ref(dis, x, y, d, ny);
// }
//
//#else
//// scalar implementation
//
// float fvec_L2sqr(const float* x, const float* y, size_t d) {
//    return fvec_L2sqr_ref(x, y, d);
//}
//
// float fvec_L1(const float* x, const float* y, size_t d) {
//    return fvec_L1_ref(x, y, d);
//}
//
// float fvec_Linf(const float* x, const float* y, size_t d) {
//    return fvec_Linf_ref(x, y, d);
//}
//
// float fvec_inner_product(const float* x, const float* y, size_t d) {
//    return fvec_inner_product_ref(x, y, d);
//}
//
// float fvec_norm_L2sqr(const float* x, size_t d) {
//    return fvec_norm_L2sqr_ref(x, d);
//}
//
// void fvec_L2sqr_ny(
//        float* dis,
//        const float* x,
//        const float* y,
//        size_t d,
//        size_t ny) {
//    fvec_L2sqr_ny_ref(dis, x, y, d, ny);
//}
//
// void fvec_inner_products_ny(
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

void
fvec_madd_ref(size_t n, const float* a, float bf, const float* b, float* c) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + bf * b[i];
}

int
fvec_madd_and_argmin_ref(size_t n, const float* a, float bf, const float* b, float* c) {
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

}  // namespace faiss
