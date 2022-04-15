/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#include <faiss/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

/***************************************************************************
 * Range search
 ***************************************************************************/

/** Find the nearest neighbors for nx queries in a set of ny vectors
 * compute_l2 = compute pairwise squared L2 distance rather than inner prod
 */
template <bool compute_l2>
static void range_search_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*>& res,
        size_t buffer_size,
        const BitsetView bitset) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    float *ip_block = new float[bs_x * bs_y];
    ScopeDeleter<float> del0(ip_block);

    float* x_norms = nullptr;
    float* y_norms = nullptr;
    ScopeDeleter<float> del1, del2;
    if (compute_l2) {
        x_norms = new float[nx];
        del1.set(x_norms);
        fvec_norms_L2sqr(x_norms, x, d, nx);

        y_norms = new float[ny];
        del2.set(y_norms);
        fvec_norms_L2sqr(y_norms, y, d, ny);
    }

    for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
        size_t j1 = j0 + bs_y;
        if (j1 > ny)
            j1 = ny;
        RangeSearchResult* tmp_res = new RangeSearchResult(nx);
        tmp_res->buffer_size = buffer_size;
        RangeSearchPartialResult* pres = new RangeSearchPartialResult(tmp_res);
        res.push_back(pres);

        for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
            size_t i1 = i0 + bs_x;
            if (i1 > nx)
                i1 = nx;

            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_(
                        "Transpose",
                        "Not transpose",
                        &nyi,
                        &nxi,
                        &di,
                        &one,
                        y + j0 * d,
                        &di,
                        x + i0 * d,
                        &di,
                        &zero,
                        ip_block,
                        &nyi);
            }

            for (size_t i = i0; i < i1; i++) {
                const float *ip_line = ip_block + (i - i0) * (j1 - j0);

                RangeQueryResult& qres = pres->new_result(i);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line++;
                    if (bitset.empty() || !bitset.test((int64_t)j)) {
                        if (compute_l2) {
                            float dis =  x_norms[i] + y_norms[j] - 2 * ip;
                            if (dis < radius) {
                                qres.add(dis, j);
                            }
                        } else {
                            if (ip > radius) {
                                qres.add(ip, j);
                            }
                        }
                    }
                }
            }
        }
        InterruptCallback::check();
    }

    // RangeSearchPartialResult::merge(partial_results);
}

template <bool compute_l2>
static void range_search_sse(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*>& res,
        size_t buffer_size,
        const BitsetView bitset) {
#pragma omp parallel
    {
        RangeSearchResult* tmp_res = new RangeSearchResult(nx);
        tmp_res->buffer_size = buffer_size;
        auto pres = new RangeSearchPartialResult(tmp_res);

#pragma omp for
        for (size_t i = 0; i < nx; i++) {
            const float * x_ = x + i * d;
            RangeQueryResult& qres = pres->new_result(i);

            for (size_t j = 0; j < ny; j++) {
                const float * y_ = y + j * d;
                if (bitset.empty() || !bitset.test((int64_t)j)) {
                    if (compute_l2) {
                        float disij = fvec_L2sqr(x_, y_, d);
                        if (disij < radius) {
                            qres.add(disij, j);
                        }
                    } else {
                        float ip = fvec_inner_product(x_, y_, d);
                        if (ip > radius) {
                            qres.add(ip, j);
                        }
                    }
                }
            }
        }
        if (!pres->buffers.empty()) {
#pragma omp critical
            res.push_back(pres);
        }
    }

    // check just at the end because the use case is typically just
    // when the nb of queries is low.
    InterruptCallback::check();
}

void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*>& res,
        size_t buffer_size,
        const BitsetView bitset) {
    if (nx < distance_compute_blas_threshold) {
        range_search_sse<true>(
                x, y, d, nx, ny, radius, res, buffer_size, bitset);
    } else {
        range_search_blas<true>(
                x, y, d, nx, ny, radius, res, buffer_size, bitset);
    }
}

void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        std::vector<RangeSearchPartialResult*>& res,
        size_t buffer_size,
        const BitsetView bitset) {
    if (nx < distance_compute_blas_threshold) {
        range_search_sse<false>(
                x, y, d, nx, ny, radius, res, buffer_size, bitset);
    } else {
        range_search_blas<false>(
                x, y, d, nx, ny, radius, res, buffer_size, bitset);
    }
}

template <class ResultHandler>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const BitsetView bitset) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;

            resi.begin(i);

            for (size_t j = 0; j < ny; j++) {
                if (bitset.empty() || !bitset.test(j)) {
                    float ip = fvec_inner_product(x_i, y_j, d);
                    resi.add_result(ip, j);
                }
                y_j += d;
            }
            resi.end();
        }
    }
}

template <class ResultHandler>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const BitsetView bitset) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                if (bitset.empty() || !bitset.test(j)) {
                    float disij = fvec_L2sqr(x_i, y_j, d);
                    resi.add_result(disij, j);
                }
                y_j += d;
            }
            resi.end();
        }
    }
}

void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const BitsetView bitset) {
    RangeSearchResultHandler<CMax<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_L2sqr_seq(x, y, d, nx, ny, resh, bitset);
    } else {
        // exhaustive_L2sqr_blas(x, y, d, nx, ny, resh, nullptr, bitset);
    }
}

void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const BitsetView bitset) {
    RangeSearchResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_inner_product_seq(x, y, d, nx, ny, resh, bitset);
    } else {
        // exhaustive_inner_product_blas(x, y, d, nx, ny, resh, nullptr, bitset);
    }
}

} // namespace faiss
