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

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <knowhere/bitsetview.h>
using knowhere::BitsetView;
namespace faiss {

/*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/

/** Compute pairwise distances between sets of vectors
 *
 * @param d     dimension of the vectors
 * @param nq    nb of query vectors
 * @param nb    nb of database vectors
 * @param xq    query vectors (size nq * d)
 * @param xb    database vectors (size nb * d)
 * @param dis   output distances (size nq * nb)
 * @param ldq,ldb, ldd strides for the matrices
 */
void pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq = -1,
        int64_t ldb = -1,
        int64_t ldd = -1);

/** compute the L2 norms for a set of vectors
 *
 * @param  norms    output norms, size nx
 * @param  x        set of vectors, size nx * d
 */
void fvec_norms_L2(float* norms, const float* x, size_t d, size_t nx);

/// same as fvec_norms_L2, but computes squared norms
void fvec_norms_L2sqr(float* norms, const float* x, size_t d, size_t nx);

/* L2-renormalize a set of vector. Nothing done if the vector is 0-normed */
void fvec_renorm_L2(size_t d, size_t nx, float* x);

/* This function exists because the Torch counterpart is extremely slow
   (not multi-threaded + unexpected overhead even in single thread).
   It is here to implement the usual property |x-y|^2=|x|^2+|y|^2-2<x|y>  */
void inner_product_to_L2sqr(
        float* dis,
        const float* nr1,
        const float* nr2,
        size_t n1,
        size_t n2);

/*********************************************************
 * Vector to vector functions
 *********************************************************/

/** compute c := a + b for vectors
 *
 * c and a can overlap, c and b can overlap
 *
 * @param a size d
 * @param b size d
 * @param c size d
 */
void fvec_add(size_t d, const float* a, const float* b, float* c);

/** compute c := a + b for a, c vectors and b a scalar
 *
 * c and a can overlap
 *
 * @param a size d
 * @param c size d
 */
void fvec_add(size_t d, const float* a, float b, float* c);

/** compute c := a - b for vectors
 *
 * c and a can overlap, c and b can overlap
 *
 * @param a size d
 * @param b size d
 * @param c size d
 */
void fvec_sub(size_t d, const float* a, const float* b, float* c);

/***************************************************************************
 * Compute a subset of  distances
 ***************************************************************************/

/* compute the inner product between x and a subset y of ny vectors,
  whose indices are given by idy.  */
void fvec_inner_products_by_idx(
        float* ip,
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny);

/* same but for a subset in y indexed by idsy (ny vectors in total) */
void fvec_L2sqr_by_idx(
        float* dis,
        const float* x,
        const float* y,
        const int64_t* ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny);

/** compute dis[j] = L2sqr(x[ix[j]], y[iy[j]]) forall j=0..n-1
 *
 * @param x  size (max(ix) + 1, d)
 * @param y  size (max(iy) + 1, d)
 * @param ix size n
 * @param iy size n
 * @param dis size n
 */
void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis);

/* same for inner product */
void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis);

/***************************************************************************
 * KNN functions
 ***************************************************************************/

// threshold on nx above which we switch to BLAS to compute distances
FAISS_API extern int distance_compute_blas_threshold;

// threshold on nx above which we switch to compute parallel on ny
FAISS_API extern int parallel_policy_threshold;

// block sizes for BLAS distance computations
FAISS_API extern int distance_compute_blas_query_bs;
FAISS_API extern int distance_compute_blas_database_bs;

// above this number of results we switch to a reservoir to collect results
// rather than a heap
FAISS_API extern int distance_compute_min_k_reservoir;

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result array, which also provides k. Sorted on output
 */
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res,
        const BitsetView bitset = nullptr);

/** Same as knn_inner_product, for the L2 distance
 *  @param y_norm2    norms for the y vectors (nullptr or size ny)
 */
void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const float* y_norm2 = nullptr,
        const BitsetView bitset = nullptr);

void knn_cosine(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* ha,
        const BitsetView bitset);

void knn_jaccard(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const BitsetView bitset = nullptr);

/* Find the nearest neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res);

void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res);

/***************************************************************************
 * Range search
 ***************************************************************************/

/// Forward declaration, see AuxIndexStructures.h
struct RangeSearchResult;

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
        RangeSearchResult* result,
        const BitsetView bitset = nullptr);

/// same as range_search_L2sqr for the inner product similarity
void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const BitsetView bitset = nullptr);

void range_search_cosine(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const BitsetView bitset = nullptr);

/***************************************************************************
 * PQ tables computations
 ***************************************************************************/

/// specialized function for PQ2
void compute_PQ_dis_tables_dsub2(
        size_t d,
        size_t ksub,
        const float* centroids,
        size_t nx,
        const float* x,
        bool is_inner_product,
        float* dis_tables);

/***************************************************************************
 * elkan
 ***************************************************************************/

/** Return the nearest neighbors of each of the nx vectors x among the ny
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param ids  result array ids
 * @param val  result array value
 */
void elkan_L2_sse(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        int64_t* ids,
        float* val);

/***************************************************************************
 * Templatized versions of distance functions
 ***************************************************************************/

} // namespace faiss
