/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../indexing/SimpleIndexFlat.h"

#include <sys/stat.h>
#include <sys/time.h>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

const int LOOPS = 5;
const size_t SHORT_NUM_Q = 3;
const char *OUT_FILENAME = "flat_blas.out";

size_t d;                            // dimension
size_t nb;                       // database size
size_t nq;                        // nb of queries
size_t k;
float *xb;
float *xq;
FILE *f_out;

// #define PLZ_OUTPUT
#define PLZ_OUTPUT_SHORT

void write_to_file(int64_t *ids, float *dis) {
    f_out = fopen(OUT_FILENAME, "w");

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            fprintf(f_out, "%ld %lf\n", ids[i * k + j], dis[i * k + j]);
        }
        fprintf(f_out, "\n");
    }

    fclose(f_out);
}

long int getTime(timeval end, timeval start) {
    return 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;
}

float *fvecs_read(const char *fname,
                  size_t *d_out, size_t *n_out) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

void Read() {
    xb = fvecs_read("sift1M/sift_base.fvecs", &d, &nb);
    xq = fvecs_read("sift1M/sift_query.fvecs", &d, &nq);
}

int main() {
    timeval t0, t1;

    Read();
    printf("read end nb %ld d %ld\n", nb, d);

    nq = 100;
    k = 10;
    printf("nq: %ld\n", nq);
    printf("topk: %ld\n", k);

    gettimeofday(&t0, 0);

    knowhere::SimpleIndexFlat flat(d, knowhere::MetricType::METRIC_L2);
    flat.add(nb, xb);

    gettimeofday(&t1, 0);
    printf("create flat cost %ldms\n", getTime(t1, t0));

    // 0 use blas; 100000000 sse
    // faiss::distance_compute_blas_threshold = 0;
    // 0 use reservoir; 100000000 heap
    // faiss::distance_compute_min_k_reservoir = 100000000;

    float *distances = new float[k * nq];
    int64_t *labels = new int64_t[k * nq];

    long min_query_cost = std::numeric_limits<long>::max();
    long avg_query_cost = 0;

    // throw away first time
    flat.search(nq, xq, k, distances, labels);

    for (int i = 0; i < LOOPS; ++i) {
        printf("Loop %d\n", i);
        gettimeofday(&t0, 0);

        flat.search(nq, xq, k, distances, labels);

        gettimeofday(&t1, 0);
        long tq = getTime(t1, t0);
        avg_query_cost += tq;
        min_query_cost = std::min(min_query_cost, tq);
    }

    if (LOOPS) {
        printf("avg query cost %.1fms, %d times\n", avg_query_cost * 1.0f / LOOPS, LOOPS);
        printf("min query cost %ldms\n", min_query_cost);
    }

    write_to_file(labels, distances);

#ifdef PLZ_OUTPUT
    for (int i=0;i<nq;i++){
        for(int j=0;j<k;j++){
            printf("%ld %lf\n", labels[i*k+j], distances[i*k+j]);
        }
        printf("\n");
    }
#elif defined(PLZ_OUTPUT_SHORT)
    printf("first %ld\n", SHORT_NUM_Q);
    for (size_t i = 0; i < std::min((size_t) SHORT_NUM_Q, nq); i++) {
        for (int j = 0; j < k; j++) {
            printf("%ld %lf\n", labels[i * k + j], distances[i * k + j]);
        }
        printf("\n");
    }

    printf("last %ld\n", SHORT_NUM_Q);
    for (size_t i = std::max((size_t) 0, nq - SHORT_NUM_Q); i < nq; i++) {
        for (int j = 0; j < k; j++) {
            printf("%ld %lf\n", labels[i * k + j], distances[i * k + j]);
        }
        printf("\n");
    }
#endif
    delete[] xb;
    delete[] xq;

    delete[] distances;
    delete[] labels;

    return 0;
}
