/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include <sys/time.h>

#include "knowhere/common/Exception.h"

#include "knowhere/common/Timer.h"
#include "knowhere/index/vector_index/impl/bruteforce/BruteForce.h"
#include "knowhere/index/vector_index/impl/bruteforce/SimpleIndexFlat.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {

class BruteForceTest : public ::testing::Test {
protected:
    void
    SetUp() override {
    }

    void
    TearDown() override {
    }

protected:
};

auto fvecs_read(const char *fname,
                size_t *d_out, size_t *n_out) -> float *
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    int d;
#ifdef __APPLE_
    fread(&d, 1, sizeof(int), f);
#else
    size_t elements_read = fread(&d, 1, sizeof(int), f);
    assert(elements_read > 0 || !"failed to read");
#endif
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    auto *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    }

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
auto ivecs_read(const char *fname, size_t *d_out, size_t *n_out) -> int *
{
    return reinterpret_cast<int*>(fvecs_read(fname, d_out, n_out));
}

auto elapsed () -> double
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

auto CalcRecall(int64_t topk, int64_t k, int nq, knowhere::SimpleIndexFlat::idx_t* gt, knowhere::SimpleIndexFlat::idx_t* nt) -> float {
    float sum_ratio = 0.0f;
    for (int i = 0; i < nq; i++) {
        //std::vector<int64_t> ids_0 = true_ids[i].ids;
        //std::vector<int64_t> ids_1 = result_ids[i].ids;
        std::vector<knowhere::SimpleIndexFlat::idx_t> ids_0(gt + i * k, gt + i * k + topk);
        std::vector<knowhere::SimpleIndexFlat::idx_t> ids_1(nt + i * k, nt + i * k + topk);
        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());
        std::vector<knowhere::SimpleIndexFlat::idx_t> v(nq * 2);
        std::vector<knowhere::SimpleIndexFlat::idx_t>::iterator it;
        it=std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it-v.begin());
        sum_ratio += 1.0f * v.size() / topk;
    }
    return 1.0 * sum_ratio / nq;
}


TEST_F(BruteForceTest, testDemo) {
    size_t d;
    size_t nb;

    size_t loops = 5;

    float *xb = fvecs_read("unittest/siftsmall_base.fvecs", &d, &nb);

    int nlist = 100;
    int m = 8;
    // knowhere::SimpleIndexFlatFlatL2 quantizer(d);
    // knowhere::SimpleIndexFlatIVFPQ index(&quantizer, d, nlist, m, 8);

    knowhere::SimpleIndexFlat index(d, knowhere::METRIC_L2);
    //index.setNumProbes(10);

    index.train(nb, xb);
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);                     // add vectors to the index

    size_t nq;
    float *xq;

    size_t d2;
    xq = fvecs_read("unittest/siftsmall_base.fvecs", &d2, &nq);
    assert(d == d2 || !"query does not have same dimension as train set");
    size_t k; // nb of results per query in the GT
    knowhere::SimpleIndexFlat::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    // load ground-truth and convert int to long
    size_t nq2;

    int *gt_int = ivecs_read("unittest/siftsmall_base.fvecs", &k, &nq2);
    assert(nq2 == nq || !"incorrect nb of ground truth entries");

    printf("d: %ld\n", d);
    printf("nb: %ld\n", nb);
    printf("nq: %ld\n", nq);
    printf("k: %ld\n", k);

    gt = new knowhere::SimpleIndexFlat::idx_t[k * nq];
    for(int i = 0; i < k * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete [] gt_int;

    {
#ifdef __APPLE_
        auto *I = new long long[k * nq];
#else
        auto *I = new knowhere::SimpleIndexFlat::idx_t[k * nq];
#endif
        auto *D = new float[k * nq];
        index.search(nq, xq, k, D, I);

        double avg = 0.0f, dur;
        double min_time = std::numeric_limits<double>::max();


        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            index.search(nq, xq, k, D, I);

            dur = elapsed() - t0;
            avg += dur;
            min_time = std::min(min_time, dur);
        }

        avg /= loops;
        // TODO: Add small sift test data and check for recall here.
        // It is trivial to check recall in brute force research here as everything is flat.
        // If really wanted, we could generate a small sift and corresponding ground truth test data set.
        printf("Recall: %.4f\nTime spent: %.3fs\nMin time: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg, min_time);
        delete [] I;
        delete [] D;
    }

    delete [] xq;
    delete [] gt;
}

}  // namespace
