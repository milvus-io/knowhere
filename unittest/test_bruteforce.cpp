/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
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
 public:
    static void testWithMetric(knowhere::MetricType metric, std::vector<knowhere::SimpleIndexFlat::idx_t>& d_vec,
                                std::vector<float>& i_vec) {
        size_t d = 3;
        size_t nb = 5;

        // Just some random values.
        auto *xbb = new float[d*nb] { 3.0, 5.0, 6.0, 4.0, 7.0, 5.0, 2.0, 8.0, 9.0, 3.0, 5.0, 6.0, 4.0, 7.0, 5.0};
        knowhere::SimpleIndexFlat index_L2(d, metric);
        index_L2.train(nb, xbb);
        index_L2.add(nb, xbb);
        size_t nq = 5;
        auto *xq = new float[d*nb] { 5.0, 2.0, 8.0, 3.0, 5.0, 6.0, 6.0, 4.0, 7.0, 5.0, 2.0, 8.0, 4.0, 7.0, 5.0};
        size_t k = 3;
        auto *I = new knowhere::SimpleIndexFlat::idx_t[k * nq];
        auto *D = new float[k * nq];
        index_L2.search(nq, xq, k, D, I);

        // Copy as vector and verify result in test units.
        d_vec = std::vector<knowhere::SimpleIndexFlat::idx_t>(D, D+15);
        i_vec = std::vector<float>(I, I+15);

        delete [] I;
        delete [] D;
        delete [] xq;
        delete [] xbb;
    }
 protected:
    void SetUp() override {}
    void TearDown() override {}
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
    size_t elements_read = fread(&d, 1, sizeof(int), f);
    assert(elements_read > 0 || !"failed to read");
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

TEST_F(BruteForceTest, simpleTestWithInnerProduct) {
    std::vector<knowhere::SimpleIndexFlat::idx_t> d_vec;
    std::vector<float> i_vec;
    BruteForceTest::testWithMetric(knowhere::METRIC_INNER_PRODUCT, d_vec, i_vec);
    EXPECT_THAT(d_vec, ::testing::ElementsAre(98, 74, 74, 100, 77, 77, 107, 87, 87, 98, 74, 74, 109, 90, 90));
    EXPECT_THAT(i_vec, ::testing::ElementsAre(2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1));
}

TEST_F(BruteForceTest, simpleTestWithL2) {
    std::vector<knowhere::SimpleIndexFlat::idx_t> d_vec;
    std::vector<float> i_vec;
    BruteForceTest::testWithMetric(knowhere::METRIC_L2, d_vec, i_vec);
    EXPECT_THAT(d_vec, ::testing::ElementsAre(17, 17, 35, 0, 0, 6, 11, 11, 17, 17, 17, 35, 0, 0, 6));
    EXPECT_THAT(i_vec, ::testing::ElementsAre(3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 1, 4, 3));
}

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/
// Commented out as this unit takes too long.
#if 0
TEST_F(BruteForceTest, testWithDemo) {
    size_t d;
    size_t nb;

    size_t loops = 5;
    float *xb = fvecs_read("sift_base.fvecs", &d, &nb);

    int nlist = 100;
    int m = 8;

    knowhere::SimpleIndexFlat index(d, knowhere::METRIC_L2);
    //index.setNumProbes(10);

    index.train(nb, xb);
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);                     // add vectors to the index
    std::cout << "ntotal = " << index.ntotal << std::endl;

    size_t nq;
    float *xq;

    size_t d2;
    xq = fvecs_read("sift_query.fvecs", &d2, &nq);
    assert(d == d2 || !"query does not have same dimension as train set");
    size_t k; // nb of results per query in the GT
    knowhere::SimpleIndexFlat::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    // load ground-truth and convert int to long
    size_t nq2;

    int *gt_int = ivecs_read("sift_groundtruth.ivecs", &k, &nq2);
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
        auto *I = new knowhere::SimpleIndexFlat::idx_t[k * nq];
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
        printf("Recall: %.4f\nTime spent: %.3fs\nMin time: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg, min_time);
        delete [] I;
        delete [] D;
    }

    delete [] xq;
    delete [] gt;
}
#endif

}  // namespace
