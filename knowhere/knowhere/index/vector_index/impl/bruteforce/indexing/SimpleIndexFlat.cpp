#include "SimpleIndexFlat.h"
#include "knowhere/common/Heap.h"
#include "knowhere/index/vector_index/impl/bruteforce/distances/BruteForce.h"

namespace knowhere {

SimpleIndexFlat::SimpleIndexFlat (idx_t d, MetricType metric) {
    this->d = d;
    this->metric_type = metric;
}

void SimpleIndexFlat::add (idx_t n, const float *x) {
    xb.insert(xb.end(), x, x + n * d);
    ntotal += n;
}

void SimpleIndexFlat::search (idx_t n, const float *x, idx_t k,
                        float *distances, idx_t *labels,
                        const faiss::BitsetView bitset) const
{
    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
        knn_inner_product_sse (x, xb.data(), d, n, ntotal, &res, bitset);
    } else {
        // metric_type == METRIC_L2
        float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances};
        knn_L2sqr_sse (x, xb.data(), d, n, ntotal, &res, bitset);
    }
}

void SimpleIndexFlat::train(idx_t n, const float* x) {
    // Do nothing.
}


} // namespace knowhere
