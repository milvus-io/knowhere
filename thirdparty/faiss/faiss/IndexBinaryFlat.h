/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_BINARY_FLAT_H
#define INDEX_BINARY_FLAT_H

#include <vector>

#include <faiss/IndexBinary.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace faiss {

/** Index that stores the full vectors and performs exhaustive search. */
struct IndexBinaryFlat : IndexBinary {
    /// database vectors, size ntotal * d / 8
    std::vector<uint8_t> xb;

    /// external database vectors, size ntotal * d / 8
    uint8_t* xb_ex = nullptr;

    /** Select between using a heap or counting to select the k smallest values
     * when scanning inverted lists.
     */
    bool use_heap = true;

    size_t query_batch_size = 32;

    explicit IndexBinaryFlat(idx_t d);

    IndexBinaryFlat(idx_t d, MetricType metric);

    void add(idx_t n, const uint8_t* x) override;

    void add_ex(idx_t n, const uint8_t* x) override;

    void get_vector_by_id(idx_t n, const idx_t* xids, uint8_t* x) override;

    void reset() override;

    void search(
            idx_t n,
            const uint8_t* x,
            idx_t k,
            int32_t* distances,
            idx_t* labels,
            const BitsetView bitset = nullptr) const override;

    void range_search(
            idx_t n,
            const uint8_t* x,
            float radius,
            RangeSearchResult* result,
            const BitsetView bitset = nullptr) const override;

    void reconstruct(idx_t key, uint8_t* recons) const override;

    /** Remove some ids. Note that because of the indexing structure,
     * the semantics of this operation are different from the usual ones:
     * the new ids are shifted. */
    size_t remove_ids(const IDSelector& sel) override;

    const uint8_t* get_xb() const {
        return xb_ex != nullptr ? (const uint8_t*)xb_ex
                                : (const uint8_t*)xb.data();
    }

    IndexBinaryFlat() {}
};

} // namespace faiss

#endif // INDEX_BINARY_FLAT_H
