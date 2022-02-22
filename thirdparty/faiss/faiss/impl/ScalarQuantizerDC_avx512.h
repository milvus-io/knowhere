/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

SQDistanceComputer* sq_get_distance_computer_avx512(
        MetricType metric,
        QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained);

Quantizer* sq_select_quantizer_avx512(
        QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained);

InvertedListScanner* sq_select_inverted_list_scanner_avx512(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        size_t dim,
        bool store_pairs,
        bool by_residual);

} // namespace faiss
