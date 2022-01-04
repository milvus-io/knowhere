cmake_minimum_required(VERSION 3.10)

set(common_srcs
        ${INDEX_SOURCE_DIR}/knowhere/knowhere/common/Heap.h
        ${INDEX_SOURCE_DIR}/knowhere/knowhere/common/Heap.cpp
        )

set(utils_srcs
        ${INDEX_SOURCE_DIR}/knowhere/knowhere/utils/distances_simd.h
        ${INDEX_SOURCE_DIR}/knowhere/knowhere/utils/distances_simd.cpp
        ${INDEX_SOURCE_DIR}/knowhere/knowhere/utils/BitsetView.h
        )

set(distances_srcs
        ${utils_srcs}
        distances/BruteForce.h
        distances/BruteForce.cpp
        )

set(indexing_srcs
        indexing/SimpleIndexFlat.h
        indexing/SimpleIndexFlat.cpp
        )

add_executable(
        BruteForce test/test_demo_sift1M.cpp
        ${common_srcs}
        ${distances_srcs}
        ${indexing_srcs}
)
