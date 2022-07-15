# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty)
include_directories(${KNOWHERE_SOURCE_DIR}/include)

include_directories(/usr/local/hdf5/include)
link_directories(/usr/local/hdf5/lib)

set(unittest_libs
        gtest gmock gtest_main gmock_main)

set(depend_libs
        knowhere
        hdf5
        ${OpenBLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        )

if ( LINUX AND ENABLE_PROFILING )
    set( depend_libs
        ${depend_libs}
        gperftools
        )
endif ()

set( UTIL_SRCS
        ${KNOWHERE_SOURCE_DIR}/unittest/utils.cpp
        )

#==============================================================================
macro(benchmark_test target file)
    set(FILE_SRCS ${UTIL_SRCS} ${file})
    add_executable(${target} ${FILE_SRCS})
    target_link_libraries(${target} ${depend_libs} ${unittest_libs})
    install(TARGETS ${target} DESTINATION unittest)
endmacro()

benchmark_test(benchmark_faiss                  benchmark_faiss.cpp)
benchmark_test(benchmark_knowhere_binary        benchmark_knowhere_binary.cpp)
benchmark_test(benchmark_knowhere_binary_range  benchmark_knowhere_binary_range.cpp)
benchmark_test(benchmark_knowhere_float         benchmark_knowhere_float.cpp)
benchmark_test(benchmark_knowhere_float_range   benchmark_knowhere_float_range.cpp)
benchmark_test(benchmark_knowhere_perf          benchmark_knowhere_perf.cpp)
