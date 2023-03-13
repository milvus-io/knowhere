# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

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

if(WITH_PROFILER)
    set(depend_libs
        ${depend_libs}
        tcmalloc_and_profiler
    )
endif()

#==============================================================================
macro(benchmark_test target file)
    set(FILE_SRCS ${file})
    add_executable(${target} ${FILE_SRCS})
    target_link_libraries(${target} ${depend_libs} ${unittest_libs})
    install(TARGETS ${target} DESTINATION unittest)
endmacro()

benchmark_test(benchmark_knowhere_binary        hdf5/benchmark_knowhere_binary.cpp)
benchmark_test(benchmark_knowhere_binary_range  hdf5/benchmark_knowhere_binary_range.cpp)
benchmark_test(benchmark_knowhere_float         hdf5/benchmark_knowhere_float.cpp)
benchmark_test(benchmark_knowhere_float_qps     hdf5/benchmark_knowhere_float_qps.cpp)
benchmark_test(benchmark_knowhere_float_range   hdf5/benchmark_knowhere_float_range.cpp)
