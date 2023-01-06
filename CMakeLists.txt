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

cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/utils/utils.cmake)

knowhere_option(USE_CUDA "Build with CUDA" OFF)
knowhere_option(WITH_UT "Build with UT test" OFF)
knowhere_option(WITH_ASAN "Build with ASAN" OFF)
knowhere_option(WITH_DISKANN "Build with diskann index" OFF)
knowhere_option(WITH_BENCHMARK "Build with benchmark" OFF)
knowhere_option(WITH_COVERAGE "Build with coverage" OFF)

if(KNOWHERE_VERSION)
    message(STATUS "Building KNOWHERE version: ${KNOWHERE_VERSION}")
    add_definitions(-DKNOWHERE_VERSION=${KNOWHERE_VERSION})
endif ()

if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75;70;61;60)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP)
add_definitions(-DELPP_THREAD_SAFE)
add_definitions(-DNOT_COMPILE_FOR_SWIG)

include(cmake/utils/compile_flags.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)
include(cmake/libs/libannoy.cmake)
include(cmake/libs/libhnsw.cmake)
include(cmake/libs/libeasylogging.cmake)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(thirdparty/faiss)
include_directories(thirdparty/bitset)
include_directories(thirdparty)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(WITH_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

knowhere_file_glob(GLOB_RECURSE KNOWHERE_SRCS src/common/*.cc
    src/index/*.cc src/io/*.cc)

if(NOT WITH_DISKANN)
    knowhere_file_glob(GLOB_RECURSE KNOWHERE_DISKANN_SRCS src/index/diskann/*.cc)
    list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_DISKANN_SRCS})
endif()

if(NOT USE_CUDA)
    knowhere_file_glob(GLOB_RECURSE KNOWHERE_GPU_SRCS src/index/ivf_gpu/*.cc
        src/index/flat_gpu/*.cc)
    list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_GPU_SRCS})
endif()

include_directories(src)
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR})



add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere faiss easyloggingpp)
target_link_libraries(knowhere PUBLIC faiss easyloggingpp)

if (WITH_DISKANN)
    include(cmake/libs/libdiskann.cmake)
    include_directories(thirdparty/DiskANN)
    target_link_libraries(knowhere PUBLIC diskann aio)
endif()

if (WITH_UT)
    add_subdirectory(tests/ut)
endif()

if(WITH_BENCHMARK)
    add_subdirectory(thirdparty/gtest)
    add_subdirectory(benchmark)
endif()

install(TARGETS knowhere DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
install(FILES "${PROJECT_SOURCE_DIR}/thirdparty/easyloggingpp/src/easylogging++.h"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/knowhere"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
