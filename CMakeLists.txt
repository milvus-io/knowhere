# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
include(GNUInstallDirs)
include(ExternalProject)
include(cmake/utils/utils.cmake)

knowhere_option(USE_CUDA "Build with CUDA" OFF)
knowhere_option(WITH_UT "Build with UT test" OFF)
knowhere_option(WITH_ASAN "Build with ASAN" OFF)
knowhere_option(WITH_DISKANN "Build with diskann index" OFF)
knowhere_option(WITH_RAFT "Build with RAFT indexes" OFF)
knowhere_option(WITH_BENCHMARK "Build with benchmark" OFF)
knowhere_option(WITH_COVERAGE "Build with coverage" OFF)
knowhere_option(WITH_CCACHE "Build with ccache" ON)
knowhere_option(WITH_PROFILER "Build with profiler" OFF)

if(WITH_RAFT AND NOT USE_CUDA)
  message(WARNING "WITH_RAFT requires USE_CUDA. Setting USE_CUDA to ON.")
  set(USE_CUDA ON)
endif()

if(KNOWHERE_VERSION)
  message(STATUS "Building KNOWHERE version: ${KNOWHERE_VERSION}")
  add_definitions(-DKNOWHERE_VERSION=${KNOWHERE_VERSION})
endif()

if(WITH_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    message(STATUS "Using ccache: ${CCACHE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_FOUND})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_FOUND})
    # let ccache preserve C++ comments, because some of them may be meaningful
    # to the compiler
    set(ENV{CCACHE_COMMENTS} "1")
  endif()
endif()

if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
  if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80;75;70;61)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(${CUDAToolkit_VERSION_MAJOR} GREATER 10)
    # cuda11 support --threads for compile some large .cu more efficient
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--threads=4>)
  endif()
endif()

add_definitions(-DNOT_COMPILE_FOR_SWIG)

include(cmake/utils/compile_flags.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)
include(cmake/libs/libannoy.cmake)
include(cmake/libs/libhnsw.cmake)
include(cmake/libs/libprometheus.cmake)

include_directories(thirdparty/faiss)
include_directories(thirdparty/bitset)

find_package(OpenMP REQUIRED)

if(WITH_RAFT)
  include(cmake/libs/libraft.cmake)
endif()

find_package(nlohmann_json)
if (NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found, fetching from github")
  Include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
    URL_HASH MD5=e8d56bc54621037842ee9f0aeae27746
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(glog)
if (NOT glog_FOUND)
  message(STATUS "glog not found, fetching from github")
  Include(FetchContent)
  FetchContent_Declare(
    glog
    URL https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz
    URL_HASH MD5=c98a6068bc9b8ad9cebaca625ca73aa2
  )
  FetchContent_MakeAvailable(glog)
endif()
set(CMAKE_CXX_STANDARD 17)

if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(WITH_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

knowhere_file_glob(GLOB_RECURSE KNOWHERE_SRCS src/common/*.cc src/index/*.cc
                   src/io/*.cc src/index/*.cu)

set(KNOWHERE_LINKER_LIBS "")

if(WITH_DISKANN)
  include(cmake/libs/libdiskann.cmake)
else()
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_DISKANN_SRCS src/index/diskann/*.cc)
  list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_DISKANN_SRCS})
endif()

if(NOT USE_CUDA)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_GPU_SRCS src/index/flat_gpu/*.cc
                     src/index/ivf_gpu/*.cc)
  list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_GPU_SRCS})
endif()

if(NOT WITH_RAFT)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_RAFT_SRCS src/index/ivf_raft/*.cc
                     src/index/ivf_raft/*.cu)
  list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_RAFT_SRCS})
endif()

include_directories(src)
include_directories(include)

list(APPEND KNOWHERE_LINKER_LIBS faiss)
list(APPEND KNOWHERE_LINKER_LIBS glog::glog)
list(APPEND KNOWHERE_LINKER_LIBS nlohmann_json::nlohmann_json)
list(APPEND KNOWHERE_LINKER_LIBS prometheus-cpp::core prometheus-cpp::pull prometheus-cpp::push)

add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere ${KNOWHERE_LINKER_LIBS})
if(WITH_RAFT)
  list(APPEND KNOWHERE_LINKER_LIBS raft::raft raft::distance raft::nn)
endif()
target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})
target_include_directories(
  knowhere PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WITH_UT)
  add_subdirectory(tests/ut)
endif()

if(WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

install(TARGETS knowhere
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/knowhere"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
