# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss
  VERSION 1.7.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "sse4", "avx2", "avx512".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_CCACHE "Build with ccache." ON)

if (FAISS_ENABLE_CCACHE)
    find_program(CCACHE_FOUND ccache)
    if (CCACHE_FOUND)
        message(STATUS "Using ccache: ${CCACHE_FOUND}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_FOUND})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_FOUND})
        # let ccache preserve C++ comments, because some of them may be
        # meaningful to the compiler
        set(ENV{CCACHE_COMMENTS} "1")
    endif (CCACHE_FOUND)
endif()

if (FAISS_CFLAGS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FAISS_CFLAGS} ${FAISS_CPPFLAGS}")
endif()

if (FAISS_CXXFLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FAISS_CXXFLAGS} ${FAISS_CPPFLAGS}")
endif()

if (FAISS_ENABLE_GPU)
  # externalproject_add use semicolon to separate commands, revert to desired format
  if (CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE ":" ";" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  enable_language(CUDA)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CPU")
endif()

add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

add_subdirectory(demos)
add_subdirectory(tutorial/cpp)

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  if(FAISS_ENABLE_GPU)
    add_subdirectory(faiss/gpu/test)
  endif()
endif()
