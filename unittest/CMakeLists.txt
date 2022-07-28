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
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/faiss)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG/AnnService)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/NGT/lib)
include_directories(${KNOWHERE_SOURCE_DIR}/knowhere)
include_directories(${KNOWHERE_SOURCE_DIR})

set( DEPEND_LIBS
        gtest
        gmock
        gtest_main
        gmock_main
        )

if ( LINUX AND ENABLE_PROFILING )
    set( DEPEND_LIBS
            ${DEPEND_LIBS}
            gperftools
            )
endif ()

if ( ( LINUX OR MSYS ) AND KNOWHERE_GPU_VERSION )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    link_directories( "${CUDA_TOOLKIT_ROOT_DIR}/lib64" )
    set( CUDA_LIB
            cudart
            cublas
            )
endif ()

set( UTIL_SRCS
        ${KNOWHERE_SOURCE_DIR}/unittest/utils.cpp
        )

set( ALL_INDEXING_SRCS
        ${UTIL_SRCS}
        test_common.cpp
        )

set(ALL_INDEXING_SRCS
        ${ALL_INDEXING_SRCS}
        test_annoy.cpp
        test_binaryidmap.cpp
        test_binaryivf.cpp
        test_bruteforce.cpp
        test_customized_index.cpp
        test_hnsw.cpp
        test_idmap.cpp
        test_ivf.cpp
        test_ivf_nm.cpp
        test_ivf_hnsw.cpp
        test_rhnsw.cpp
        test_vecindex.cpp
        )

if ( LINUX OR MSYS )
    set(ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_instructionset.cpp
            )
endif()

if ( KNOWHERE_GPU_VERSION )
    set(ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_gpuresource.cpp
            )
endif ()

if ( KNOWHERE_SUPPORT_SPTAG )
    set( ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_sptag.cpp
            )
endif ()

if ( KNOWHERE_SUPPORT_NGT )
    set( ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_ngtonng.cpp
            test_ngtpanng.cpp
            )
    set( DEPEND_LIBS
            ${DEPEND_LIBS}
            ngt
            )
endif ()

if ( KNOWHERE_SUPPORT_NSG )
    set( ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_nsg.cpp
            )
endif ()

add_executable( test_knowhere ${ALL_INDEXING_SRCS} )
target_link_libraries( test_knowhere knowhere ${DEPEND_LIBS} )

install(TARGETS test_knowhere DESTINATION unittest)

#add_subdirectory(benchmark)
