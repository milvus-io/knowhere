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

if ( LINUX OR MSYS )
    include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG/AnnService)
    include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/NGT/lib)
endif ()
include_directories(${KNOWHERE_SOURCE_DIR}/knowhere)
include_directories(${KNOWHERE_SOURCE_DIR})

set( DEPEND_LIBS
        gtest
        gmock
        gtest_main
        gmock_main
        )

if ( LINUX )
    set( DEPEND_LIBS
            ${DEPEND_LIBS}
            faiss
            ngt
            openblas
            )
endif ()

set( BASIC_LIBS gomp gfortran pthread )

set( UTIL_SRCS
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/adapter/VectorAdapter.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/helpers/IndexParameter.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/Statistics.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/IndexType.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/common/Exception.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/common/Timer.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/common/Utils.cpp
        ${KNOWHERE_SOURCE_DIR}/unittest/utils.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/impl/bruteforce/SimpleIndexFlat.cpp
        ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/impl/bruteforce/BruteForce.cpp
        )

if ( LINUX OR MSYS )
    set( UTIL_SRCS
            ${UTIL_SRCS}
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/helpers/FaissIO.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/helpers/DynamicResultSet.cpp
            )
endif ()

if ( ( LINUX OR MSYS ) AND KNOWHERE_GPU_VERSION )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    link_directories( "${CUDA_TOOLKIT_ROOT_DIR}/lib64" )
    set( CUDA_LIB
            cudart
            cublas
            )
    set( BASIC_LIBS ${basic_libs} ${cuda_lib} )
    set( UTIL_SRCS ${util_srcs}
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/helpers/FaissGpuResourceMgr.cpp
            )
endif ()

if ( LINUX OR MSYS )
    set( FAISS_SRCS
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/FaissBaseIndex.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/FaissBaseBinaryIndex.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexBinaryIDMAP.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexBinaryIVF.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexIDMAP.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexIVF.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexIVFSQ.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexIVFPQ.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/IndexIVFHNSW.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_offset_index/OffsetBaseIndex.cpp
            ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_offset_index/IndexIVF_NM.cpp
            )

    if ( KNOWHERE_GPU_VERSION )
        set( FAISS_SRCS ${faiss_srcs}
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/ConfAdapter.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/helpers/Cloner.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/gpu/IndexGPUIDMAP.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/gpu/IndexGPUIVF.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/gpu/IndexGPUIVFSQ.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/gpu/IndexGPUIVFPQ.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_index/gpu/IndexIVFSQHybrid.cpp
                ${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.cpp
                )
    endif ()

    ###############################################################################
    #<NSG-TEST>
    add_definitions( -std=c++11 -O3 -march=native -Werror -DINFO )
endif ()

find_package( OpenMP REQUIRED )
if ( OpenMP_FOUND )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
else ()
    message( FATAL_ERROR "no OpenMP supprot" )
endif ()

set( ALL_INDEXING_SRCS
        ${UTIL_SRCS}
        test_bruteforce.cpp
        test_common.cpp
        test_customized_index.cpp
        test_structured_index_sort.cpp
        test_structured_index_flat.cpp
        )

if ( LINUX OR MSYS )
    set(ALL_INDEXING_SRCS
            ${ALL_INDEXING_SRCS}
            test_annoy.cpp
            test_binaryidmap.cpp
            test_binaryivf.cpp
            test_hnsw.cpp
            test_idmap.cpp
            test_instructionset.cpp
            test_ivf.cpp
            test_ivf_cpu_nm.cpp
            test_ivf_hnsw.cpp
            test_ngtonng.cpp
            test_ngtpanng.cpp
            test_nsg.cpp
            test_rhnsw_flat.cpp
            test_rhnsw_pq.cpp
            test_rhnsw_sq8.cpp
            )
    if ( KNOWHERE_GPU_VERSION )
        set(ALL_INDEXING_SRCS
                ${ALL_INDEXING_SRCS}
                test_ivf_gpu_nm.cpp
                test_gpuresource.cpp
                )
    endif ()
endif ()

if ( KNOWHERE_SUPPORT_SPTAG )
    set( ALL_INDEXING_SRCS
            test_sptag.cpp
            )
endif ()

add_executable( test_knowhere ${ALL_INDEXING_SRCS} )
target_link_libraries( test_knowhere knowhere ${DEPEND_LIBS} )

install(TARGETS test_knowhere DESTINATION unittest)
