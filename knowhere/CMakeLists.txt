#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

include_directories(${KNOWHERE_SOURCE_DIR}/knowhere)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty)
include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/faiss)
include_directories(${KNOWHERE_SOURCE_DIR}/knowhere/utils)

set(KNOWHERE_THIRDPARTY_SRC ${KNOWHERE_SOURCE_DIR}/thirdparty)
set(external_srcs
        common/Exception.cpp
        common/Timer.cpp
        common/Log.cpp
        ${KNOWHERE_THIRDPARTY_SRC}/easyloggingpp/easylogging++.cc
        )

set(config_srcs
        archive/KnowhereConfig.cpp
        archive/BruteForce.cpp
        )

if (KNOWHERE_SUPPORT_NGT)
    include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/NGT/lib)
endif ()

if (KNOWHERE_SUPPORT_SPTAG)
    include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG/AnnService)

    set(SPTAG_SOURCE_DIR ${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG)
    file(GLOB HDR_FILES
            ${SPTAG_SOURCE_DIR}/AnnService/inc/Core/*.h
            ${SPTAG_SOURCE_DIR}/AnnService/inc/Core/Common/*.h
            ${SPTAG_SOURCE_DIR}/AnnService/inc/Core/BKT/*.h
            ${SPTAG_SOURCE_DIR}/AnnService/inc/Core/KDT/*.h
            ${SPTAG_SOURCE_DIR}/AnnService/inc/Helper/*.h)
    file(GLOB SRC_FILES
            ${SPTAG_SOURCE_DIR}/AnnService/src/Core/*.cpp
            ${SPTAG_SOURCE_DIR}/AnnService/src/Core/Common/*.cpp
            ${SPTAG_SOURCE_DIR}/AnnService/src/Core/BKT/*.cpp
            ${SPTAG_SOURCE_DIR}/AnnService/src/Core/KDT/*.cpp
            ${SPTAG_SOURCE_DIR}/AnnService/src/Helper/*.cpp)

    if (NOT TARGET SPTAGLibStatic)
        add_library(SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
    endif ()
endif ()

set(vector_index_srcs
        index/VecIndexFactory.cpp
        index/vector_index/adapter/VectorAdapter.cpp
        index/vector_index/helpers/FaissIO.cpp
        index/vector_index/helpers/Slice.cpp
        index/vector_index/ConfAdapter.cpp
        index/vector_index/ConfAdapterMgr.cpp
        index/vector_index/FaissBaseBinaryIndex.cpp
        index/vector_index/FaissBaseIndex.cpp
        index/vector_index/IndexBinaryIDMAP.cpp
        index/vector_index/IndexBinaryIVF.cpp
        index/vector_index/IndexIDMAP.cpp
        index/vector_index/IndexIVF.cpp
        index/vector_index/IndexIVFPQ.cpp
        index/vector_index/IndexIVFSQ.cpp
        index/vector_index/IndexIVFHNSW.cpp
        index/vector_index/IndexAnnoy.cpp
        index/vector_index/IndexRHNSW.cpp
        index/vector_index/IndexHNSW.cpp
        index/vector_index/IndexRHNSWFlat.cpp
        index/vector_index/IndexRHNSWSQ.cpp
        index/vector_index/IndexRHNSWPQ.cpp
        index/vector_index/Statistics.cpp
        )

set(vector_offset_index_srcs
        index/vector_offset_index/OffsetBaseIndex.cpp
        index/vector_offset_index/IndexIVF_NM.cpp
        )

if (KNOWHERE_SUPPORT_SPTAG)
    set(vector_index_srcs
            index/vector_index/adapter/SptagAdapter.cpp
            index/vector_index/helpers/SPTAGParameterMgr.cpp
            index/vector_index/IndexSPTAG.cpp
            ${vector_index_srcs}
            )
    set(KNOWHERE_INCLUDE_DIRS
            ${KNOWHERE_INCLUDE_DIRS}
            ${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG/AnnService
            )
endif ()

if (KNOWHERE_SUPPORT_NGT)
    set(vector_index_srcs
            index/vector_index/IndexNGT.cpp
            index/vector_index/IndexNGTPANNG.cpp
            index/vector_index/IndexNGTONNG.cpp
            ${vector_index_srcs}
            )
endif ()

if (KNOWHERE_SUPPORT_NSG)
    set(vector_index_srcs
            index/vector_index/impl/nsg/Distance.cpp
            index/vector_index/impl/nsg/NSG.cpp
            index/vector_index/impl/nsg/NSGHelper.cpp
            index/vector_index/impl/nsg/NSGIO.cpp
            ${vector_index_srcs}
            )
    set(vector_offset_index_srcs
            index/vector_offset_index/IndexNSG_NM.cpp
            ${vector_offset_index_srcs}
            )
endif ()

set(depend_libs
        faiss
        pthread
        knowhere_utils
        )

if (KNOWHERE_SUPPORT_SPTAG)
    set(depend_libs
            SPTAGLibStatic
            ${depend_libs}
            )
endif ()

if (KNOWHERE_SUPPORT_NGT)
    set(depend_libs
            ngt
            ${depend_libs}
            )
endif ()

if (KNOWHERE_GPU_VERSION)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    set(cuda_lib
            cudart
            cublas
            )
    set(depend_libs ${depend_libs}
            ${cuda_lib}
            )

    set(vector_index_srcs ${vector_index_srcs}
            index/vector_index/gpu/IndexGPUIDMAP.cpp
            index/vector_index/gpu/IndexGPUIVF.cpp
            index/vector_index/gpu/IndexGPUIVFPQ.cpp
            index/vector_index/gpu/IndexGPUIVFSQ.cpp
            index/vector_index/gpu/IndexIVFSQHybrid.cpp
            index/vector_index/helpers/Cloner.cpp
            index/vector_index/helpers/FaissGpuResourceMgr.cpp
            )

    set(vector_offset_index_srcs ${vector_offset_index_srcs}
            index/vector_offset_index/gpu/IndexGPUIVF_NM.cpp
            )
endif ()

if (NOT TARGET knowhere)
    add_library(
            knowhere SHARED
            ${external_srcs}
            ${config_srcs}
            ${vector_index_srcs}
            ${vector_offset_index_srcs}
    )
    target_include_directories(knowhere PUBLIC ${KNOWHERE_SOURCE_DIR}/knowere)
endif ()

target_link_libraries(knowhere ${depend_libs})

set (KNOWHERE_INCLUDE_DIRS
        ${KNOWHERE_SOURCE_DIR}
        ${KNOWHERE_SOURCE_DIR}/thirdparty
        ${FAISS_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
        ${LAPACK_INCLUDE_DIR}
        )

target_include_directories(knowhere PUBLIC ${KNOWHERE_INCLUDE_DIRS})

install(TARGETS knowhere DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${KNOWHERE_SOURCE_DIR}/knowhere/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/knowhere"
        FILES_MATCHING
        PATTERN "*.h"
)
