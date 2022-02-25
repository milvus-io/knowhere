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
include_directories(${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/utils)

set(KNOWHERE_THIRDPARTY_SRC ${KNOWHERE_SOURCE_DIR}/thirdparty)
set(external_srcs
        knowhere/common/Exception.cpp
        knowhere/common/Timer.cpp
        knowhere/common/Utils.cpp
        knowhere/common/Heap.cpp
        knowhere/common/Log.cpp
        ${KNOWHERE_THIRDPARTY_SRC}/easyloggingpp/easylogging++.cc
        )

set(config_srcs
        knowhere/archive/KnowhereConfig.cpp
        )

if ( LINUX OR MSYS )
    include_directories(${KNOWHERE_SOURCE_DIR}/thirdparty/NGT/lib)
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
            knowhere/index/IndexType.cpp
            knowhere/index/vector_index/adapter/VectorAdapter.cpp
            knowhere/index/vector_index/helpers/FaissIO.cpp
            knowhere/index/vector_index/helpers/IndexParameter.cpp
            knowhere/index/vector_index/helpers/DynamicResultSet.cpp
            knowhere/index/vector_index/impl/bruteforce/BruteForce.cpp
            knowhere/index/vector_index/impl/nsg/Distance.cpp
            knowhere/index/vector_index/impl/nsg/NSG.cpp
            knowhere/index/vector_index/impl/nsg/NSGHelper.cpp
            knowhere/index/vector_index/impl/nsg/NSGIO.cpp
            knowhere/index/vector_index/ConfAdapter.cpp
            knowhere/index/vector_index/ConfAdapterMgr.cpp
            knowhere/index/vector_index/FaissBaseBinaryIndex.cpp
            knowhere/index/vector_index/FaissBaseIndex.cpp
            knowhere/index/vector_index/IndexBinaryIDMAP.cpp
            knowhere/index/vector_index/IndexBinaryIVF.cpp
            knowhere/index/vector_index/IndexIDMAP.cpp
            knowhere/index/vector_index/IndexIVF.cpp
            knowhere/index/vector_index/IndexIVFPQ.cpp
            knowhere/index/vector_index/IndexIVFSQ.cpp
            knowhere/index/vector_index/IndexIVFHNSW.cpp
            knowhere/index/vector_index/IndexAnnoy.cpp
            knowhere/index/vector_index/IndexRHNSW.cpp
            knowhere/index/vector_index/IndexHNSW.cpp
            knowhere/index/vector_index/IndexRHNSWFlat.cpp
            knowhere/index/vector_index/IndexRHNSWSQ.cpp
            knowhere/index/vector_index/IndexRHNSWPQ.cpp
            knowhere/index/vector_index/IndexNGT.cpp
            knowhere/index/vector_index/IndexNGTPANNG.cpp
            knowhere/index/vector_index/IndexNGTONNG.cpp
            knowhere/index/vector_index/Statistics.cpp
            knowhere/index/vector_index/VecIndexFactory.cpp
            )

    set(vector_offset_index_srcs
            knowhere/index/vector_offset_index/OffsetBaseIndex.cpp
            knowhere/index/vector_offset_index/IndexIVF_NM.cpp
            knowhere/index/vector_offset_index/IndexNSG_NM.cpp
            )

    if (KNOWHERE_SUPPORT_SPTAG)
        set(vector_index_srcs
                knowhere/index/vector_index/adapter/SptagAdapter.cpp
                knowhere/index/vector_index/helpers/SPTAGParameterMgr.cpp
                knowhere/index/vector_index/IndexSPTAG.cpp
                ${vector_index_srcs}
                )
        set(KNOWHERE_INCLUDE_DIRS
                ${KNOWHERE_SOURCE_DIR}/thirdparty/SPTAG/AnnService
                ${KNOWHERE_INCLUDE_DIRS}
                )
    endif ()

    set(depend_libs
            faiss
            gomp
            gfortran
            pthread
            ngt
            knowhere_utils
            )

    if (KNOWHERE_SUPPORT_SPTAG)
        set(depend_libs
                SPTAGLibStatic
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
                knowhere/index/vector_index/gpu/IndexGPUIDMAP.cpp
                knowhere/index/vector_index/gpu/IndexGPUIVF.cpp
                knowhere/index/vector_index/gpu/IndexGPUIVFPQ.cpp
                knowhere/index/vector_index/gpu/IndexGPUIVFSQ.cpp
                knowhere/index/vector_index/gpu/IndexIVFSQHybrid.cpp
                knowhere/index/vector_index/helpers/Cloner.cpp
                knowhere/index/vector_index/helpers/FaissGpuResourceMgr.cpp
                )

        set(vector_offset_index_srcs ${vector_offset_index_srcs}
                knowhere/index/vector_offset_index/gpu/IndexGPUIVF_NM.cpp
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
    endif ()
    target_link_libraries(knowhere PRIVATE ${depend_libs})
    set(KNOWHERE_INCLUDE_DIRS
        ${KNOWHERE_SOURCE_DIR}/knowhere
        ${KNOWHERE_SOURCE_DIR}/thirdparty
        ${FAISS_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
        ${LAPACK_INCLUDE_DIR}
        )
endif()

if (MACOS)
    set(vector_index_srcs
            knowhere/index/vector_index/impl/bruteforce/BruteForce.cpp
            knowhere/index/IndexType.cpp
            knowhere/index/vector_index/adapter/VectorAdapter.cpp
            knowhere/index/vector_index/helpers/IndexParameter.cpp
            knowhere/index/vector_index/ConfAdapter.cpp
            knowhere/index/vector_index/ConfAdapterMgr.cpp
            knowhere/index/vector_index/Statistics.cpp
            knowhere/index/vector_index/VecIndexFactory.cpp
            )
    if (NOT TARGET knowhere)
        add_library(
                knowhere SHARED
                ${external_srcs}
                ${config_srcs}
                ${vector_index_srcs}
        )
    endif ()

    target_link_libraries(knowhere pthread knowhere_utils)
    set(KNOWHERE_INCLUDE_DIRS ${KNOWHERE_SOURCE_DIR}/knowhere)
endif()

target_include_directories(knowhere PUBLIC ${KNOWHERE_INCLUDE_DIRS})

install(TARGETS knowhere DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY "${KNOWHERE_SOURCE_DIR}/knowhere/knowhere/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/knowhere"
        FILES_MATCHING 
        PATTERN "*.h" 
)
