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

include_directories(${INDEX_SOURCE_DIR}/knowhere)
include_directories(${INDEX_SOURCE_DIR}/thirdparty)
add_subdirectory(${INDEX_SOURCE_DIR}/knowhere/knowhere/utils)

set(KNOWHERE_THIRDPARTY_SRC ${INDEX_SOURCE_DIR}/thirdparty)
set(external_srcs
        knowhere/common/Exception.cpp
        knowhere/common/Timer.cpp
        knowhere/common/Status.cpp
        knowhere/common/Utils.cpp
        ${MILVUS_THIRDPARTY_SRC}/nlohmann/json.hpp
        )

set (LOG_SRC
        knowhere/common/Log.cpp
        ${MILVUS_THIRDPARTY_SRC}/easyloggingpp/easylogging++.cc
        )

add_library(index_log STATIC ${LOG_SRC})
set_target_properties(index_log PROPERTIES RULE_LAUNCH_COMPILE "")
set_target_properties(index_log PROPERTIES RULE_LAUNCH_LINK "")

set(config_srcs
        knowhere/archive/KnowhereConfig.cpp
        )

if (LINUX)
    include_directories(${INDEX_SOURCE_DIR}/thirdparty/NGT/lib)
    if (MILVUS_SUPPORT_SPTAG)
        include_directories(${INDEX_SOURCE_DIR}/thirdparty/SPTAG/AnnService)

        set(SPTAG_SOURCE_DIR ${INDEX_SOURCE_DIR}/thirdparty/SPTAG)
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

    if (MILVUS_SUPPORT_SPTAG)
        set(vector_index_srcs
                knowhere/index/vector_index/adapter/SptagAdapter.cpp
                knowhere/index/vector_index/helpers/SPTAGParameterMgr.cpp
                knowhere/index/vector_index/IndexSPTAG.cpp
                ${vector_index_srcs}
                )
        set(INDEX_INCLUDE_DIRS
                ${INDEX_SOURCE_DIR}/thirdparty/SPTAG/AnnService
                ${INDEX_INCLUDE_DIRS}
                )
    endif ()

    set(depend_libs
            faiss
            gomp
            gfortran
            pthread
            ngt
            index_log
            )

    if (MILVUS_SUPPORT_SPTAG)
        set(depend_libs
                SPTAGLibStatic
                ${depend_libs}
                )
    endif ()


    if (MILVUS_GPU_VERSION)
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
                knowhere STATIC
                ${external_srcs}
                ${config_srcs}
                ${vector_index_srcs}
                ${vector_offset_index_srcs}
        )
    endif ()
    target_link_libraries(knowhere ${depend_libs})
    set(INDEX_INCLUDE_DIRS
        ${INDEX_SOURCE_DIR}/knowhere
        ${INDEX_SOURCE_DIR}/thirdparty
        ${FAISS_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
        ${LAPACK_INCLUDE_DIR}
        )
endif()

if (MACOS)
    set(vector_index_srcs
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
                knowhere STATIC
                ${external_srcs}
                ${config_srcs}
                ${vector_index_srcs}
        )
    endif ()

    target_link_libraries(knowhere gfortran pthread index_log knowhere_utils)
    set(INDEX_INCLUDE_DIRS ${INDEX_SOURCE_DIR}/knowhere)
endif()

target_include_directories(knowhere PUBLIC ${INDEX_INCLUDE_DIRS})
install(TARGETS index_log DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS knowhere DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY "${INDEX_SOURCE_DIR}/knowhere/cache/" 
        DESTINATION "include/knowhere/cache"
        FILES_MATCHING 
        PATTERN "*.h" 
)

install(DIRECTORY "${INDEX_SOURCE_DIR}/knowhere/knowhere/" 
        DESTINATION "include/knowhere"
        FILES_MATCHING 
        PATTERN "*.h" 
)
