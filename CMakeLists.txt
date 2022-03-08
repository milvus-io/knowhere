
cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)

include(cmake/utils/utils.cmake)
include(cmake/utils/compiler_check.cmake)
include(cmake/utils/platform_check.cmake)
knowhere_option(USE_CUDA "Build with CUDA" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(KNOWHERE_ENABLE_GPU)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  enable_language(CUDA)
endif()

add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP=1)
add_definitions(-DELPP_THREAD_SAFE=1)
add_definitions(-DINFO=1)
if(__X86_64)
  include(cmake/libs/libcpu_features.cmake)
endif()
#include(cmake/libs/libopenblas.cmake)
find_package(BLAS REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(thirdparty/easyloggingpp/src)
include_directories(thirdparty/faiss)
include_directories(knowhere)
include_directories(thirdparty)
include_directories(thirdparty/nlohmann_json/include)
include_directories(thirdparty/NGT/lib)
include_directories(thirdparty/SPTAG/AnnService)
include_directories(thirdparty/cpu_features/include)
include_directories(thirdparty/any/include)

knowhere_file_glob(
  GLOB_RECURSE KNOWHERE_SRCS knowhere/archive/*.cpp knowhere/cache/*.cpp
  knowhere/common/*.cpp knowhere/index/*.cpp)

if(__X86_64)
knowhere_file_glob(GLOB_RECURSE KNOWHERE_UTILS_SRCS knowhere/utils/*.cpp)
endif()

if(__AARCH64)
knowhere_file_glob(GLOB KNOWHERE_UTILS_SRCS
    knowhere/utils/FaissHookFvec.cpp
    knowhere/utils/distances_simd.cpp)
endif()

knowhere_file_glob(
  GLOB_RECURSE
  KNOWHERE_GPU_SRCS
  knowhere/index/vector_index/gpu/*.cpp
  knowhere/index/vector_offset_index/gpu/*.cpp
  knowhere/index/vector_index/helpers/Cloner.cpp
  knowhere/index/vector_index/helpers/FaissGpuResourceMgr.cpp)

list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_GPU_SRCS})

knowhere_file_glob(
  GLOB_RECURSE
  KNOWHERE_SPTAG_SRCS
  knowhere/index/vector_index/IndexSPTAG.cpp
  knowhere/index/vector_index/IndexNGTPANNG.cpp
  knowhere/index/vector_index/IndexNGTONNG.cpp
  knowhere/index/vector_index/IndexSPTAG.cpp
  knowhere/index/vector_index/helpers/SPTAGParameterMgr.cpp
  knowhere/index/vector_index/adapter/SptagAdapter.cpp
  knowhere/index/vector_index/IndexNGT.cpp)

# TODO: need a better solution
list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_SPTAG_SRCS})

knowhere_file_glob(GLOB_RECURSE KNOWHERE_SPTAG_SRCS
                   knowhere/index/vector_index/IndexSPTAG.cpp)

list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_SPTAG_SRCS})

knowhere_file_glob(
  GLOB FAISS_SRCS thirdparty/faiss/faiss/*.cpp
  thirdparty/faiss/faiss/impl/*.cpp thirdparty/faiss/faiss/invlists/*.cpp
  thirdparty/faiss/faiss/utils/*.cpp)

knowhere_file_glob(GLOB FAISS_AVX512_SRCS
                   thirdparty/faiss/faiss/impl/*avx512.cpp)

list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SRCS})

if(__AARCH64)
knowhere_file_glob(GLOB FAISS_AVX_SRCS
                   thirdparty/faiss/faiss/impl/*avx.cpp)
list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX_SRCS})
endif()


knowhere_file_glob(GLOB_RECURSE EASYLOGGINGPP_SRCS
                   thirdparty/easyloggingpp/src/*.cc)

list(APPEND KNOWHERE_SRCS ${FAISS_SRCS} ${EASYLOGGINGPP_SRCS})

if(__X86_64)
  add_library(faiss_avx512 OBJECT ${FAISS_AVX512_SRCS})
  target_compile_options(
    faiss_avx512
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -mavx512dq
            -mavx512bw>)

  add_library(knowhere_utils OBJECT ${KNOWHERE_UTILS_SRCS})

  target_compile_options(
    knowhere_utils
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -mavx512dq
            -mavx512bw>)
endif()

if(__AARCH64)
    list(APPEND KNOWHERE_SRCS ${KNOWHERE_UTILS_SRCS})
endif()

add_library(knowhere SHARED ${KNOWHERE_SRCS})
if(__X86_64)
  target_compile_options(knowhere PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -msse4.2
                                          -mavx2 -mfma -mf16c>)
endif()

if(__AARCH64)
    target_compile_options(knowhere PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -mno-mmx>)
endif()

set_target_properties(knowhere PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(knowhere PRIVATE FINTEGER=int)
target_compile_definitions(knowhere PRIVATE USE_CPU=1)
find_package(OpenMP REQUIRED)
#find_package(Boost REQUIRED)
include_directories(${OpenMP_CXX_INCLUDE_DIRS})
#add_dependencies(knowhere libopenblas)
if(__X86_64)
  add_dependencies(knowhere libcpu_features)
  add_dependencies(knowhere faiss_avx512)
  add_dependencies(knowhere knowhere_utils)
  target_link_libraries(
    knowhere PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} libcpu_features
                     faiss_avx512 knowhere_utils)
endif()

if(__AARCH64)
  target_link_libraries(knowhere PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# knowhere_file_glob(GLOB KNOWHERE_UNITTEST_SRCS unittest/*.cpp )

set(KNOWHERE_UNITTEST_SRCS
    #unittest/test_annoy.cpp
    unittest/test_binaryidmap.cpp
    unittest/test_binaryivf.cpp
    unittest/test_hnsw.cpp
    unittest/test_idmap.cpp
    unittest/test_ivf.cpp
    unittest/test_ivf_cpu_nm.cpp
    unittest/test_ivf_hnsw.cpp
    unittest/test_rhnsw_flat.cpp
    unittest/test_rhnsw_pq.cpp
    unittest/test_rhnsw_sq8.cpp
    unittest/utils.cpp)

add_executable(knowhere_test_run ${KNOWHERE_UNITTEST_SRCS})

target_compile_definitions(knowhere_test_run PRIVATE ELPP_DISABLE_LOGS=1)

add_dependencies(knowhere_test_run knowhere)
target_include_directories(knowhere_test_run AFTER
                           PRIVATE thirdparty/nlohmann_json/include)
target_include_directories(knowhere_test_run AFTER PRIVATE knowhere)
target_include_directories(knowhere_test_run AFTER PRIVATE thirdparty)

target_include_directories(knowhere_test_run AFTER PRIVATE thirdparty/NGT/lib)

target_include_directories(knowhere_test_run AFTER
                           PRIVATE thirdparty/faiss/faiss)

target_include_directories(knowhere_test_run AFTER
                           PRIVATE thirdparty/SPTAG/AnnService)

target_link_libraries(knowhere_test_run gtest_main gmock knowhere)
add_test(NAME knowhere_test COMMAND knowhere_test_run)

set(FAISS_UNITTEST_SRCS
    thirdparty/faiss/tests/test_binary_flat.cpp
    thirdparty/faiss/tests/test_dealloc_invlists.cpp
    thirdparty/faiss/tests/test_ivfpq_codec.cpp
    thirdparty/faiss/tests/test_ivfpq_indexing.cpp
    thirdparty/faiss/tests/test_lowlevel_ivf.cpp
    thirdparty/faiss/tests/test_merge.cpp
    thirdparty/faiss/tests/test_omp_threads.cpp
    thirdparty/faiss/tests/test_ondisk_ivf.cpp
    thirdparty/faiss/tests/test_pairs_decoding.cpp
    thirdparty/faiss/tests/test_params_override.cpp
    thirdparty/faiss/tests/test_pq_encoding.cpp
    thirdparty/faiss/tests/test_sliding_ivf.cpp
    thirdparty/faiss/tests/test_threaded_index.cpp
    thirdparty/faiss/tests/test_transfer_invlists.cpp)

add_executable(faiss_test_run ${FAISS_UNITTEST_SRCS})

target_compile_definitions(faiss_test_run PRIVATE ELPP_DISABLE_LOGS=1)

target_link_libraries(faiss_test_run gtest_main gmock knowhere OpenMP::OpenMP_CXX)
add_test(NAME faiss_test COMMAND faiss_test_run)

