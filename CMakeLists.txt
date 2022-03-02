
cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)

include(cmake/utils/utils.cmake)
include(CheckCXXCompilerFlag)
knowhere_option(USE_CUDA "Build with CUDA" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
check_cxx_compiler_flag(-std=c++17 COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

if(NOT COMPILER_SUPPORTS_CXX17)
  message(
    FATAL_ERROR
      "C++17 needed. Therefore a gcc compiler with a version higher than 4.3 is needed."
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  check_cxx_compiler_flag("-fopenmp=libomp" COMPILER_SUPPORTS_OMP)
  set(CMAKE_CXX_FLAGS "-fopenmp=libomp ${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  check_cxx_compiler_flag("-Xclang -fopenmp" COMPILER_SUPPORTS_OMP)
  set(CMAKE_CXX_FLAGS "-Xclang -fopenmp ${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORTS_OMP)
  set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
endif()

if(NOT COMPILER_SUPPORTS_OMP)
  message(FATAL_ERROR "compiler must support openmp.")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS
      "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS
      "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
endif()

if(KNOWHERE_ENABLE_GPU)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  enable_language(CUDA)
endif()

add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP=1)
add_definitions(-DELPP_THREAD_SAFE=1)
add_definitions(-DINFO=1)

include(cmake/libs/libcpu_features.cmake)
# include(cmake/libs/libopenblas.cmake)
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

knowhere_file_glob(
  GLOB_RECURSE KNOWHERE_SRCS knowhere/archive/*.cpp knowhere/cache/*.cpp
  knowhere/common/*.cpp knowhere/index/*.cpp)

knowhere_file_glob(GLOB_RECURSE KNOWHERE_UTILS_SRCS knowhere/utils/*.cpp)

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

knowhere_file_glob(GLOB_RECURSE EASYLOGGINGPP_SRCS
                   thirdparty/easyloggingpp/src/*.cc)

list(APPEND KNOWHERE_SRCS ${FAISS_SRCS} ${EASYLOGGINGPP_SRCS})

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
add_library(knowhere SHARED ${KNOWHERE_SRCS})
target_compile_options(knowhere PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -msse4.2
                                        -mavx2 -mfma -mf16c>)
set_target_properties(knowhere PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(knowhere PRIVATE FINTEGER=int)
target_compile_definitions(knowhere PRIVATE USE_CPU=1)
find_package(OpenMP REQUIRED)
# add_dependencies(knowhere libopenblas)
add_dependencies(knowhere libcpu_features)
add_dependencies(knowhere faiss_avx512)
add_dependencies(knowhere knowhere_utils)
target_link_libraries(
  knowhere PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} libcpu_features
                   faiss_avx512 knowhere_utils)

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
    unittest/test_annoy.cpp
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

target_link_libraries(faiss_test_run gtest_main gmock knowhere)
add_test(NAME faiss_test COMMAND faiss_test_run)

