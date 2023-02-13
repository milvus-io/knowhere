knowhere_file_glob(
  GLOB FAISS_SRCS thirdparty/faiss/faiss/*.cpp
  thirdparty/faiss/faiss/impl/*.cpp thirdparty/faiss/faiss/invlists/*.cpp
  thirdparty/faiss/faiss/utils/*.cpp)

knowhere_file_glob(GLOB FAISS_AVX512_SRCS
                   thirdparty/faiss/faiss/impl/*avx512.cpp)

list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX512_SRCS})

if(USE_CUDA)
  knowhere_file_glob(
    GLOB
    FAISS_GPU_SRCS
    thirdparty/faiss/faiss/gpu/*.cu
    thirdparty/faiss/faiss/gpu/*.cpp
    thirdparty/faiss/faiss/gpu/impl/*.cu
    thirdparty/faiss/faiss/gpu/impl/*.cpp
    thirdparty/faiss/faiss/gpu/utils/*.cu
    thirdparty/faiss/faiss/gpu/utils/*.cpp
    thirdparty/faiss/faiss/gpu/impl/scan/*.cu
    thirdparty/faiss/faiss/gpu/utils/blockselect/*.cu
    thirdparty/faiss/faiss/gpu/utils/blockselect/*.cpp
    thirdparty/faiss/faiss/gpu/utils/warpselect/*.cu)
  list(APPEND FAISS_SRCS ${FAISS_GPU_SRCS})
endif()

if(__X86_64)
  set(UTILS_SRC src/simd/distances_ref.cc src/simd/hook.cc)
  set(UTILS_SSE_SRC src/simd/distances_sse.cc)
  set(UTILS_AVX_SRC src/simd/distances_avx.cc)
  set(UTILS_AVX512_SRC src/simd/distances_avx512.cc)

  add_library(utils_sse OBJECT ${UTILS_SSE_SRC})
  add_library(utils_avx OBJECT ${UTILS_AVX_SRC})
  add_library(utils_avx512 OBJECT ${UTILS_AVX512_SRC})

  target_compile_options(utils_sse PRIVATE -msse4.2)
  target_compile_options(utils_avx PRIVATE -mf16c -mavx2)
  target_compile_options(utils_avx512 PRIVATE -mf16c -mavx512f -mavx512dq
                                              -mavx512bw)

  add_library(
    knowhere_utils STATIC
    ${UTILS_SRC} $<TARGET_OBJECTS:utils_sse> $<TARGET_OBJECTS:utils_avx>
    $<TARGET_OBJECTS:utils_avx512>)
endif()

if(__AARCH64)
  set(UTILS_SRC src/simd/hook.cc src/simd/distances_ref.cc)
  add_library(knowhere_utils STATIC ${UTILS_SRC})
endif()

if(LINUX)
  set(BLA_VENDOR OpenBLAS)
endif()

if(APPLE)
  set(BLA_VENDOR Apple)
endif()

find_package(BLAS REQUIRED)
if(LINUX)
  set(BLA_VENDOR "")
endif()

find_package(LAPACK REQUIRED)

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

  add_library(faiss STATIC ${FAISS_SRCS})

  if(USE_CUDA)
    target_link_libraries(faiss PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(
      faiss PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
  endif()

  add_dependencies(faiss faiss_avx512 knowhere_utils)
  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)
  target_link_libraries(
    faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                 faiss_avx512 knowhere_utils)
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()

if(__AARCH64)
  knowhere_file_glob(GLOB FAISS_AVX_SRCS thirdparty/faiss/faiss/impl/*avx.cpp)

  list(REMOVE_ITEM FAISS_SRCS ${FAISS_AVX_SRCS})
  add_library(faiss STATIC ${FAISS_SRCS})

  if(USE_CUDA)
    target_link_libraries(faiss PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(
      faiss PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
  endif()

  target_compile_options(
    faiss
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  add_dependencies(faiss knowhere_utils)
  target_link_libraries(faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES}
                                     ${LAPACK_LIBRARIES} knowhere_utils)
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()
