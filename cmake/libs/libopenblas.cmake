include(ExternalProject)

set(MAKE_COMMAND "make")
set(OPENBLAS_DIR ${CMAKE_BINARY_DIR}/openblas)
set(OPENBLAS_INC ${OPENBLAS_DIR}/include)
set(OPENBLAS_LIB ${OPENBLAS_DIR}/lib/libopenblas.a)
ExternalProject_Add(
  openblas
  URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.19/OpenBLAS-0.3.19.tar.gz
  URL_HASH MD5=9721d04d72a7d601c81eafb54520ba2c
  CMAKE_GENERATOR "Unix Makefiles"
  PREFIX ${OPENBLAS_DIR}
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_INSTALL_PREFIX=${OPENBLAS_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DNO_LAPACKE=0
             -DBUILD_STATIC_LIBS=ON
             -DBUILD_SHARED_LIBS=ON
             -DDYNAMIC_ARCH=1
             -DINTERFACE64=0
             -DNUM_THREADS=128
             -DUSE_THREAD=0
             -DUSE_OPENMP=0
  BUILD_COMMAND ${MAKE_COMMAND}
  BUILD_BYPRODUCTS ${OPENBLAS_LIB})

file(MAKE_DIRECTORY ${OPENBLAS_INC})

add_library(libopenblas STATIC IMPORTED GLOBAL)
add_dependencies(libopenblas openblas)
set_target_properties(
  libopenblas PROPERTIES IMPORTED_LOCATION ${OPENBLAS_LIB}
                         INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_DIR})
