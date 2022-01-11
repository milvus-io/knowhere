#!/bin/bash

# Exit immediately for non zero status
set -e

UNAME="$(uname -s)"

case "${UNAME}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${UNAME}"
esac

BUILD_OUTPUT_DIR="cmake_build"
BUILD_TYPE="Debug"
BUILD_UNITTEST="OFF"
INSTALL_PREFIX=$(pwd)/knowhere
MAKE_CLEAN="OFF"
BUILD_COVERAGE="OFF"
SUPPORT_PROFILING="OFF"
RUN_CPPLINT="OFF"
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
SUPPORT_GPU="OFF" #defaults to CPU version

while getopts "p:t:cglruzh" arg; do
    case $arg in
        c)
            BUILD_COVERAGE="ON"
            ;;
        g)
            SUPPORT_GPU="ON"
            ;;
        l)
            RUN_CPPLINT="ON"
            ;;
        p)
            INSTALL_PREFIX=$OPTARG
            ;;
        r)
            MAKE_CLEAN="ON"
            ;;
        t)
            BUILD_TYPE=$OPTARG # BUILD_TYPE
            ;;
        u)
            echo "Build and run unittest cases" ;
            BUILD_UNITTEST="ON";
            ;;
        z)
            SUPPORT_PROFILING="ON"
            ;;
        h) # help
            echo "

parameter:
-c: code coverage(default: OFF)
-g: build GPU version(default: OFF)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-p: install prefix(default: $(pwd)/knowhere)
-r: remove previous build directory(default: OFF)
-t: build type(default: Debug)
-u: building unit test options(default: OFF)
-z: support CPU profiling(default: OFF)
-h: help

usage:
./build.sh -t \${BUILD_TYPE} [-c] [-g] [-l] [-r] [-u] [-z]
            "
            exit 0
            ;;
        ?)
            echo "unknown argument"
            exit 1
            ;;
    esac
done

if [[ ! -d ${BUILD_OUTPUT_DIR} ]]; then
    mkdir ${BUILD_OUTPUT_DIR}
fi

cd ${BUILD_OUTPUT_DIR}

if [[ ${MAKE_CLEAN} == "ON" ]]; then
  echo "Running make clean in ${BUILD_OUTPUT_DIR} ..."
  make clean
  echo "Running make clean in thirdparty/faiss ..."
  cd ../thirdparty/faiss
  make clean
  exit 0
fi

CMAKE_CMD="cmake -DBUILD_UNIT_TEST=${BUILD_UNITTEST} \
-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
-DMILVUS_ENABLE_PROFILING=${SUPPORT_PROFILING} \
-DKNOWHERE_GPU_VERSION=${SUPPORT_GPU} \
../"

echo ${CMAKE_CMD}
${CMAKE_CMD}

if [[ ${RUN_CPPLINT} == "ON" ]]; then
  # cpplint check
  make lint-knowhere
  if [ $? -ne 0 ]; then
    echo "ERROR! cpplint check failed"
    exit 1
  fi
  echo "cpplint check passed!"

  # clang-format check
  make check-clang-format-knowhere
  if [ $? -ne 0 ]; then
    echo "ERROR! clang-format check failed"
    exit 1
  fi
  echo "clang-format check passed!"

  # clang-tidy check
#  make check-clang-tidy-knowhere
#  if [ $? -ne 0 ]; then
#      echo "ERROR! clang-tidy check failed"
#      exit 1
#  fi
#  echo "clang-tidy check passed!"
else
  # compile and build
  make -j 8 install || exit 1
fi
