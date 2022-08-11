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

BUILD_DIR="cmake_build"
OUTPUT_DIR="output"
BUILD_TYPE="Debug"
BUILD_UNITTEST="OFF"
INSTALL_PREFIX=$(pwd)/${OUTPUT_DIR}
MAKE_CLEAN="OFF"
BUILD_COVERAGE="OFF"
SUPPORT_PROFILING="OFF"
RUN_CPPLINT="OFF"
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
SUPPORT_GPU="OFF" #defaults to CPU version
ENABLE_SANITIZER="OFF"

while getopts "p:t:cglrsuzh" arg; do
    case $arg in
        c)
            BUILD_COVERAGE="ON" ;;
        g)
            SUPPORT_GPU="ON" ;;
        l)
            RUN_CPPLINT="ON" ;;
        p)
            INSTALL_PREFIX=$OPTARG ;;
        r)
            MAKE_CLEAN="ON" ;;
        s)
            ENABLE_SANITIZER="ON";;
        t)
            BUILD_TYPE=$OPTARG ;;
        u)
            echo "Build and run unittest cases"
            BUILD_UNITTEST="ON" ;;
        z)
            SUPPORT_PROFILING="ON" ;;
        h) # help
            echo "

parameter:
-c: code coverage(default: OFF)
-g: build GPU version(default: OFF)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-p: install prefix(default: $(pwd)/knowhere)
-r: remove previous build directory(default: OFF)
-s: run sanitizer check (default: OFF)
-t: build type(default: Debug)
-u: building unit test options(default: OFF)
-z: support CPU profiling(default: OFF)
-h: help

usage:
./build.sh -t \${BUILD_TYPE} [-c] [-g] [-l] [-r] [-s] [-u] [-z]
            "
            exit 0 ;;
        ?)
            echo "unknown argument"
            exit 1 ;;
    esac
done

if [[ ${MAKE_CLEAN} == "ON" ]]; then
  echo "Remove ${BUILD_DIR} ${OUTPUT_DIR} ..."
  rm -rf ${BUILD_DIR} ${OUTPUT_DIR}
  echo "Clean faiss ..."
  cd thirdparty/faiss
  rm -rf CMakeFiles _deps CMakeCache.txt
  exit 0
fi

if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir ${BUILD_DIR}
fi

cd ${BUILD_DIR}

CMAKE_CMD="cmake -DBUILD_UNIT_TEST=${BUILD_UNITTEST} \
-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DBUILD_COVERAGE=${BUILD_COVERAGE} \
-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
-DENABLE_PROFILING=${SUPPORT_PROFILING} \
-DKNOWHERE_GPU_VERSION=${SUPPORT_GPU} \
-DENABLE_SANITIZER=${ENABLE_SANITIZER} \
../"

echo ${CMAKE_CMD}
if [[ "$MACHINE" == "MinGw" ]] ; then
    # force makefile for MinGW
    ${CMAKE_CMD} -G "MSYS Makefiles"
else
    ${CMAKE_CMD}
fi

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
