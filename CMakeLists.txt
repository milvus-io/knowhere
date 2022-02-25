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

cmake_minimum_required(VERSION 3.18)
message( STATUS "------------------------------KNOWHERE-----------------------------------" )
message( STATUS "Building using CMake version: ${CMAKE_VERSION}" )

add_definitions( -DAUTO_INITIALIZE_EASYLOGGINGPP )

# Note: Cross compiling arguments must be set before project annotation, when ${CMAKE_SYSTEM_NAME} is not available.
# We have to use the "APPLE" variable here as a result.
if ( APPLE )
    set ( MACOS TRUE )
    set ( CMAKE_CROSSCOMPILING TRUE )
    set ( RUN_HAVE_GNU_POSIX_REGEX 0 )
    set ( CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang" )
    set ( CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++" )
endif ()

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include( UtilsKw )

#******************* Build time, type and code version *****************
get_current_time( BUILD_TIME )
message( STATUS "Build time = ${BUILD_TIME}" )

get_build_type( TARGET  BUILD_TYPE
                        DEFAULT "Release" )
message( STATUS "Build type = ${BUILD_TYPE}" )

get_knowhere_version( TARGET KNOWHERE_VERSION
                             DEFAULT "1.0.1" )
message( STATUS "Build version = ${KNOWHERE_VERSION}" )

get_last_commit_id( LAST_COMMIT_ID )
message( STATUS "LAST_COMMIT_ID = ${LAST_COMMIT_ID}" )

#configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in
#                ${CMAKE_CURRENT_SOURCE_DIR}/src/version.h @ONLY )

# unset( CMAKE_EXPORT_COMPILE_COMMANDS CACHE)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )


#************************** Project Settings ***************************
project( knowhere VERSION "${KNOWHERE_VERSION}" )
project( knowhere LANGUAGES C CXX )

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED on )

set( KNOWHERE_SOURCE_DIR ${PROJECT_SOURCE_DIR} )
set( KNOWHERE_BINARY_DIR ${PROJECT_BINARY_DIR} )
set( KNOWHERE_THIRDPARTY_SRC ${PROJECT_SOURCE_DIR}/thirdparty )

# This will set RPATH to all excutable TARGET
# self-installed dynamic libraries will be correctly linked by excutable
set( CMAKE_INSTALL_RPATH "/usr/lib" "${CMAKE_INSTALL_PREFIX}/lib" )
set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

set( CMAKE_CUDA_COMPILER  "/usr/local/cuda/bin/nvcc" )
include( CheckCXXCompilerFlag )

if ( ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
    message(STATUS "==============Darwin Environment==============")
    set( MACOS TRUE )
    set( CMAKE_CXX_FLAGS "-L/usr/local/opt/libomp/lib -I/usr/local/opt/llvm/include -I/usr/local/include -I/usr/local/opt/openblas/include -I/usr/local/opt/libomp/include -DBOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED=1" )
    check_cxx_compiler_flag( -std=c++11 HAS_STD_CPP11_FLAG )
    if ( HAS_STD_CPP11_FLAG )
        add_compile_options(-std=c++11)
    endif ()
    if ( CMAKE_C_COMPILER_ID MATCHES "Clang" )
        set( OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE )
        set( OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE )
        set( OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE )
        set( OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE )
        set( OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE )
        set( OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE )
    endif ()
    if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
        set( OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
        set( OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE )
        set( OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE )
        set( OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE )
        set( OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE )
        set( OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE )
    endif ()
elseif ( "${CMAKE_SYSTEM}" MATCHES "Linux" )
    message( STATUS "==============Linux Environment===============" )
    set ( LINUX TRUE )
elseif ( MSYS )
    message( STATUS "==============MSYS Environment===============" )
else ()
    message( FATAL_ERROR "Unsupported platform!" )
endif ()


#***************************** Dependencies ****************************
include( GNUInstallDirs )
include( DefineOptionsKw )
include( BuildUtilsKw )
include( ExternalProject )

if ( LINUX OR MSYS )
    include( ThirdPartyPackagesKw )
endif ()

using_ccache_if_defined( KNOWHERE_USE_CCACHE )


#************************** Compiler arguments *************************
if ( KNOWHERE_GPU_VERSION )
    message( STATUS "Building Knowhere GPU version" )
    add_compile_definitions( "KNOWHERE_GPU_VERSION" )
    enable_language( CUDA )
    find_package( CUDA 10 REQUIRED )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64 )
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -std=c++11 -D_FORCE_INLINES --expt-extended-lambda" )
    if ( CCACHE_FOUND )
        set( CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_FOUND}" )
    endif ()
else ()
    message( STATUS "Building Knowhere CPU version" )
endif ()

if ( KNOWHERE_SUPPORT_SPTAG )
    message( STATUS "Building Knowhere with SPTAG supported" )
    add_compile_definitions( "KNOWHERE_SUPPORT_SPTAG" )
endif ()

if ( CMAKE_BUILD_TYPE STREQUAL "Release" )
    append_flags( CMAKE_CXX_FLAGS FLAGS "-O3" )
    append_flags( CUDA_NVCC_FLAGS FLAGS "-O3" )
else ()
    append_flags( CMAKE_CXX_FLAGS FLAGS "-O0" "-g" )
    append_flags( CUDA_NVCC_FLAGS FLAGS "-O0" "-g" )
endif ()

append_flags( CMAKE_CXX_FLAGS
              FLAGS
                    "-fPIC"
                    "-DELPP_THREAD_SAFE"
                    "-fopenmp"
                    )

# option "-rdynamic" is to print out detailed backtrace info for debug-ease
if ( LINUX )
    append_flags( CMAKE_CXX_FLAGS FLAGS "-Werror" "-rdynamic" )
endif ()

if ( MSYS )
    append_flags( CMAKE_CXX_FLAGS FLAGS "-Werror" )
endif ()

#*********************** Coding style check tools **********************
find_package( Python COMPONENTS Interpreter Development )
find_package( ClangToolsKw )
set( BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build-support" )

if ( "$ENV{CMAKE_EXPORT_COMPILE_COMMANDS}" STREQUAL "1" OR CLANG_TIDY_FOUND )
  # Generate a Clang compile_commands.json "compilation database" file for use
  # with various development tools, such as Vim's YouCompleteMe plugin.
  # See http://clang.llvm.org/docs/JSONCompilationDatabase.html
  set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
  set( CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} )
endif ()

#
# "make lint" target
#
if ( NOT KNOWHERE_VERBOSE_LINT )
    set( KNOWHERE_LINT_QUIET "--quiet" )
endif ()

if ( NOT LINT_EXCLUSIONS_FILE )
    # source files matching a glob from a line in this file
    # will be excluded from linting (cpplint, clang-tidy, clang-format)
    set( LINT_EXCLUSIONS_FILE ${BUILD_SUPPORT_DIR}/lint_exclusions.txt )
endif ()

if ( NOT IGNORE_CHECKS_FILE )
    set( IGNORE_CHECKS_FILE ${BUILD_SUPPORT_DIR}/ignore_checks.txt )
endif ()

find_program( CPPLINT_BIN NAMES cpplint cpplint.py HINTS ${BUILD_SUPPORT_DIR} )
message( STATUS "Found cpplint executable at ${CPPLINT_BIN}" )

#
# "make lint-knowhere" targets
#
add_custom_target( lint-knowhere
    ${PYTHON_EXECUTABLE}    ${BUILD_SUPPORT_DIR}/run_cpplint.py
                            --cpplint_binary    ${CPPLINT_BIN}
                            --exclude_globs     ${LINT_EXCLUSIONS_FILE}
                            --source_dir        ${CMAKE_CURRENT_SOURCE_DIR}
                            ${KNOWHERE_LINT_QUIET}
                            )

#
# "make clang-format" and "make check-clang-format" targets
#
if ( ${CLANG_FORMAT_FOUND} )
    # runs clang format and updates files in place.
    add_custom_target( clang-format-knowhere
        ${PYTHON_EXECUTABLE}    ${BUILD_SUPPORT_DIR}/run_clang_format.py
                                --clang_format_binary   ${CLANG_FORMAT_BIN}
                                --exclude_globs         ${LINT_EXCLUSIONS_FILE}
                                --source_dir            ${CMAKE_CURRENT_SOURCE_DIR}/src
                                --fix
                                ${KNOWHERE_LINT_QUIET} )

    # runs clang format and exits with a non-zero exit code if any files need to be reformatted
    add_custom_target( check-clang-format-knowhere
        ${PYTHON_EXECUTABLE}    ${BUILD_SUPPORT_DIR}/run_clang_format.py
                                --clang_format_binary   ${CLANG_FORMAT_BIN}
                                --exclude_globs         ${LINT_EXCLUSIONS_FILE}
                                --source_dir            ${CMAKE_CURRENT_SOURCE_DIR}/src
                                ${KNOWHERE_LINT_QUIET} )
endif ()

#
# "make clang-tidy" and "make check-clang-tidy" targets
#
if ( ${CLANG_TIDY_FOUND} )
    # runs clang-tidy and attempts to fix any warning automatically
    add_custom_target( clang-tidy-knowhere
        ${PYTHON_EXECUTABLE}    ${BUILD_SUPPORT_DIR}/run_clang_tidy.py
                                --clang_tidy_binary ${CLANG_TIDY_BIN}
                                --exclude_globs     ${LINT_EXCLUSIONS_FILE}
                                --compile_commands  ${CMAKE_BINARY_DIR}/compile_commands.json
                                --source_dir        ${CMAKE_CURRENT_SOURCE_DIR}/src
                                --fix
                                ${KNOWHERE_LINT_QUIET} )

    # runs clang-tidy and exits with a non-zero exit code if any errors are found.
    add_custom_target( check-clang-tidy-knowhere
        ${PYTHON_EXECUTABLE}    ${BUILD_SUPPORT_DIR}/run_clang_tidy.py
                                --clang_tidy_binary ${CLANG_TIDY_BIN}
                                --exclude_globs     ${LINT_EXCLUSIONS_FILE}
                                --ignore_checks     ${IGNORE_CHECKS_FILE}
                                --compile_commands  ${CMAKE_BINARY_DIR}/compile_commands.json
                                --source_dir        ${CMAKE_CURRENT_SOURCE_DIR}/src
                                ${KNOWHERE_LINT_QUIET} )
endif ()

#
# Validate and print out Milvus configuration options
#
config_summary()


#***************************** Source files ****************************
add_subdirectory( knowhere/knowhere/utils )
add_subdirectory( thirdparty )
add_subdirectory( knowhere )

if ( KNOWHERE_BUILD_TESTS )
    if ( BUILD_COVERAGE STREQUAL "ON" )
        append_flags( CMAKE_CXX_FLAGS
                      FLAGS
                           "-fprofile-arcs"
                           "-ftest-coverage"
                           )
    endif ()
    append_flags( CMAKE_CXX_FLAGS FLAGS "-DELPP_DISABLE_LOGS")

    add_subdirectory( unittest )
endif ()
