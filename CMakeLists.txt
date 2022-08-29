cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/utils/utils.cmake)
include(cmake/utils/compiler_check.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)
include(cmake/libs/libannoy.cmake)
include(cmake/libs/libhnsw.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP=1)
add_definitions(-DELPP_THREAD_SAFE=1)
add_definitions(-DINFO=1)

find_package(BLAS REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(thirdparty/faiss)
include_directories(thirdparty/bitset)
include_directories(thirdparty)
find_package(OpenMP REQUIRED)


if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Boost REQUIRED)
knowhere_file_glob(GLOB_RECURSE KNOWHERE_SRCS src/common/*.cc
    src/index/*.cc src/io/*.cc)
include_directories(src)
include_directories(include)
include_directories(.)

add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere faiss)
target_link_libraries(knowhere PRIVATE faiss)

add_subdirectory(tests)
