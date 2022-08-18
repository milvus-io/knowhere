cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/utils/utils.cmake)
include(cmake/utils/compiler_check.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)

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
find_package(Boost REQUIRED)
include_directories(${OpenMP_CXX_INCLUDE_DIRS})


knowhere_file_glob(GLOB KNOWHERE_SRCS src/common/*.cc 
    src/index/flat/*.cc knowhere/helper/*.cc src/io/*.cc)
include_directories(src)
include_directories(include)
include_directories(.)

add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere faiss)
target_link_libraries(knowhere PRIVATE faiss)

add_subdirectory(tests)




