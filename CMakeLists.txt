cmake_minimum_required(VERSION 3.2)
project(knowhere CXX C)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/utils/utils.cmake)


knowhere_option(USE_CUDA "Build with CUDA" OFF)

if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75;70;61;60)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()


include(cmake/utils/compile_flags.cmake)
include(cmake/utils/platform_check.cmake)
include(cmake/libs/libfaiss.cmake)
include(cmake/libs/libannoy.cmake)
include(cmake/libs/libhnsw.cmake)


add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP=1)
add_definitions(-DELPP_THREAD_SAFE=1)
add_definitions(-DINFO=1)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(thirdparty/faiss)
include_directories(thirdparty/bitset)
include_directories(thirdparty)
find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

knowhere_file_glob(GLOB_RECURSE KNOWHERE_SRCS src/common/*.cc
    src/index/*.cc src/io/*.cc)

if(NOT USE_CUDA)
    knowhere_file_glob(GLOB_RECURSE KNOWHERE_GPU_SRCS src/index/gpu_ivf/*.cc)
    list(REMOVE_ITEM KNOWHERE_SRCS ${KNOWHERE_GPU_SRCS})
endif()

include_directories(src)
include_directories(include)
include_directories(.)

add_library(knowhere SHARED ${KNOWHERE_SRCS})
add_dependencies(knowhere faiss)
target_link_libraries(knowhere PRIVATE faiss)

add_subdirectory(tests)
