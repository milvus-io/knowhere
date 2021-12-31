cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 11)

project(BruteForce)

#set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/13.0.0_2/bin/clang")
#set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/13.0.0_2/bin/clang++")
#set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/13.0.0_2/lib")
#set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/13.0.0_2/include")

OPTION (USE_OpenMP "Use OpenMP to enable <omp.h>" ON)

if(APPLE AND USE_OpenMP)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

if(USE_OpenMP)
    find_package(OpenMP REQUIRED)
endif(USE_OpenMP)

#if (OPENMP_FOUND)
#    include_directories("${OPENMP_INCLUDES}")
#    link_directories("${OPENMP_LIBRARIES}")
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif(OPENMP_FOUND)

add_subdirectory(include)

add_subdirectory(indexing)

add_subdirectory(distances)

include_directories(include)

include_directories(indexing)

include_directories(distances)

add_executable(BruteForce test/test_demo_sift1M.cpp)

target_link_libraries(BruteForce PUBLIC BruteForceIndexing BruteForceInclude BruteForceDistances)

target_include_directories(BruteForce PUBLIC
        "indexing"
        "include"
        "distances"
        )
