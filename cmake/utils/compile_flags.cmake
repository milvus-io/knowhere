include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++17 COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

if(NOT COMPILER_SUPPORTS_CXX17)
  message(
    FATAL_ERROR
      "C++17 needed. Therefore a gcc compiler with a version higher than 4.3 is needed."
  )
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS
      "-O0 -g -Xcompiler=-w -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if (USE_CUDA)
    set(CMAKE_CUDA_FLAGS
        "-O2  -Xcompiler=-w -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  endif()
endif()
