include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=gnu++17 COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_FLAGS "-std=gnu++17 ${CMAKE_CXX_FLAGS}")

if(NOT COMPILER_SUPPORTS_CXX17)
  message(
    FATAL_ERROR
      "C++17 needed. Therefore a gcc compiler with a version higher than 4.3 is needed."
  )
endif()

if(WITH_ASAN)
  set(CMAKE_CXX_FLAGS
      "-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address ${CMAKE_CXX_FLAGS}"
  )
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if(USE_CUDA)
    set(CMAKE_CUDA_FLAGS
        "-O0 -g -Xcompiler=-w -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  endif()
else()
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
  if(USE_CUDA)
    set(CMAKE_CUDA_FLAGS
        "-O2 -Xcompiler=-w -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  endif()
endif()
