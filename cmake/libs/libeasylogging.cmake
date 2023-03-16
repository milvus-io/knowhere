add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP)
add_definitions(-DELPP_THREAD_SAFE)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
endif()

include_directories(thirdparty/easyloggingpp/src)
add_library(easyloggingpp STATIC thirdparty/easyloggingpp/src/easylogging++.cc)
