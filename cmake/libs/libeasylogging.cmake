add_definitions(-DAUTO_INITIALIZE_EASYLOGGINGPP)
add_definitions(-DELPP_THREAD_SAFE)
include_directories(thirdparty/easyloggingpp/src)
add_library(easyloggingpp STATIC thirdparty/easyloggingpp/src/easylogging++.cc)
