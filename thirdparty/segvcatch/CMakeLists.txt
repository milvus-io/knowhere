project(segvcatch)

cmake_minimum_required(VERSION 2.6)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -fnon-call-exceptions")
endif()

set(TARGET segvcatch)

set(SOURCES
  segvcatch.cpp 
  segvcatch.h
  i386-signal.h
  x86_64-signal.h
)

add_library(${TARGET} STATIC ${SOURCES})