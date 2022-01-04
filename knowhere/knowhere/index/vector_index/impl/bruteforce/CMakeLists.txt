cmake_minimum_required(VERSION 3.10)

add_subdirectory(include)

add_subdirectory(indexing)

add_subdirectory(distances)

add_executable(BruteForce test/test_demo_sift1M.cpp)

target_link_libraries(BruteForce PUBLIC BruteForceIndexing BruteForceInclude BruteForceDistances)