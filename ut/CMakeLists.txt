knowhere_file_glob(GLOB_RECURSE KNOWHERE_UT_SRCS *.cc)
add_executable(knowhere_tests ${KNOWHERE_UT_SRCS})
target_link_libraries(knowhere_tests PRIVATE knowhere)
target_include_directories(knowhere_tests PRIVATE ../thirdparty/catch2/single_include)
