knowhere_file_glob(GLOB_RECURSE KNOWHERE_UT_SRCS *.cc)
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.1.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)
if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
add_executable(knowhere_tests ${KNOWHERE_UT_SRCS})
target_link_libraries(knowhere_tests PRIVATE Catch2::Catch2WithMain knowhere)
