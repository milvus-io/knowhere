knowhere_file_glob(GLOB_RECURSE KNOWHERE_UT_SRCS *.cc)
add_executable(knowhere_tests ${KNOWHERE_UT_SRCS})
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.1.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)
set_target_properties(knowhere_tests PROPERTIES
  CXX_STANDARD 17
)

if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()

target_link_libraries(knowhere_tests PRIVATE Catch2::Catch2WithMain knowhere)
