include(ExternalProject)

set(MAKE_COMMAND "make")
set(CPU_FEATURES_DIR ${CMAKE_BINARY_DIR}/cpu_features)
set(CPU_FEATURES_LIB ${CMAKE_BINARY_DIR}/cpu_features/lib/libcpu_features.a)

ExternalProject_Add(
  cpu_features
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/cpu_features
  BINARY_DIR ${CPU_FEATURES_DIR}
  CMAKE_GENERATOR "Unix Makefiles"
  PREFIX ${CPU_FEATURES_DIR}
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DBUILD_TESTING=OFF
             -DCMAKE_INSTALL_PREFIX=${CPU_FEATURES_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_COMMAND ${MAKE_COMMAND}
  BUILD_BYPRODUCTS ${CPU_FEATURES_LIB})

add_library(libcpu_features STATIC IMPORTED GLOBAL)
add_dependencies(libcpu_features cpu_features)
set_target_properties(
  libcpu_features
  PROPERTIES IMPORTED_LOCATION ${CPU_FEATURES_LIB}
             INTERFACE_INCLUDE_DIRECTORIES
             ${CMAKE_SOURCE_DIR}/thirdparty/cpu_features/include)
