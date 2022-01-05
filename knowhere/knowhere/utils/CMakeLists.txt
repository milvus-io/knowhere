# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(UTILS_SRC
        distances_simd.cpp
        FaissHookFvec.cpp
        )

set(UTILS_AVX_SRC
        distances_simd_avx.cpp
        )

set(UTILS_AVX512_SRC
        distances_simd_avx512.cpp
        )

add_library(utils_avx OBJECT
        ${UTILS_AVX_SRC}
        )
add_library(utils_avx512 OBJECT
        ${UTILS_AVX512_SRC}
        )

target_compile_options(utils_avx PUBLIC "-mf16c;-mavx2")
target_compile_options(utils_avx512 PUBLIC "-mf16c;-mavx512f;-mavx512dq;-mavx512bw")

add_library(knowhere_utils STATIC
        ${UTILS_SRC}
        $<TARGET_OBJECTS:utils_avx>
        $<TARGET_OBJECTS:utils_avx512>
        )

target_compile_options(knowhere_utils PUBLIC "-msse4.2")
target_link_libraries(knowhere_utils PUBLIC openblas)

target_include_directories(knowhere_utils PUBLIC ${INDEX_SOURCE_DIR}/knowhere/knowhere/utils)
