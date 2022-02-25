#!/usr/bin/env bash

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

# Exit immediately for non zero status
set -e

UNAME="$(uname -s)"

case "${UNAME}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${UNAME}"
esac


if [[ "${MACHINE}" == "Linux"  ]]; then
    if [[ -x "$(command -v apt)" ]]; then
        # for Ubuntu 18.04
        sudo apt install -y g++ gcc make ccache libssl-dev zlib1g-dev libboost-regex-dev \
            libboost-program-options-dev libboost-system-dev libboost-filesystem-dev \
            libboost-serialization-dev python3-dev libboost-python-dev libcurl4-openssl-dev gfortran libtbb-dev
    fi
fi

if [[ "${MACHINE}" == "Mac"  ]]; then
    brew install boost libomp llvm ninja tbb
fi

if [[ "${MACHINE}" == "MinGw"  ]]; then
    pacman -Su --noconfirm --needed \
    git make tar dos2unix zip unzip patch \
    mingw-w64-x86_64-toolchain \
    mingw-w64-x86_64-make \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-boost \
    mingw-w64-x86_64-intel-tbb \
    mingw-w64-x86_64-openblas
fi
