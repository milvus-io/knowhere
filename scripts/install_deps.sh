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


if [[ "${MACHINE}" == "Linux" ]]; then
    if [[ -x "$(command -v apt)" ]]; then
        # for Ubuntu 18.04
        sudo apt install -y g++ gcc make ccache libssl-dev zlib1g-dev libboost-regex-dev \
            libboost-program-options-dev libboost-system-dev libboost-filesystem-dev \
            libboost-serialization-dev python3-dev libboost-python-dev libcurl4-openssl-dev \
            gfortran libtbb-dev
        # Pre-installation of openblas can save about 15 minutes of openblas building time.
        # But the apt-installed openblas version is 0.2.20, while the latest openblas version is 0.3.19.
        # So we only pre-install openblas in Unittest, and compile openblas-0.3.19 when release.
        if [[ "${INSTALL_OPENBLAS}" == "true" ]]; then
          sudo apt install -y libopenblas-dev
        fi
    elif [[ -x "$(command -v yum)" ]]; then
        # for CentOS 7
        sudo yum install -y epel-release centos-release-scl-rh wget && \
        sudo yum install -y git make automake ccache openssl-devel zlib-devel \
            libcurl-devel python3-devel \
            devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran \
            llvm-toolset-7.0-clang llvm-toolset-7.0-clang-tools-extra 

        echo "source scl_source enable devtoolset-7" | sudo tee -a /etc/profile.d/devtoolset-7.sh
        echo "source scl_source enable llvm-toolset-7.0" | sudo tee -a /etc/profile.d/llvm-toolset-7.sh
        echo "export CLANG_TOOLS_PATH=/opt/rh/llvm-toolset-7.0/root/usr/bin" | sudo tee -a /etc/profile.d/llvm-toolset-7.sh
        source "/etc/profile.d/llvm-toolset-7.sh"
        #CMake 3.18 or higher is required
        wget -c https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.tar.gz && \
        tar -zxvf cmake-3.22.2-linux-x86_64.tar.gz && \
        sudo ln -sf $(pwd)/cmake-3.22.2-linux-x86_64/bin/cmake /usr/bin/cmake

        # Install tbb
        # git clone https://github.com/wjakob/tbb.git && \
        # cd tbb/build && \
        # cmake .. && make -j && \
        # sudo make install && \
        # cd ../../ && rm -rf tbb/

        # # Install boost
        # wget -q https://boostorg.jfrog.io/artifactory/main/release/1.65.1/source/boost_1_65_1.tar.gz && \
        #     tar zxf boost_1_65_1.tar.gz && cd boost_1_65_1 && \
        #     ./bootstrap.sh --prefix=/usr/local --with-toolset=gcc --without-libraries=python && \
        #     sudo ./b2 -j2 --prefix=/usr/local --without-python toolset=gcc install && \
        #     cd ../ && rm -rf ./boost_1_65_1*

    fi
fi

if [[ "${MACHINE}" == "Mac"  ]]; then
    brew install boost libomp llvm ninja tbb openblas
fi
