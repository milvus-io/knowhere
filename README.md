<p>
    <img src="static/knowhere-logo.png" alt="Knowhere Logo"/>
</p>

This document will help you to build the Knowhere repository from source code and to run unit tests. Please [file an issue](https://github.com/milvus-io/knowhere/issues/new) if there's a problem.

## Introduction

Knowhere is written in C++. It is an independent project that act as Milvus's internal core.

## Building Knowhere Within Milvus

If you wish to only use Knowhere within Milvus without changing any of the Knowhere source code, we suggest that you move to the [Milvus main project](https://github.com/milvus-io/milvus) and build Milvus directly, where Knowhere is then built implicitly during Milvus build.

## System Requirements

All Linux distributions are available for Knowhere development. However, a majority of our contributor worked with Ubuntu or CentOS systems, with a small portion of Mac (both x86_64 and Apple Silicon) contributors. If you would like Knowhere to build and run on other distributions, you are more than welcome to file an issue and contribute!

Here's a list of verified OS types where Knowhere can successfully build and run:

- Ubuntu 20.04 x86_64
- Ubuntu 20.04 Aarch64
- MacOS (x86_64)
- MacOS (Apple Silicon)

## Building Knowhere From Source Code

#### Install Dependencies

```bash
$ sudo apt install build-essential libopenblas-dev ninja-build libaio-dev libboost-program-options-dev
```

#### Build From Source Code

##### Option 1 (Recommandded)

```bash
# Install Conan
# Conan is a dependency manager for C/C++. see https://conan.io/
$ pip3 install conan==1.58.0

$ mkdir build && cd build
#DEBUG CPU
$ conan install .. --build=missing -s build_type=Debug -o with_ut=True
#RELEASE CPU
$ conan install .. --build=missing -o with_ut=True
#DEBUG GPU
$ conan install .. --build=missing -s build_type=Debug -o with_ut=True -o with_cuda=True
#RELEASE GPU
$ conan install .. --build=missing -o with_ut=True -o with_cuda=True
#ADD -DWITH_DISKANN=ON TO BUILD DISKANN INDEX
$ conan install .. --build=missing -o with_ut=True -o with_diskann=True
#verbose compile
$ conan build ..
```

##### Option 2

```bash
#fetch submodule
$ git submodule update --recursive --init

$ mkdir build && cd build
#DEBUG CPU
$ cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_UT=ON -G Ninja
#RELEASE CPU
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_UT=ON -G Ninja
#DEBUG GPU
$ cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DWITH_UT=ON -G Ninja
#COMPILE with new GPUs, define your CMAKE_CUDA_ARCHITECTURES
$ cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DWITH_UT=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -G Ninja
#RELEASE GPU
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWITH_UT=ON -G Ninja
#ADD -DWITH_DISKANN=ON TO BUILD DISKANN INDEX
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_UT=ON -DWITH_DISKANN=ON -G Ninja
#verbose compile
$ninja -v
```

#### Running Unit Tests

```bash
# in build directories
$ ./tests/ut/knowhere_tests
```

#### Clean up

```bash
$ git clean -fxd
```

## GEN PYTHON WHEEL

install dependency:

```
sudo apt install swig python3-dev
```

after build knowhere:

```bash
cd python
python3 setup.py bdist_wheel
```

install knowhere wheel:

```bash
pip3 install dist/knowhere-1.0.0-cp38-cp38-linux_x86_64.whl
```

clean

```bash
cd python
rm -rf build
rm -rf dist
rm -rf knowhere.egg-info
rm knowhere/knowhere_wrap.cpp
rm knowhere/swigknowhere.py
```
