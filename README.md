This document will help you to build the Knowhere repository from source code and to run unit tests. Please [file an issue](https://github.com/milvus-io/knowhere/issues/new) if there's a problem.

## Introduction

Knowhere is written in C++. It is an independent project that act as Milvus's internal core.

## Building Knowhere Within Milvus

If you wish to only use Knowhere within Milvus without changing any of the Knowhere source code, we suggest that you move to the [Milvus main project](https://github.com/milvus-io/milvus) and build Milvus directly, where Knowhere is then built implicitly during Milvus build.

## System Requirements

All Linux distributions are available for Knowhere development. However, a majority of our contributor worked with Ubuntu or CentOS systems, with a small portion of Mac (both x86_64 and Apple Silicon) contributors. If you would like Knowhere to build and run on other distributions, you are more than welcome to file an issue and contribute!

Here's a list of verified OS types where Knowhere can successfully build and run:

- Ubuntu 18.04
- CentOS 7
- MacOS (x86_64)
- MacOS (Apple Silicon)
- MinGW64

## Building Knowhere From Source Code

#### Install Dependencies

```bash
$ ./scripts/install_deps.sh
```

#### Build From Source Code

```bash
$ ./build.sh -t Release
```

#### Running Unit Tests

```bash
$ ./build.sh -t Release -u && output/unittest/test_knowhere
```

#### Clean up

```bash
$ ./build.sh -r
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

