#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

ROOT="$(dirname "$(dirname "$SCRIPTPATH")")"
HDF5_DIR=$ROOT"/hdf5-hdf5-1_13_2"
wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_2.tar.gz
tar xvfz hdf5-1_13_2.tar.gz
rm hdf5-1_13_2.tar.gz
cd $HDF5_DIR
./configure --prefix=/usr/local/hdf5 --enable-fortran
make -j8
make install
cd $ROOT
rm -r hdf5-hdf5-1_13_2

./build.sh  -u -t Release -b

SIFT_FILE=$ROOT"/output/unittest/sift-128-euclidean.hdf5"

wget -P $SIFT_FILE http://ann-benchmarks.com/sift-128-euclidean.hdf5

cp $ROOT"/unittest/benchmark/ref_log/Makefile" $ROOT"/output/unittest/"
