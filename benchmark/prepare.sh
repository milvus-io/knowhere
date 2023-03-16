#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

ROOT="$(dirname "$(dirname "$SCRIPTPATH")")"
SIFT_FILE=$ROOT"/output/unittest/sift-128-euclidean.hdf5"

wget -P $SIFT_FILE http://ann-benchmarks.com/sift-128-euclidean.hdf5

cp $ROOT"/unittest/benchmark/ref_log/Makefile" $ROOT"/output/unittest/"
