#!/bin/bash

# A sample script for running the rocFFT benchmarks with dloaded libraries.


pushd ../..


# Compile two libs with 
mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=hipcc -DSINGLELIB=on -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS_SAMPLES=on -DBUILD_CLIENTS_RIDER=on  .. && make -j$(nproc)
popd

mkdir build1
pushd build1
cmake -DCMAKE_CXX_COMPILER=hipcc -DSINGLELIB=on -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS_SAMPLES=on -DBUILD_CLIENTS_RIDER=on  .. && make -j$(nproc) 
popd


popd


# Test the dload version:
./alltime.py -b ../../build/clients/staging/dyna-rocfft-rider -i ../../build/library/src/librocfft.so -i ../../build1/library/src/librocfft.so -o dtest/dir0 -o dtest/dir1 -w dtest/doc -D 1 -s
# produces dtest/doc/figs.pdf


# Test the normal version:
./alltime.py -i ../../build/clients/staging/rocfft-rider -i  ../../build/clients/staging/rocfft-rider -o otest/dir0 -o otest/dir1 -w otest/doc -D 1 -s
# produces otest/doc/figs.pdf
