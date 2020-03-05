#!/bin/bash

# A sample script for running the rocFFT benchmarks with dloaded libraries.


pushd ../..


# Compile two libs with 
mkdir build -p
pushd build
cmake -DCMAKE_CXX_COMPILER=hcc -DSINGLELIB=on -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS_SAMPLES=on -DBUILD_CLIENTS_RIDER=on  .. && make -j$(nproc)
popd

mkdir build1 -p
pushd build1
cmake -DCMAKE_CXX_COMPILER=hcc -DSINGLELIB=on -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS_SAMPLES=on -DBUILD_CLIENTS_RIDER=on  .. && make -j$(nproc) 
popd


popd


# Test the dload version:
./alltime.py -b ../../build/clients/staging/ -i ../../build/library/src/ -i ../../build1/library/src/ -o dtest/dir0 -o dtest/dir1 -w dtest/doc -D 1 -s
# produces dtest/doc/figs.pdf


# Test the normal version:
./alltime.py -i ../../build/clients/staging/ -i  ../../build/clients/staging/ -o otest/dir0 -o otest/dir1 -w otest/doc -D 1 -s
# produces otest/doc/figs.pdf
