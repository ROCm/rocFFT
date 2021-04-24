#!/bin/bash

# A test script for the rocFFT benchmarks with dloaded libraries.

export devicenum=0

if [ "$1" != "" ]; then
    devicenum = $1
fi


# Compile two libs with 

mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=hipcc  -DBUILD_CLIENTS_ALL=on -DSINGLELIB=on -DAMDGPU_TARGETS= ../../.. && make -j$(nproc)
popd

mkdir build1
pushd build1
cmake -DCMAKE_CXX_COMPILER=hipcc  -DBUILD_CLIENTS_ALL=on -DSINGLELIB=on -DAMDGPU_TARGETS= ../../.. && make -j$(nproc)
popd


# Test the dload version:
./alltime.py -b build/clients/staging/dyna-rocfft-rider -i build/library/src/librocfft.so -i build1/library/src/librocfft.so -o dtest/dir0 -o dtest/dir1 -w dtest/doc -D 1 -s -d ${devicenum}
# produces dtest/doc/figs.pdf


# Test the normal version:
./alltime.py -i build/clients/staging/rocfft-rider -i build/clients/staging/rocfft-rider -o otest/dir0 -o otest/dir1 -w otest/doc -D 1 -s -d ${devicenum}
# produces otest/doc/figs.pdf
