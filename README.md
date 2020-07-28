# rocFFT

rocFFT is a software library for computing Fast Fourier Transforms
(FFT) written in HIP. It is part of AMD's software ecosystem based on
[ROCm](https://github.com/RadeonOpenCompute). In addition to AMD GPU
devices, the library can also be compiled with the CUDA compiler using
HIP tools for running on Nvidia GPU devices.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package
servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories)
or by clicking the github releases tab and downloading the source,
which could be more recent than the pre-build packages.  Release notes
are available for each release on the releases tab.

* `sudo apt update && sudo apt install rocfft`

## Building from source

rocFFT is compiled with hipcc and uses cmake.  To compile the library one calls,
for example, the following commands:
```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc .. 
```

A static library can be compiled by using the option `-DBUILD_SHARED_LIBS=off`

The hip-clang compiler (https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang), one must specify 
`-DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang` to cmake.

One can use nvcc as a backend compiler by passing the option `-DUSE_CUDA=yes`
and setting `HIP_PLATFORM=nvcc` in your environment.


rocfft-rider is a client which will run general transforms and is
useful for performance analysis.  Compilation is enabled via the
`-DBUILD_CLIENTS_RIDER=on` cmake option.  rocfft-rider uses boost
program options.

rocfft-test runs functionality tests and uses FFTW, Google test, and
boost program options.  Compilation is enabled by calling cmake with the `-DBUILD_CLIENTS_TESTS=on` option.

To install the clients depencencies on Ubuntu, run
`sudo apt install libgtest-dev libfftw3-dev libboost-program-options-dev`.
We use version 1.10 of gtest.

The file `install.sh` is a bash script that is a wrapper for the cmake
script, which also install dependencies on certain Linux
distributions.  The preferred method for compiling rocFFT is to call
cmake directly.

## Library and API Documentation

Please refer to the [Library
documentation](http://rocfft.readthedocs.io/) for current
documentation.

## Examples

Examples may be found in the clients/samples subdirectory.
