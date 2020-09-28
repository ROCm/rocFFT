# rocFFT

rocFFT is a software library for computing Fast Fourier Transforms
(FFT) written in HIP. It is part of AMD's software ecosystem based on
[ROCm][1]. In addition to AMD GPU devices, the library can also be
compiled with the CUDA compiler using HIP tools for running on Nvidia
GPU devices.

## Installing pre-built packages

Download pre-built packages either from [ROCm's package servers][2]
or by clicking the github releases tab and downloading the source,
which could be more recent than the pre-build packages.  Release notes
are available for each release on the releases tab.

* `sudo apt update && sudo apt install rocfft`

## Building from source

rocFFT is compiled with hipcc and uses cmake.  A number of options can
be provided to cmake, but to the library may be compiled with the
following commands:

```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc .. 
make -j
```

A static library can be compiled by using the option `-DBUILD_SHARED_LIBS=off`

To use the [hip-clang compiler][3], one must specify
`-DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang` to cmake.

One can use nvcc as a backend compiler by passing the option `-DUSE_CUDA=yes`
and setting `HIP_PLATFORM=nvcc` in your environment.


rocfft-rider is a client which will run general transforms and is
useful for performance analysis.  Compilation is enabled via the
`-DBUILD_CLIENTS_RIDER=on` cmake option.  rocfft-rider uses boost
program options.

rocfft-test runs functionality tests and uses FFTW, Google test, and
boost program options.  Compilation is enabled by calling cmake with the
`-DBUILD_CLIENTS_TESTS=on` option.

To install the clients depencencies on Ubuntu, run
`sudo apt install libgtest-dev libfftw3-dev libboost-program-options-dev`.
We use version 1.10 of gtest.

`install.sh` is a bash script that will install dependencies on certain Linux
distributions, such as Ubuntu, CentOS, RHEL, Fedora, and SLES and invoke cmake.
However, the preferred method for compiling rocFFT is to call cmake directly.

## Library and API Documentation

Please refer to the [library documentation][4] for current documentation.

## Examples

Examples may be found in the [clients/samples][5] subdirectory.


[1]: https://github.com/RadeonOpenCompute
[2]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
[3]: https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang
[4]: https://rocfft.readthedocs.io/
[5]: clients/samples
