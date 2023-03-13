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

## Dependencies

rocFFT requires python3 libraries to be present at runtime.  These are
typically present by default in most Linux distributions.

## Building from source

rocFFT is compiled with hipcc and uses cmake.  There are a number of options
that can be provided to cmake to customize the build, but the following
commands will build a shared library for supported AMD GPUs:

```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc .. 
make -j
```

A static library can be compiled by using the option `-DBUILD_SHARED_LIBS=off`

To use the [hip-clang compiler][3], one must specify
`-DUSE_HIP_CLANG=ON -DHIP_COMPILER=clang` to cmake.  rocFFT enables
use of indirect function calls by default and requires ROCm 4.3 or
higher to build successfully.  `-DROCFFT_CALLBACKS_ENABLED=off`
may be specified to cmake to disable those calls on older ROCm
compilers, though callbacks will not work correctly in this configuration.

There are several clients included with rocFFT:
1. rocfft-rider runs general transforms and is useful for performance analysis;
2. rocfft-test runs various regression tests; and
3. various small samples are included.

Clients are not built by default.  To build them:

| Client          | CMake option                  | Dependencies                             |
|-----------------|-------------------------------|------------------------------------------|
| rocfft-rider    | `-DBUILD_CLIENTS_RIDER=on`    | Boost program options                    |
| rocfft-test     | `-DBUILD_CLIENTS_TESTS=on`    | Boost program options, FFTW, Google Test |
| samples         | `-DBUILD_CLIENTS_SAMPLES=on`  | Boost program options, FFTW              |

To build all of the above clients, use `-DBUILD_CLIENTS=on`. The build process will 
download and build Google Test and FFTW if they are not installed.

Clients may be built separately from the main library. For example, one may build
all the clients with an existing rocFFT library by invoking cmake from within the 
rocFFT-src/clients folder: 

```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/path/to/rocFFT-lib ..
make -j
```

To install the client dependencies on Ubuntu, run:

```
sudo apt install libgtest-dev libfftw3-dev libboost-program-options-dev`
```

We use version 1.11 of Google Test (gtest).

`install.sh` is a bash script that will install dependencies on certain Linux
distributions, such as Ubuntu, CentOS, RHEL, Fedora, and SLES and invoke cmake.
However, the preferred method for compiling rocFFT is to call cmake directly.

## Library and API Documentation

Please refer to the [library documentation][4] for current documentation.

### How to build documentation

Please follow the steps below to build the documentation.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Examples

Examples may be found in the [clients/samples][5] subdirectory.

[1]: https://github.com/RadeonOpenCompute
[2]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
[3]: https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang
[4]: https://rocfft.readthedocs.io/
[5]: clients/samples

## Contribution Rules

### Source code formatting

* C++ source code must be formatted with clang-format with .clang-format

* Python source code must be formatted with yapf --style pep8
