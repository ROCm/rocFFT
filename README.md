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

rocFFT is compiled with hipcc and uses cmake.  There are a number of options
that can be provided to cmake to customize the build, but the following
commands will build a shared library for supported AMD GPUs:

```Bash
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc .. 
make -j
```

A static library can be compiled by using the option `-DBUILD_SHARED_LIBS=off`

rocFFT enables use of indirect function calls by default and requires
ROCm 4.3 or higher to build successfully.
`-DROCFFT_CALLBACKS_ENABLED=off` may be specified to cmake to disable
those calls on older ROCm compilers, though callbacks will not work
correctly in this configuration.

There are several clients included with rocFFT:

1. rocfft-bench runs general transforms and is useful for performance analysis;
2. rocfft-test runs various regression tests; and
3. various small samples are included.

Clients are not built by default.  To build them:

| Client          | CMake option                  | Dependencies                             |
|-----------------|-------------------------------|------------------------------------------|
| rocfft-bench    | `-DBUILD_CLIENTS_BENCH=on`    | Boost program options                    |
| rocfft-test     | `-DBUILD_CLIENTS_TESTS=on`    | Boost program options, FFTW, Google Test |
| samples         | `-DBUILD_CLIENTS_SAMPLES=on`  | Boost program options, FFTW              |

To build all of the above clients, use `-DBUILD_CLIENTS=on`. The build process will 
download and build Google Test and FFTW if they are not installed.

Clients may be built separately from the main library. For example, one may build
all the clients with an existing rocFFT library by invoking cmake from within the 
rocFFT-src/clients folder: 

```Bash
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/path/to/rocFFT-lib ..
make -j
```

To install the client dependencies on Ubuntu, run:

```
sudo apt install libgtest-dev libfftw3-dev libboost-program-options-dev
```

We use version 1.11 of Google Test (gtest).

## Library and API Documentation

Please refer to the [library documentation][3] for current documentation.

### How to build documentation

Please follow the steps below to build the documentation.

```Bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Examples

A summary of the latest functionality and workflow to compute an FFT with rocFFT is available [on the ROCm documentation portal][3].

Further examples may be found in the [clients/samples][4] subdirectory.

[1]: https://github.com/RadeonOpenCompute
[2]: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html
[3]: https://rocm.docs.amd.com/projects/rocFFT/en/latest/
[4]: clients/samples

## Contribution Rules

### Source code formatting

* C++ source code must be formatted with clang-format with .clang-format

* Python source code must be formatted with yapf --style pep8
