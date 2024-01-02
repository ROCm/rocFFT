# rocFFT

rocFFT is a software library for computing fast Fourier transforms (FFTs) written in the HIP
programming language. It's part of AMD's software ecosystem based on
[ROCm](https://github.com/RadeonOpenCompute). The rocFFT library can be used with AMD and
NVIDIA GPUs.

## Documentation

Documentation for rocFFT is available at
[rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocFFT/en/latest/).

To build our documentation locally, use the following code:

```Bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

You can install rocFFT using pre-built packages or building from source.

* Installing pre-built packages:

    1. Download the pre-built packages from the
        [ROCm package servers](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) or use the
        GitHub releases tab to download the source (this may give you a more recent version than the
        pre-built packages).

    2. Run: `sudo apt update && sudo apt install rocfft`

* Building from source:

    rocFFT is compiled with HIPCC and uses CMake. You can specify several options to customize your
    build. The following commands build a shared library for supported AMD GPUs:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc ..
    make -j
    ```

    You can compile a static library using the `-DBUILD_SHARED_LIBS=off` option.

    With rocFFT, you can use indirect function calls by default; this requires ROCm 4.3 or higher. You can
    use `-DROCFFT_CALLBACKS_ENABLED=off` with CMake to prevent these calls on older ROCm
    compilers. Note that with this configuration, callbacks won't work correctly.

    rocFFT includes the following clients:

  * `rocfft-bench`: Runs general transforms and is useful for performance analysis
  * `rocfft-test`: Runs various regression tests
  * Various small samples

    | Client | CMake option | Dependencies |
    |:------|:-----------------|:-----------------|
    | `rocfft-bench` | `-DBUILD_CLIENTS_BENCH=on` | Boost program options |
    | `rocfft-test` | `-DBUILD_CLIENTS_TESTS=on` | Boost program options, Fastest Fourier Transform in the West (FFTW), GoogleTest |
    | samples | `-DBUILD_CLIENTS_SAMPLES=on` | Boost program options, FFTW |

    Clients are not built by default. To build them, use `-DBUILD_CLIENTS=on`. The build process
    downloads and builds GoogleTest and FFTW if they are not already installed.

    Clients can be built separately from the main library. For example, you can build all the clients with
    an existing rocFFT library by invoking CMake from within the `rocFFT-src/clients` folder:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/path/to/rocFFT-lib ..
    make -j
    ```

    To install client dependencies on Ubuntu, run:

    ```bash
    sudo apt install libgtest-dev libfftw3-dev libboost-program-options-dev
    ```

    We use version 1.11 of GoogleTest.

## Examples

A summary of the latest functionality and workflow to compute an FFT with rocFFT is available on the
[rocFFT documentation portal](https://rocm.docs.amd.com/projects/rocFFT/en/latest/).

You can find additional examples in the `clients/samples` subdirectory.

## Contribute

If you want to contribute to rocFFT, you must format your code as follows:

* C++: Format with ClangFormat (see `.clang-format`)

* Python: Format with `yapf --style pep8`
