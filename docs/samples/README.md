# Samples to demo using rocfft

## `complex_1d`

You may need to add the directories for hcc and rocFFT to your
`CMAKE_PREFIX_PATH`, and ensure that `hcc` is in your `PATH`.

``` bash
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=hcc ..
$ make
```
