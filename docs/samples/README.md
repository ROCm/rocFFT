# Samples to demo using rocfft

## `complex_1d`

You may need to add the directories for hipcc and rocFFT to your
`CMAKE_PREFIX_PATH`, and ensure that `hipcc` is in your `PATH`.

``` bash
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=hipcc ..
$ make
```
