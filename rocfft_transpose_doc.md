# rocFFT Transpose API Documentation

## Summary

This file serves the purpose of documenting the interface, implementation and limitation of the `rocfft_transpose` as of 07/05/2016.
This file also describes the state of samples and test cases for `rocfft_transpose` as of 07/05/2016.

## Interface

The API for `rocfft_transpose` are declared in `rocfft_transpose.h` under `rocFFT/src/include/`. Only three functions can be called directly by the users.

``````
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision,
                                                                 rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,// inplace or outplace
                                                                 size_t dimensions, // dimension of input and output matrix
                                                                 const size_t *lengths, // sizes for each dimension of input matrix
                                                                 const size_t *in_stride, // input matrix stride
                                                                 const size_t *out_stride, // output matrix stide
                                                                 const size_t in_dist, // input distance between batches
                                                                 const size_t out_dist, // output distance between batches
                                                                 size_t number_of_transforms, // number of batches
                                                                 const rocfft_transpose_description *description );
```

creates a plan stored in the memory pointed by the first argument.
* `rocfft_transpose_precision` is a `enum` type that supports the setting of `single` and `double` precision.
* `rocfft_transpose_array_type` is a `enum` type that supports the setting of transpose type such as `real to real` and `complex interleaved to complex planar`
* `rocfft_transpose_placement` is a `enum` type that supports the setting of `inplace` and `notinplace` transpose
* `dimensions` is a `size_t` type that sets the number of dimensions that input matrix and output matrix have
* `lengths` is a pointer to `size_t` that states the sizes of input matrix for each dimension. The array it points to should have size `dimensions`
* `in_stride` is a pointer to `size_t` that states the strides of input matrix for each dimension. The array it points to should be of size `dimensions`
* `out_stride` is a pointer to `size_t` that states the strides of output matrix for each dimension. The array it points to should be of size `dimensions`
* `in_dist` is a `size_t` type states the distance between batches of the input matrix
* `out_dist` is a `size_t` type states the distance between batches of the output matrix
* `number_of_transforms` is a `size_t` type states the number of batches for input and output matrices
* return parameter `rocfft_transpose_status` is a `enum` type that states whether the plan creation was successful

```c
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             void **in_buffer,
                                                             void **out_buffer,
                                                             rocfft_transpose_execution_info info );
```

* `rocfft_transpose_plan` is a plan type created by `rocfft_transpose_status rocfft_transpose_plan_create`
* `in_buffer` is a pointer to pointer of void type. For `real` and `complex interleaved` data type, the size of the pointer array is 1. For `complex planar` data type, the size of the pointer array is 2.
* `out_buffer` is a pointer to pointer of void type. For `real` and `complex interleaved` data type, the size of the pointer array is 1. For `complex planar` data type, the size of the pointer array is 2.
* return parameter `rocfft_transpose_status` is a `enum` type that states whether the plan execution was successful

```c
DLL_PUBLIC rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan );
```
destroys the plan.
* return parameter `rocfft_transpose_status` is a `enum` type that states whether the plan destruction was successful

## Implementation and Limitation
The implementation for `rocfft_transpose` are defined in `rocfft_transpose.cpp` under `rocFFT/src/library/`. The kernels are defined in `.cu` files in `rocFFT/src/library/internal/include/`.

A subset of transpose has been implemented. In general,
* the inplace transpose is not implemented.
* the outplace transpose is implemented if both input and output matrix are packed
* the packed outplace transpose support:
  * real to real
  * complex interleaved to complex interleaved
  * complex planar to complex planar
  * complex interleaved to complex planar
  * complex planar to complex interleaved
* transpose is implemented if only the first dimension is padded
* only two dimensional transpose is supported

Kernels are written with HIP language. `rocFFT/src/library/internal_include/rocfft_transpose_kernel.h` declares the templated kernels. In general, there are five template parameters to each kernel:
`template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>`
, where `T` states the data type (`float` or `double`, sometimes `T2` is also needed for `float2` and `double2`), `micro_tile_col_size` and `micro_tile_row_size` are the sizes of micro tile for different dimensions. `wg_col_size` and `wg_row_size` depicts the work-group size.
