// Copyright (C) 2022 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "accuracy_test.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

__host__ __device__ float multiply_by_scalar(float a, double b)
{
    return a * b;
}
__host__ __device__ float2 multiply_by_scalar(float2 a, double b)
{
    return hipCmulf(a, make_float2(b, 0.0));
}
__host__ __device__ double multiply_by_scalar(double a, double b)
{
    return a * b;
}
__host__ __device__ double2 multiply_by_scalar(double2 a, double b)
{
    return hipCmul(a, make_double2(b, 0.0));
}

__host__ __device__ float add_scalar(float a, double b)
{
    return a + b;
}
__host__ __device__ float2 add_scalar(float2 a, double b)
{
    return hipCaddf(a, make_float2(b, 0.0));
}
__host__ __device__ double add_scalar(double a, double b)
{
    return a + b;
}
__host__ __device__ double2 add_scalar(double2 a, double b)
{
    return hipCadd(a, make_double2(b, 0.0));
}

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata load_callback(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    if(input == testdata->base)
        return multiply_by_scalar(input[offset], testdata->scalar);
    // wrong base address passed, return something obviously wrong
    else
    {
        // wrong base address passed, return something obviously wrong
        return input[0];
    }
}

__device__ auto load_callback_dev_float   = load_callback<float>;
__device__ auto load_callback_dev_float2  = load_callback<float2>;
__device__ auto load_callback_dev_double  = load_callback<double>;
__device__ auto load_callback_dev_double2 = load_callback<double2>;

void* get_load_callback_host(fft_array_type itype, fft_precision precision)
{
    void* load_callback_host = nullptr;
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_float2), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_dev_double2),
                                          sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_float), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_double), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return load_callback_host;
    }
}

template <typename Tdata>
__host__ __device__ static void
    store_callback(Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    if(output == testdata->base)
    {
        output[offset] = add_scalar(element, testdata->scalar);
    }
    // otherwise, wrong base address passed, just don't write
}
__device__ auto store_callback_dev_float   = store_callback<float>;
__device__ auto store_callback_dev_float2  = store_callback<float2>;
__device__ auto store_callback_dev_double  = store_callback<double>;
__device__ auto store_callback_dev_double2 = store_callback<double2>;

void* get_store_callback_host(fft_array_type otype, fft_precision precision)
{
    void* store_callback_host = nullptr;
    switch(otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_float2),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_double2),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_float),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_double),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return store_callback_host;
    }
}

// Apply store callback if necessary
void apply_store_callback(const fft_params& params, fftw_data_t& output)
{
    if(!params.run_callbacks)
        return;

    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.store_cb_scalar;
    cbdata.base   = output.front().data();

    switch(params.otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float2*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = 2 * sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double2*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

// apply load callback if necessary
void apply_load_callback(const fft_params& params, fftw_data_t& input)
{
    if(!params.run_callbacks)
        return;
    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.load_cb_scalar;
    cbdata.base   = input.front().data();

    switch(params.itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float2*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = 2 * sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double2*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}
