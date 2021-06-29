// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "./kernels/common.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename Tcomplex, CallbackType cbtype>
__global__ static void complex2real_kernel(const size_t           input_size,
                                           const size_t           idist1D,
                                           const size_t           odist1D,
                                           const Tcomplex*        input0,
                                           const size_t           idist,
                                           real_type_t<Tcomplex>* output0,
                                           const size_t           odist,
                                           void* __restrict__ load_cb_fn,
                                           void* __restrict__ load_cb_data,
                                           uint32_t load_cb_lds_bytes,
                                           void* __restrict__ store_cb_fn,
                                           void* __restrict__ store_cb_data)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < input_size)
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const auto input     = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        auto       outputIdx = blockIdx.y * odist1D + blockIdx.z * odist;

        // we would do real2complex at the end of a C2R transform, so
        // it would never be the first kernel to read from global
        // memory.  don't bother going through load callback to read
        // global memory.

        auto store_cb = get_store_cb<real_type_t<Tcomplex>, cbtype>(store_cb_fn);
        store_cb(output0, outputIdx + tid, input[tid].x, store_cb_data, nullptr);
    }
}

/// \brief auxiliary function
///   Convert a complex vector into a real one by only taking the real part of the complex
///   vector.  Currently only works for stride=1 cases
/// @param[in] input_size size of input buffer
/// @param[in] input_buffer data type : float2 or double2
/// @param[in] input_distance distance between consecutive batch members for input buffer
/// @param[in,output] output_buffer data type : float or double
/// @param[in] output_distance distance between consecutive batch members for output
/// buffer
/// @param[in] batch number of transforms
/// @param[in] precision data type of input buffer. rocfft_precision_single or
/// rocfft_precsion_double
ROCFFT_DEVICE_EXPORT void complex2real(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0];

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (input_size - 1) / LAUNCH_BOUNDS_R2C_C2R_KERNEL + 1;

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_size*batch);
    // hipMemcpy(tmp, input_buffer, sizeof(float2)*input_size*batch,
    //           hipMemcpyDeviceToHost);
    // for(size_t j=0; j< (data->node->length.size() == 2 ? (data->node->length[1]) :
    //                     1); j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", i, j,
    //                tmp[j*data->node->length[0]+i].x, tmp[j*data->node->length[0]+i].y);
    //     }
    // }
    // free(tmp);

    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(
            data->get_callback_type() == CallbackType::USER_LOAD_STORE
                ? HIP_KERNEL_NAME(complex2real_kernel<float2, CallbackType::USER_LOAD_STORE>)
                : HIP_KERNEL_NAME(complex2real_kernel<float2, CallbackType::NONE>),
            grid,
            threads,
            0,
            rocfft_stream,
            input_size,
            input_stride,
            output_stride,
            (float2*)input_buffer,
            input_distance,
            (float*)output_buffer,
            output_distance,
            data->callbacks.load_cb_fn,
            data->callbacks.load_cb_data,
            data->callbacks.load_cb_lds_bytes,
            data->callbacks.store_cb_fn,
            data->callbacks.store_cb_data);
    else
        hipLaunchKernelGGL(
            data->get_callback_type() == CallbackType::USER_LOAD_STORE
                ? HIP_KERNEL_NAME(complex2real_kernel<double2, CallbackType::USER_LOAD_STORE>)
                : HIP_KERNEL_NAME(complex2real_kernel<double2, CallbackType::NONE>),
            grid,
            threads,
            0,
            rocfft_stream,
            input_size,
            input_stride,
            output_stride,
            (double2*)input_buffer,
            input_distance,
            (double*)output_buffer,
            output_distance,
            data->callbacks.load_cb_fn,
            data->callbacks.load_cb_data,
            data->callbacks.load_cb_lds_bytes,
            data->callbacks.store_cb_fn,
            data->callbacks.store_cb_data);
}

template <typename T, CallbackType cbtype>
__global__ static void hermitian2complex_kernel(const size_t hermitian_size,
                                                const size_t dim_0,
                                                const size_t dim_1,
                                                const size_t dim_2,
                                                const size_t input_stride,
                                                const size_t output_stride,
                                                T*           input,
                                                const size_t input_distance,
                                                T*           output,
                                                const size_t output_distance,
                                                void* __restrict__ load_cb_fn,
                                                void* __restrict__ load_cb_data,
                                                uint32_t load_cb_lds_bytes,
                                                void* __restrict__ store_cb_fn,
                                                void* __restrict__ store_cb_data)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    size_t input_offset   = hipBlockIdx_z * input_distance;
    size_t outputs_offset = hipBlockIdx_z * output_distance; // straight copy
    size_t outputc_offset = hipBlockIdx_z * output_distance; // conjugate copy

    // straight copy indices
    size_t is0 = tid;
    size_t is1 = hipBlockIdx_y % dim_1;
    size_t is2 = hipBlockIdx_y / dim_1;

    // conjugate copy indices
    size_t ic0 = (is0 == 0) ? 0 : dim_0 - is0;
    size_t ic1 = (is1 == 0) ? 0 : dim_1 - is1;
    size_t ic2 = (is2 == 0) ? 0 : dim_2 - is2;

    input_offset += hipBlockIdx_y * input_stride + is0; // notice for 1D,
    // hipBlockIdx_y == 0 and
    // thus has no effect for
    // input_offset
    outputs_offset += (is2 * dim_1 + is1) * output_stride + is0;
    outputc_offset += (ic2 * dim_1 + ic1) * output_stride + ic0;

    auto load_cb = get_load_cb<T, cbtype>(load_cb_fn);

    // we would do hermitian2complex at the start of a C2R transform,
    // so it would never be the last kernel to write to global
    // memory.  don't bother going through the store callback to
    // write global memory.

    T* outputs = output + outputs_offset;
    T* outputc = output + outputc_offset;

    if((is0 == 0) || (is0 * 2 == dim_0)) // simply write the element to output
    {
        outputs[0] = load_cb(input, input_offset, load_cb_data, nullptr);
        return;
    }

    if(is0 < hermitian_size)
    {
        T res      = load_cb(input, input_offset, load_cb_data, nullptr);
        outputs[0] = res;
        res.y      = -res.y;
        outputc[0] = res;
    }
}

template <typename T>
__global__ static void hermitian2complex_kernel(const size_t    hermitian_size,
                                                const size_t    dim_0,
                                                const size_t    dim_1,
                                                const size_t    dim_2,
                                                const size_t    input_stride,
                                                const size_t    output_stride,
                                                real_type_t<T>* inputRe,
                                                real_type_t<T>* inputIm,
                                                const size_t    input_distance,
                                                T*              output,
                                                const size_t    output_distance)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    size_t input_offset   = hipBlockIdx_z * input_distance;
    size_t outputs_offset = hipBlockIdx_z * output_distance; // straight copy
    size_t outputc_offset = hipBlockIdx_z * output_distance; // conjugate copy

    // straight copy indices
    size_t is0 = tid;
    size_t is1 = hipBlockIdx_y % dim_1;
    size_t is2 = hipBlockIdx_y / dim_1;

    // conjugate copy indices
    size_t ic0 = (is0 == 0) ? 0 : dim_0 - is0;
    size_t ic1 = (is1 == 0) ? 0 : dim_1 - is1;
    size_t ic2 = (is2 == 0) ? 0 : dim_2 - is2;

    input_offset += hipBlockIdx_y * input_stride + is0; // notice for 1D,
    // hipBlockIdx_y == 0 and
    // thus has no effect for
    // input_offset
    outputs_offset += (is2 * dim_1 + is1) * output_stride + is0;
    outputc_offset += (ic2 * dim_1 + ic1) * output_stride + ic0;

    inputRe += input_offset;
    inputIm += input_offset;
    T* outputs = output + outputs_offset;
    T* outputc = output + outputc_offset;

    if((is0 == 0) || (is0 * 2 == dim_0)) // simply write the element to output
    {
        outputs[0].x = inputRe[0];
        outputs[0].y = inputIm[0];
        return;
    }

    if(is0 < hermitian_size)
    {
        outputs[0].x = inputRe[0];
        outputs[0].y = inputIm[0];
        outputc[0].x = inputRe[0];
        outputc[0].y = -inputIm[0];
    }
}

/// \brief auxiliary function
///   Read from input_buffer of hermitian structure into an output_buffer of regular
///   complex structure by padding 0.
/// @param[in] dim_0 size of problem, not the size of input buffer
/// @param[in] input_buffer data type : complex type (float2 or double2) but only store
/// first [1 + dim_0/2] elements according to conjugate symmetry
/// @param[in] input_distance distance between consecutive batch members for input buffer
/// @param[in,output] output_buffer data type : complex type (float2 or double2) of size
/// dim_0
/// @param[in] output_distance distance between consecutive batch members for output
/// buffer
/// @param[in] batch number of transforms
/// @param[in] precision data type of input and output buffer. rocfft_precision_single or
/// rocfft_precsion_double
ROCFFT_DEVICE_EXPORT void hermitian2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t dim_0          = data->node->length[0]; // dim_0 is the innermost dimension
    size_t hermitian_size = dim_0 / 2 + 1;

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (hermitian_size - 1) / LAUNCH_BOUNDS_R2C_C2R_KERNEL + 1;

    if(data->node->length.size() > 3)
        throw std::runtime_error("Error: dimension larger than 3, which is not handled");

    size_t dim_1 = 1, dim_2 = 1;
    if(data->node->length.size() >= 2)
        dim_1 = data->node->length[1];
    if(data->node->length.size() == 3)
        dim_2 = data->node->length[2];

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    //   float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
    //   hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
    //             hipMemcpyDeviceToHost);
    //   printf("herm size %d, dim0 %d, dim1 %d, dim2 %d\n", hermitian_size, dim_0,
    //          dim_1, dim_2);
    //   printf("input_stride %d output_stride %d input_distance %d output_distance
    // %d\n", input_stride, output_stride, input_distance, output_distance);
    //   //for(size_t j=0;j<data->node->length[1]; j++)
    //   size_t j = 0;
    //   {
    //       for(size_t i=0; i<input_stride; i++)
    //       {
    //           printf("kernel input[%zu][%zu]=(%f, %f) \n", j, i, tmp[j*input_stride
    //                                                                  + i].x, tmp[j*input_stride + i].y);
    //       }
    //   }

    if(data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
    {
        if(precision == rocfft_precision_single)
            hipLaunchKernelGGL(
                data->get_callback_type() == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        hermitian2complex_kernel<float2, CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(hermitian2complex_kernel<float2, CallbackType::NONE>),
                grid,
                threads,
                0,
                rocfft_stream,
                hermitian_size,
                dim_0,
                dim_1,
                dim_2,
                input_stride,
                output_stride,
                (float2*)input_buffer,
                input_distance,
                (float2*)output_buffer,
                output_distance,
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        else
            hipLaunchKernelGGL(
                data->get_callback_type() == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        hermitian2complex_kernel<double2, CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(hermitian2complex_kernel<double2, CallbackType::NONE>),
                grid,
                threads,
                0,
                rocfft_stream,
                hermitian_size,
                dim_0,
                dim_1,
                dim_2,
                input_stride,
                output_stride,
                (double2*)input_buffer,
                input_distance,
                (double2*)output_buffer,
                output_distance,
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
    }
    else if(data->node->inArrayType == rocfft_array_type_hermitian_planar)
    {
        if(precision == rocfft_precision_single)
            hipLaunchKernelGGL(hermitian2complex_kernel<float2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               hermitian_size,
                               dim_0,
                               dim_1,
                               dim_2,
                               input_stride,
                               output_stride,
                               (float*)data->bufIn[0],
                               (float*)data->bufIn[1],
                               input_distance,
                               (float2*)output_buffer,
                               output_distance);
        else
            hipLaunchKernelGGL(hermitian2complex_kernel<double2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               hermitian_size,
                               dim_0,
                               dim_1,
                               dim_2,
                               input_stride,
                               output_stride,
                               (double*)data->bufIn[0],
                               (double*)data->bufIn[1],
                               input_distance,
                               (double2*)output_buffer,
                               output_distance);
    }

    // float2* tmpo; tmpo = (float2*)malloc(sizeof(float2)*output_distance*batch);
    // hipMemcpy(tmpo, output_buffer, sizeof(float2)*output_distance*batch,
    //           hipMemcpyDeviceToHost);

    // {
    //     for(size_t i=0; i<output_stride; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmpo[j*output_stride + i].x, tmpo[j*output_stride + i].y);
    //     }
    // }
}
