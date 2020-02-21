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
#include <numeric>

template <typename Tcomplex>
__global__ static void real2complex_kernel(const size_t                 input_size,
                                           const size_t                 idist1D,
                                           const size_t                 odist1D,
                                           const real_type_t<Tcomplex>* input0,
                                           const size_t                 idist,
                                           Tcomplex*                    output0,
                                           const size_t                 odist)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < input_size)
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const auto input  = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        auto       output = output0 + blockIdx.y * odist1D + blockIdx.z * odist;

        output[tid].x = input[tid];
        output[tid].y = 0.0;
    }
}

/// \brief auxiliary function
///    convert a real vector into a complex one by padding the imaginary part with  0.
///    @param[in] input_size, size of input buffer
///    @param[in] input_buffer, data type : float or double
///    @param[in] idist, distance between consecutive batch members for input buffer
///    @param[in,output] output_buffer, data type : complex type (float2 or double2)
///    @param[in] odist, distance between consecutive batch members for output buffer
///    @param[in] batch, number of transforms
///    @param[in] precision, data type of input buffer. rocfft_precision_single or
///                          rocfft_precsion_double
void real2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

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

    size_t blocks = (input_size - 1) / 512 + 1;

    // TODO: verify with API that high_dimension and batch aren't too big.

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(real2complex_kernel<float2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (float*)input_buffer,
                           input_distance,
                           (float2*)output_buffer,
                           output_distance);
    else
        hipLaunchKernelGGL(real2complex_kernel<double2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (double*)input_buffer,
                           input_distance,
                           (double2*)output_buffer,
                           output_distance);

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*output_distance*batch);
    // hipMemcpy(tmp, output_buffer, sizeof(float2)*output_distance*batch,
    //           hipMemcpyDeviceToHost);

    // for(size_t j=0;j<data->node->length[1]; j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
    //     }
    // }
}

// The complex to hermitian simple copy kernel for interleaved format
template <typename Tcomplex>
__global__ static void complex2hermitian_kernel(const size_t    input_size,
                                                const size_t    idist1D,
                                                const size_t    odist1D,
                                                const Tcomplex* input0,
                                                const size_t    idist,
                                                Tcomplex*       output0,
                                                const size_t    odist)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // only read and write the first [input_size/2+1] elements due to conjugate redundancy
    if(tid < (1 + input_size / 2))
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const auto input  = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        auto       output = output0 + blockIdx.y * odist1D + blockIdx.z * odist;

        output[tid] = input[tid];
    }
}

// The planar overload function of the above interleaved one
template <typename Tcomplex>
__global__ static void complex2hermitian_kernel(const size_t           input_size,
                                                const size_t           idist1D,
                                                const size_t           odist1D,
                                                const Tcomplex*        input0,
                                                const size_t           idist,
                                                real_type_t<Tcomplex>* outputRe0,
                                                real_type_t<Tcomplex>* outputIm0,
                                                const size_t           odist)
{
    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // only read and write the first [input_size/2+1] elements due to conjugate redundancy
    if(tid < (1 + input_size / 2))
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const auto input    = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        auto       outputRe = outputRe0 + blockIdx.y * odist1D + blockIdx.z * odist;
        auto       outputIm = outputIm0 + blockIdx.y * odist1D + blockIdx.z * odist;

        outputRe[tid] = input[tid].x;
        outputIm[tid] = input[tid].y;
    }
}

/// \brief auxiliary function
///   read from input_buffer and store the first  [1 + input_size/2] elements to
///   the output_buffer
/// @param[in] input_size, size of input buffer
/// @param[in] input_buffer, data type dictated by precision parameter but complex type
///                          (float2 or double2)
/// @param[in] idist, distance between consecutive batch members for input buffer
/// @param[in,output] output_buffer, data type dictated by precision parameter but complex
///                   type (float2 or double2) but only store first [1 + input_size/2]
///                   elements according to conjugate symmetry
/// @param[in] odist, distance between consecutive batch members for output buffer
/// @param[in] batch, number of transforms
/// @param[in] precision, data type of input and output buffer. rocfft_precision_single or
///            rocfft_precsion_double
void complex2hermitian(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

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

    size_t blocks = (input_size - 1) / 512 + 1;

    // TODO: verify with API that high_dimension and batch aren't too big.

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
    // hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
    //           hipMemcpyDeviceToHost);

    // for(size_t j=0;j<data->node->length[1]; j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
    //     }
    // }

    // TODO: check the input type
    if(data->node->outArrayType == rocfft_array_type_hermitian_interleaved)
    {
        if(precision == rocfft_precision_single)
            hipLaunchKernelGGL(complex2hermitian_kernel<float2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               input_size,
                               input_stride,
                               output_stride,
                               (float2*)input_buffer,
                               input_distance,
                               (float2*)output_buffer,
                               output_distance);
        else
            hipLaunchKernelGGL(complex2hermitian_kernel<double2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               input_size,
                               input_stride,
                               output_stride,
                               (double2*)input_buffer,
                               input_distance,
                               (double2*)output_buffer,
                               output_distance);
    }
    else if(data->node->outArrayType == rocfft_array_type_hermitian_planar)
    {
        if(precision == rocfft_precision_single)
            hipLaunchKernelGGL(complex2hermitian_kernel<float2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               input_size,
                               input_stride,
                               output_stride,
                               (float2*)input_buffer,
                               input_distance,
                               (float*)data->bufOut[0],
                               (float*)data->bufOut[1],
                               output_distance);
        else
            hipLaunchKernelGGL(complex2hermitian_kernel<double2>,
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               input_size,
                               input_stride,
                               output_stride,
                               (double2*)input_buffer,
                               input_distance,
                               (double*)data->bufOut[0],
                               (double*)data->bufOut[1],
                               output_distance);
    }
    else
    {
        assert(0);
        std::cout << "Unsupported output format in complex2hermitian kernel!" << std::endl;
    }
}
