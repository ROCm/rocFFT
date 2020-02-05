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

template <typename T>
__global__ void real2complex_kernel(size_t          input_size,
                                    size_t          input_stride,
                                    size_t          output_stride,
                                    real_type_t<T>* input,
                                    size_t          input_distance,
                                    T*              output,
                                    size_t          output_distance)
{
    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch offset

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < input_size)
    {
        output[tid].y = 0.0;
        output[tid].x = input[tid];
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
template <typename T>
__global__ void complex2hermitian_kernel(size_t input_size,
                                         size_t input_stride,
                                         size_t output_stride,
                                         T*     input,
                                         size_t input_distance,
                                         T*     output,
                                         size_t output_distance)
{

    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < (1 + input_size / 2)) // only read and write the first
    // [input_size/2+1] elements due to conjugate
    // redundancy
    {
        output[tid] = input[tid];
    }
}

// The planar overload function of the above interleaved one
template <typename T>
__global__ void complex2hermitian_kernel(size_t          input_size,
                                         size_t          input_stride,
                                         size_t          output_stride,
                                         T*              input,
                                         size_t          input_distance,
                                         real_type_t<T>* outputRe,
                                         real_type_t<T>* outputIm,
                                         size_t          output_distance)
{

    size_t input_offset = hipBlockIdx_z * input_distance; // batch offset

    size_t output_offset = hipBlockIdx_z * output_distance; // batch

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    outputRe += output_offset;
    outputIm += output_offset;

    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < (1 + input_size / 2)) // only read and write the first
    // [input_size/2+1] elements due to conjugate
    // redundancy
    {
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

    return;
}

// GPU kernel for 1d r2c post-process
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
// When N is divisible by 4, one value is handled separately; this is controlled by Ndiv4.
template <typename Tcomplex, bool Ndiv4>
__global__ void real_post_process_kernel(const size_t    half_N,
                                         const size_t    idist1D,
                                         const size_t    odist1D,
                                         const Tcomplex* input0,
                                         const size_t    idist,
                                         Tcomplex*       output0,
                                         const size_t    odist,
                                         Tcomplex const* twiddles)
{
    // blockIdx.y gives the multi-dimensional offset
    // blockIdx.z gives the batch offset

    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const Tcomplex* input  = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        Tcomplex*       output = output0 + blockIdx.y * odist1D + blockIdx.z * odist;

        if(idx_p == 0)
        {
            output[half_N].x = input[0].x - input[0].y;
            output[half_N].y = 0;
            output[0].x      = input[0].x + input[0].y;
            output[0].y      = 0;

            if(Ndiv4)
            {
                output[quarter_N].x = input[quarter_N].x;
                output[quarter_N].y = -input[quarter_N].y;
            }
        }
        else
        {
            const Tcomplex p = input[idx_p];
            const Tcomplex q = input[idx_q];
            const Tcomplex u = 0.5 * (p + q);
            const Tcomplex v = 0.5 * (p - q);

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

            output[idx_p].x = u.x + v.x * twd_p.y + u.y * twd_p.x;
            output[idx_p].y = v.y + u.y * twd_p.y - v.x * twd_p.x;

            output[idx_q].x = u.x - v.x * twd_p.y - u.y * twd_p.x;
            output[idx_q].y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        }
    }
}

// Overloaded version of the above for planar format
template <typename Tcomplex, bool Ndiv4>
__global__ void real_post_process_kernel(const size_t           half_N,
                                         const size_t           idist1D,
                                         const size_t           odist1D,
                                         const Tcomplex*        input0,
                                         const size_t           idist,
                                         real_type_t<Tcomplex>* output0Re,
                                         real_type_t<Tcomplex>* output0Im,
                                         const size_t           odist,
                                         Tcomplex const*        twiddles)
{
    // blockIdx.y gives the multi-dimensional offset
    // blockIdx.z gives the batch offset

    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        const Tcomplex*        input    = input0 + blockIdx.y * idist1D + blockIdx.z * idist;
        real_type_t<Tcomplex>* outputRe = output0Re + blockIdx.y * odist1D + blockIdx.z * odist;
        real_type_t<Tcomplex>* outputIm = output0Im + blockIdx.y * odist1D + blockIdx.z * odist;

        if(idx_p == 0)
        {
            outputRe[half_N] = input[0].x - input[0].y;
            outputIm[half_N] = 0;
            outputRe[0]      = input[0].x + input[0].y;
            outputIm[0]      = 0;

            if(Ndiv4)
            {
                outputRe[quarter_N] = input[quarter_N].x;
                outputIm[quarter_N] = -input[quarter_N].y;
            }
        }
        else
        {
            const Tcomplex p = input[idx_p];
            const Tcomplex q = input[idx_q];
            const Tcomplex u = 0.5 * (p + q);
            const Tcomplex v = 0.5 * (p - q);

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

            outputRe[idx_p] = u.x + v.x * twd_p.y + u.y * twd_p.x;
            outputIm[idx_p] = v.y + u.y * twd_p.y - v.x * twd_p.x;

            outputRe[idx_q] = u.x - v.x * twd_p.y - u.y * twd_p.x;
            outputIm[idx_q] = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        }
    }
}

// GPU kernel for 1d c2r pre-process
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
// When N is divisible by 4, one value is handled separately; this is controlled by Ndiv4.
template <typename Tcomplex, bool Ndiv4>
__global__ void real_pre_process_kernel(const size_t    half_N,
                                        const size_t    idist1D,
                                        const size_t    odist1D,
                                        const Tcomplex* input0,
                                        const size_t    idist,
                                        Tcomplex*       output0,
                                        const size_t    odist,
                                        Tcomplex const* twiddles)
{
    const size_t idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset, stride is [i/o]dist1D.
        // blockIdx.z gives the batch offset, stride is [i/o]dist.
        // clang format off
        const Tcomplex* input  = input0 + idist1D * blockIdx.y + idist * blockIdx.z;
        Tcomplex*       output = output0 + odist1D * blockIdx.y + odist * blockIdx.z;
        // clang format on

        if(idx_p == 0)
        {
            // NB: multi-dimensional transforms may have non-zero
            // imaginary part at index 0 or at the Nyquist frequency.

            const Tcomplex p = input[idx_p];
            const Tcomplex q = input[idx_q];
            output[idx_p].x  = p.x - p.y + q.x + q.y;
            output[idx_p].y  = p.x + p.y - q.x + q.y;

            if(Ndiv4)
            {
                output[quarter_N].x = 2.0 * input[quarter_N].x;
                output[quarter_N].y = -2.0 * input[quarter_N].y;
            }
        }
        else
        {
            const Tcomplex p = input[idx_p];
            const Tcomplex q = input[idx_q];

            const Tcomplex u = p + q;
            const Tcomplex v = p - q;

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p);

            output[idx_p].x = u.x + v.x * twd_p.y - u.y * twd_p.x;
            output[idx_p].y = v.y + u.y * twd_p.y + v.x * twd_p.x;

            output[idx_q].x = u.x - v.x * twd_p.y + u.y * twd_p.x;
            output[idx_q].y = -v.y + u.y * twd_p.y + v.x * twd_p.x;
        }
    }
}

// Overloaded version of the above for planar format
template <typename Tcomplex, bool Ndiv4>
__global__ void real_pre_process_kernel(const size_t           half_N,
                                        const size_t           idist1D,
                                        const size_t           odist1D,
                                        real_type_t<Tcomplex>* input0Re,
                                        real_type_t<Tcomplex>* input0Im,
                                        const size_t           idist,
                                        Tcomplex*              output0,
                                        const size_t           odist,
                                        Tcomplex const*        twiddles)
{
    const size_t idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset, stride is [i/o]dist1D.
        // blockIdx.z gives the batch offset, stride is [i/o]dist.
        // clang format off
        const real_type_t<Tcomplex>* inputRe = input0Re + idist1D * blockIdx.y + idist * blockIdx.z;
        const real_type_t<Tcomplex>* inputIm = input0Im + idist1D * blockIdx.y + idist * blockIdx.z;
        Tcomplex*                    output  = output0 + odist1D * blockIdx.y + odist * blockIdx.z;
        // clang format on

        if(idx_p == 0)
        {
            // NB: multi-dimensional transforms may have non-zero
            // imaginary part at index 0 or at the Nyquist frequency.

            Tcomplex p, q;
            p.x             = inputRe[idx_p];
            p.y             = inputIm[idx_p];
            q.x             = inputRe[idx_q];
            q.y             = inputIm[idx_q];
            output[idx_p].x = p.x - p.y + q.x + q.y;
            output[idx_p].y = p.x + p.y - q.x + q.y;

            if(Ndiv4)
            {
                output[quarter_N].x = 2.0 * inputRe[quarter_N];
                output[quarter_N].y = -2.0 * inputIm[quarter_N];
            }
        }
        else
        {
            Tcomplex p, q;
            p.x = inputRe[idx_p];
            p.y = inputIm[idx_p];
            q.x = inputRe[idx_q];
            q.y = inputIm[idx_q];

            const Tcomplex u = p + q;
            const Tcomplex v = p - q;

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p);

            output[idx_p].x = u.x + v.x * twd_p.y - u.y * twd_p.x;
            output[idx_p].y = v.y + u.y * twd_p.y + v.x * twd_p.x;

            output[idx_q].x = u.x - v.x * twd_p.y + u.y * twd_p.x;
            output[idx_q].y = -v.y + u.y * twd_p.y + v.x * twd_p.x;
        }
    }
}

// GPU intermediate host code
template <typename Tcomplex, bool R2C>
void real_1d_pre_post_process(size_t const half_N,
                              size_t       batch,
                              Tcomplex*    d_input,
                              Tcomplex*    d_output,
                              Tcomplex*    d_twiddles,
                              size_t       high_dimension,
                              size_t       istride,
                              size_t       ostride,
                              size_t       idist,
                              size_t       odist,
                              hipStream_t  rocfft_stream)
{
    const size_t block_size = 512;
    size_t       blocks     = ((half_N + 1) / 2 + block_size - 1) / block_size;
    // The total number of 1D threads is N / 4, rounded up.

    // TODO: verify with API that high_dimension and batch aren't too big.

    const size_t idist1D = istride;
    const size_t odist1D = ostride;

    // std::cout << "idist1D: " << idist1D << std::endl;
    // std::cout << "odist1D: " << odist1D << std::endl;
    // std::cout << "high_dimension: " << high_dimension << std::endl;
    // std::cout << "batch: " << batch << std::endl;

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    const bool Ndiv4 = half_N % 2 == 0;

    // std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    // std::cout << "threads: " << threads.x << " " << threads.y << " " << threads.z << std::endl;

    if(R2C)
    {
        if(Ndiv4)
            hipLaunchKernelGGL((real_post_process_kernel<Tcomplex, true>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_input,
                               idist,
                               d_output,
                               odist,
                               d_twiddles);
        else
            hipLaunchKernelGGL((real_post_process_kernel<Tcomplex, false>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_input,
                               idist,
                               d_output,
                               odist,
                               d_twiddles);
    }
    else
    {
        if(Ndiv4)
            hipLaunchKernelGGL((real_pre_process_kernel<Tcomplex, true>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_input,
                               idist,
                               d_output,
                               odist,
                               d_twiddles);
        else
            hipLaunchKernelGGL((real_pre_process_kernel<Tcomplex, false>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_input,
                               idist,
                               d_output,
                               odist,
                               d_twiddles);
    }
}

// Overloaded version of the above for planar format
// For R2C, d_real is the input, d_complexRe and d_complexIm are outputs.
// For C2R, d_complexRe and d_complexIm are inputs, and d_real is output.
template <typename Tcomplex, bool R2C>
void real_1d_pre_post_process(size_t const           half_N,
                              size_t                 batch,
                              Tcomplex*              d_real,
                              real_type_t<Tcomplex>* d_complexRe,
                              real_type_t<Tcomplex>* d_complexIm,
                              Tcomplex*              d_twiddles,
                              size_t                 high_dimension,
                              size_t                 istride,
                              size_t                 ostride,
                              size_t                 idist,
                              size_t                 odist,
                              hipStream_t            rocfft_stream)
{
    const size_t block_size = 512;
    size_t       blocks     = ((half_N + 1) / 2 + block_size - 1) / block_size;
    // The total number of 1D threads is N / 4, rounded up.

    // TODO: verify with API that high_dimension and batch aren't too big.

    const size_t idist1D = istride;
    const size_t odist1D = ostride;

    // std::cout << "idist1D: " << idist1D << std::endl;
    // std::cout << "odist1D: " << odist1D << std::endl;
    // std::cout << "high_dimension: " << high_dimension << std::endl;
    // std::cout << "batch: " << batch << std::endl;
    // std::cout << "d_complexRe: " << &d_complexRe << ", d_complexIm " << &d_complexIm << std::endl;
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    const bool Ndiv4 = half_N % 2 == 0;

    // std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    // std::cout << "threads: " << threads.x << " " << threads.y << " " << threads.z << std::endl;

    if(R2C)
    {
        if(Ndiv4)
            hipLaunchKernelGGL((real_post_process_kernel<Tcomplex, true>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_real,
                               idist,
                               d_complexRe,
                               d_complexIm,
                               odist,
                               d_twiddles);
        else
            hipLaunchKernelGGL((real_post_process_kernel<Tcomplex, false>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_real,
                               idist,
                               d_complexRe,
                               d_complexIm,
                               odist,
                               d_twiddles);
    }
    else
    {
        if(Ndiv4)
            hipLaunchKernelGGL((real_pre_process_kernel<Tcomplex, true>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_complexRe,
                               d_complexIm,
                               idist,
                               d_real,
                               odist,
                               d_twiddles);
        else
            hipLaunchKernelGGL((real_pre_process_kernel<Tcomplex, false>),
                               grid,
                               threads,
                               0,
                               rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               d_complexRe,
                               d_complexIm,
                               idist,
                               d_real,
                               odist,
                               d_twiddles);
    }
}

// Local macro, to shorten the redundant expression
// in the below real_1d_pre_post()
#define RUN_REAL_1D_PRE_POST_INTERLEAVED(PRECISION, BUF_IN, BUF_OUT)             \
    real_1d_pre_post_process<PRECISION, R2C>(half_N,                             \
                                             batch,                              \
                                             (PRECISION*)BUF_IN,                 \
                                             (PRECISION*)BUF_OUT,                \
                                             (PRECISION*)(data->node->twiddles), \
                                             high_dimension,                     \
                                             istride,                            \
                                             ostride,                            \
                                             idist,                              \
                                             odist,                              \
                                             data->rocfft_stream)

#define RUN_REAL_1D_PRE_POST_PLANAR(                                             \
    PRECISION, PRECISION_REAL, BUF_REAL, BUF_COMPLEX_RE, BUF_COMPLEX_IM)         \
    real_1d_pre_post_process<PRECISION, R2C>(half_N,                             \
                                             batch,                              \
                                             (PRECISION*)BUF_REAL,               \
                                             (PRECISION_REAL*)BUF_COMPLEX_RE,    \
                                             (PRECISION_REAL*)BUF_COMPLEX_IM,    \
                                             (PRECISION*)(data->node->twiddles), \
                                             high_dimension,                     \
                                             istride,                            \
                                             ostride,                            \
                                             idist,                              \
                                             odist,                              \
                                             data->rocfft_stream)

template <bool R2C>
void real_1d_pre_post(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    // input_size is the innermost dimension
    // the upper level provides always N/2, that is regular complex fft size
    size_t half_N = data->node->length[0];

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    void* bufIn0  = data->bufIn[0];
    void* bufOut0 = data->bufOut[0];
    void* bufIn1  = data->bufIn[1];
    void* bufOut1 = data->bufOut[1];

    size_t batch = data->node->batch;

    const size_t high_dimension = std::accumulate(
        data->node->length.begin() + 1, data->node->length.end(), 1, std::multiplies<size_t>());
    // Strides are actually distances between contiguous data vectors.
    const size_t istride = high_dimension > 1 ? data->node->inStride[1] : 0;
    const size_t ostride = high_dimension > 1 ? data->node->outStride[1] : 0;

    if((R2C && data->node->inArrayType == rocfft_array_type_complex_interleaved
        && data->node->outArrayType == rocfft_array_type_hermitian_interleaved)
       || ((!R2C) && data->node->inArrayType == rocfft_array_type_hermitian_interleaved
           && data->node->outArrayType == rocfft_array_type_complex_interleaved))
    { // case for regular real <-> hermitian interleaved
        if(data->node->precision == rocfft_precision_single)
            RUN_REAL_1D_PRE_POST_INTERLEAVED(float2, bufIn0, bufOut0);
        else
            RUN_REAL_1D_PRE_POST_INTERLEAVED(double2, bufIn0, bufOut0);
    }
    else if(R2C && data->node->inArrayType == rocfft_array_type_complex_interleaved
            && data->node->outArrayType == rocfft_array_type_hermitian_planar)
    { // case for real to hermitian planar
        if(data->node->precision == rocfft_precision_single)
            RUN_REAL_1D_PRE_POST_PLANAR(float2, float, bufIn0, bufOut0, bufOut1);
        else
            RUN_REAL_1D_PRE_POST_PLANAR(double2, double, bufIn0, bufOut0, bufOut1);
    }
    else if((!R2C) && data->node->inArrayType == rocfft_array_type_hermitian_planar
            && data->node->outArrayType == rocfft_array_type_complex_interleaved)
    { // case for hermitian planar to real
        if(data->node->precision == rocfft_precision_single)
            RUN_REAL_1D_PRE_POST_PLANAR(float2, float, bufOut0, bufIn0, bufIn1);
        else
            RUN_REAL_1D_PRE_POST_PLANAR(double2, double, bufOut0, bufIn0, bufIn1);
    }
    else
    {
        assert(0);
        std::cout << "Unsupported array type in real_1d_pre_post\n";
        return;
    }
}

void r2c_1d_post(const void* data_p, void* back_p)
{
    real_1d_pre_post<true>(data_p, back_p);
}

void c2r_1d_pre(const void* data_p, void* back_p)
{
    real_1d_pre_post<false>(data_p, back_p);
}
