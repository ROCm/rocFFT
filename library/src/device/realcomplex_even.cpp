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

#include "real2complex.h"
#include <iostream>
#include <numeric>

// NB: The kernel arguments for the buffers are void* instead of Tcomplex* (or the corresponding
// real type) in order to maintain the signature so that we can add the pointer to a std::map.  If
// we find another solution for organizing the calling structure, we should be explicit with the
// type.

// Interleaved version of r2c post-process kernel, 1D
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
// When N is divisible by 4, one value is handled separately; this is controlled by Ndiv4.
template <typename Tcomplex, bool Ndiv4>
__global__ static void __launch_bounds__(MAX_LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    real_post_process_kernel_interleaved_1D(const size_t half_N,
                                            const void*  input0,
                                            const size_t idist,
                                            void*        output0,
                                            const size_t odist,
                                            const void*  twiddles0)
{
    // blockIdx.y gives the multi-dimensional offset
    // blockIdx.z gives the batch offset

    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;
    const auto twiddles  = (Tcomplex*)twiddles0;

    if(idx_p < quarter_N)
    {
        // blockIdx.z gives the batch offset
        // clang format off
        const auto input  = (Tcomplex*)(input0) + blockIdx.z * idist;
        auto       output = (Tcomplex*)(output0) + blockIdx.z * odist;
        // clang format on

        post_process_interleaved<Tcomplex, Ndiv4>(
            idx_p, idx_q, half_N, quarter_N, input, output, twiddles);
    }
}

template <typename Tcomplex, bool Ndiv4>
__device__ inline void post_process_planar(const size_t           idx_p,
                                           const size_t           idx_q,
                                           const size_t           half_N,
                                           const size_t           quarter_N,
                                           const Tcomplex*        input,
                                           real_type_t<Tcomplex>* outputRe,
                                           real_type_t<Tcomplex>* outputIm,
                                           const Tcomplex*        twiddles)
{
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

// Planar version of r2c post-process kernel, 1D
template <typename Tcomplex, bool Ndiv4>
__global__ static void __launch_bounds__(MAX_LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    real_post_process_kernel_planar_1D(const size_t half_N,
                                       const void*  input0,
                                       const size_t idist,
                                       void*        output0,
                                       void*        output1,
                                       const size_t odist,
                                       const void*  twiddles0)
{
    // blockIdx.y gives the multi-dimensional offset
    // blockIdx.z gives the batch offset

    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;
    const auto twiddles  = (Tcomplex*)twiddles0;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset
        // blockIdx.z gives the batch offset
        // clang format off
        const auto input    = (Tcomplex*)(input0) + blockIdx.z * idist;
        auto       outputRe = (real_type_t<Tcomplex>*)(output0) + blockIdx.z * odist;
        auto       outputIm = (real_type_t<Tcomplex>*)(output1) + blockIdx.z * odist;
        // clang format on

        post_process_planar<Tcomplex, Ndiv4>(
            idx_p, idx_q, half_N, quarter_N, input, outputRe, outputIm, twiddles);
    }
}

// Entrance function for r2c post-processing kernel
void r2c_1d_post(const void* data_p, void*)
{
    // Map to 1D interleaved kernels:
    std::map<std::tuple<rocfft_precision, bool>,
             decltype(&real_post_process_kernel_interleaved_1D<float2, true>)>
        kernelmap_interleaved_1D;
    kernelmap_interleaved_1D.emplace(std::make_tuple(rocfft_precision_single, true),
                                     &(real_post_process_kernel_interleaved_1D<float2, true>));
    kernelmap_interleaved_1D.emplace(std::make_tuple(rocfft_precision_single, false),
                                     &(real_post_process_kernel_interleaved_1D<float2, false>));
    kernelmap_interleaved_1D.emplace(std::make_tuple(rocfft_precision_double, true),
                                     &(real_post_process_kernel_interleaved_1D<double2, true>));
    kernelmap_interleaved_1D.emplace(std::make_tuple(rocfft_precision_double, false),
                                     &(real_post_process_kernel_interleaved_1D<double2, false>));

    // Map to planar 1D kernels:
    std::map<std::tuple<rocfft_precision, bool>,
             decltype(&real_post_process_kernel_planar_1D<float2, true>)>
        kernelmap_planar_1D;
    kernelmap_planar_1D.emplace(std::make_tuple(rocfft_precision_single, true),
                                &(real_post_process_kernel_planar_1D<float2, true>));
    kernelmap_planar_1D.emplace(std::make_tuple(rocfft_precision_single, false),
                                &(real_post_process_kernel_planar_1D<float2, false>));
    kernelmap_planar_1D.emplace(std::make_tuple(rocfft_precision_double, true),
                                &(real_post_process_kernel_planar_1D<double2, true>));
    kernelmap_planar_1D.emplace(std::make_tuple(rocfft_precision_double, false),
                                &(real_post_process_kernel_planar_1D<double2, false>));

    const DeviceCallIn* data = (DeviceCallIn*)data_p;

    // Input_size is the innermost dimension
    // The upper level provides always N/2, that is regular complex fft size
    const size_t half_N = data->node->length[0];

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    const void* bufIn0  = data->bufIn[0];
    void*       bufOut0 = data->bufOut[0];
    void*       bufOut1 = data->bufOut[1];

    const size_t batch = data->node->batch;

    const size_t high_dimension = std::accumulate(
        data->node->length.begin() + 1, data->node->length.end(), 1, std::multiplies<size_t>());
    // Strides are actually distances between contiguous data vectors.

    // 2D and 3D post-processing should be fused with following
    // transpose, so this function should only be for 1D
    assert(high_dimension == 1);

    const bool                               Ndiv4  = half_N % 2 == 0;
    const std::tuple<rocfft_precision, bool> params = std::make_tuple(data->node->precision, Ndiv4);

    const size_t block_size = 64;
    const size_t blocks     = ((half_N + 1) / 2 + block_size - 1) / block_size;
    // The total number of 1D threads is N / 4, rounded up.

    const dim3 grid(blocks, high_dimension, batch);
    const dim3 threads(block_size, 1, 1);

    try
    {
        if(data->node->outArrayType == rocfft_array_type_hermitian_interleaved)
        {
            hipLaunchKernelGGL(kernelmap_interleaved_1D.at(params),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               half_N,
                               bufIn0,
                               idist,
                               bufOut0,
                               odist,
                               data->node->twiddles.data());
        }
        else
        {
            hipLaunchKernelGGL(kernelmap_planar_1D.at(params),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               half_N,
                               bufIn0,
                               idist,
                               bufOut0,
                               bufOut1,
                               odist,
                               data->node->twiddles.data());
        }
    }
    catch(std::exception& e)
    {
        rocfft_cerr << e.what() << std::endl;
    }
}

// Interleaved version of c2r pre-process kernel
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
// When N is divisible by 4, one value is handled separately; this is controlled by Ndiv4.
template <typename Tcomplex, bool Ndiv4>
__global__ static void __launch_bounds__(MAX_LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    real_pre_process_kernel(const size_t half_N,
                            const size_t idist1D,
                            const size_t odist1D,
                            const void*  input0,
                            const size_t idist,
                            void*        output0,
                            const size_t odist,
                            const void*  twiddles0)
{
    const size_t idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;
    const auto twiddles  = (Tcomplex*)twiddles0;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset, stride is [i/o]dist1D.
        // blockIdx.z gives the batch offset, stride is [i/o]dist.
        // clang format off
        const auto input  = (Tcomplex*)(input0) + idist1D * blockIdx.y + idist * blockIdx.z;
        auto       output = (Tcomplex*)(output0) + odist1D * blockIdx.y + odist * blockIdx.z;
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

// Planar version of c2r pre-process kernel
template <typename Tcomplex, bool Ndiv4>
__global__ static void __launch_bounds__(MAX_LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    real_pre_process_kernel_planar(const size_t      half_N,
                                   const size_t      idist1D,
                                   const size_t      odist1D,
                                   const void*       input0,
                                   const void*       input1,
                                   const size_t      idist,
                                   void*             output0,
                                   const size_t      odist,
                                   const void* const twiddles0)
{
    const size_t idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;
    const auto twiddles  = (Tcomplex*)twiddles0;

    if(idx_p < quarter_N)
    {
        // blockIdx.y gives the multi-dimensional offset, stride is [i/o]dist1D.
        // blockIdx.z gives the batch offset, stride is [i/o]dist.
        // clang format off
        const auto inputRe
            = (real_type_t<Tcomplex>*)(input0) + idist1D * blockIdx.y + idist * blockIdx.z;
        const auto inputIm
            = (real_type_t<Tcomplex>*)(input1) + idist1D * blockIdx.y + idist * blockIdx.z;
        auto output = (Tcomplex*)(output0) + odist1D * blockIdx.y + odist * blockIdx.z;
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

// Entrance function for c2r pre-processing kernel
void c2r_1d_pre(const void* data_p, void*)
{
    // map to interleaved kernels
    std::map<std::tuple<rocfft_precision, bool>, decltype(&real_pre_process_kernel<double2, true>)>
        kernelmap_interleaved;
    kernelmap_interleaved.emplace(std::make_tuple(rocfft_precision_single, true),
                                  &(real_pre_process_kernel<float2, true>));
    kernelmap_interleaved.emplace(std::make_tuple(rocfft_precision_single, false),
                                  &(real_pre_process_kernel<float2, false>));
    kernelmap_interleaved.emplace(std::make_tuple(rocfft_precision_double, true),
                                  &(real_pre_process_kernel<double2, true>));
    kernelmap_interleaved.emplace(std::make_tuple(rocfft_precision_double, false),
                                  &(real_pre_process_kernel<double2, false>));

    // map to planar kernels
    std::map<std::tuple<rocfft_precision, bool>,
             decltype(&real_pre_process_kernel_planar<double2, true>)>
        kernelmap_planar;
    kernelmap_planar.emplace(std::make_tuple(rocfft_precision_single, true),
                             &(real_pre_process_kernel_planar<float2, true>));
    kernelmap_planar.emplace(std::make_tuple(rocfft_precision_single, false),
                             &(real_pre_process_kernel_planar<float2, false>));
    kernelmap_planar.emplace(std::make_tuple(rocfft_precision_double, true),
                             &(real_pre_process_kernel_planar<double2, true>));
    kernelmap_planar.emplace(std::make_tuple(rocfft_precision_double, false),
                             &(real_pre_process_kernel_planar<double2, false>));

    const DeviceCallIn* data = (DeviceCallIn*)data_p;

    // Input_size is the innermost dimension
    // The upper level provides always N/2, that is regular complex fft size
    const size_t half_N = data->node->length[0];

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    const void* bufIn0  = data->bufIn[0];
    const void* bufIn1  = data->bufIn[1];
    void*       bufOut0 = data->bufOut[0];

    const size_t batch = data->node->batch;

    const size_t high_dimension = std::accumulate(
        data->node->length.begin() + 1, data->node->length.end(), 1, std::multiplies<size_t>());

    // 2D and 3D pre-processing should be fused with preceding
    // transpose, so this function should only be for 1D
    assert(high_dimension == 1);

    // Strides are actually distances between contiguous data vectors.
    const size_t istride = 0;
    const size_t ostride = 0;

    const bool                               Ndiv4  = half_N % 2 == 0;
    const std::tuple<rocfft_precision, bool> params = std::make_tuple(data->node->precision, Ndiv4);

    const size_t block_size = 64;
    const size_t blocks     = ((half_N + 1) / 2 + block_size - 1) / block_size;
    // The total number of 1D threads is N / 4, rounded up.

    const dim3 grid(blocks, high_dimension, batch);
    const dim3 threads(block_size, 1, 1);

    const size_t idist1D = istride;
    const size_t odist1D = ostride;

    try
    {
        if(data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
        {
            hipLaunchKernelGGL(kernelmap_interleaved.at(params),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               bufIn0,
                               idist,
                               bufOut0,
                               odist,
                               data->node->twiddles.data());
        }
        else
        {
            hipLaunchKernelGGL(kernelmap_planar.at(params),
                               grid,
                               threads,
                               0,
                               data->rocfft_stream,
                               half_N,
                               idist1D,
                               odist1D,
                               bufIn0,
                               bufIn1,
                               idist,
                               bufOut0,
                               odist,
                               data->node->twiddles.data());
        }
    }
    catch(std::exception& e)
    {
        rocfft_cout << e.what() << std::endl;
    }
}
