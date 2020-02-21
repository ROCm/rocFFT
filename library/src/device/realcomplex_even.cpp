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

// GPU kernel for 1d r2c post-process
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
// When N is divisible by 4, one value is handled separately; this is controlled by Ndiv4.
template <typename Tcomplex, bool Ndiv4>
__global__ static void real_post_process_kernel(const size_t    half_N,
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
__global__ static void real_post_process_kernel(const size_t           half_N,
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
__global__ static void real_pre_process_kernel(const size_t    half_N,
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
__global__ static void real_pre_process_kernel(const size_t           half_N,
                                               const size_t           idist1D,
                                               const size_t           odist1D,
                                               real_type_t<Tcomplex>* input0Re,
                                               real_type_t<Tcomplex>* input0Im,
                                               const size_t           idist,
                                               Tcomplex*              output0,
                                               const size_t           odist,
                                               const Tcomplex* const  twiddles)
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
static void real_1d_pre_post_process(const size_t half_N,
                                     const size_t batch,
                                     Tcomplex*    d_input,
                                     Tcomplex*    d_output,
                                     Tcomplex*    d_twiddles,
                                     const size_t high_dimension,
                                     const size_t istride,
                                     const size_t ostride,
                                     const size_t idist,
                                     const size_t odist,
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
static void real_1d_pre_post_process(const size_t           half_N,
                                     const size_t           batch,
                                     Tcomplex*              d_real,
                                     real_type_t<Tcomplex>* d_complexRe,
                                     real_type_t<Tcomplex>* d_complexIm,
                                     Tcomplex*              d_twiddles,
                                     const size_t           high_dimension,
                                     const size_t           istride,
                                     const size_t           ostride,
                                     const size_t           idist,
                                     const size_t           odist,
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
static void real_1d_pre_post(const void* data_p, void* back_p)
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
