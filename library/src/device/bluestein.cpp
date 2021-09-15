/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "bluestein.h"
#include "kernel_launch.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename T>
rocfft_status chirp_launch(
    size_t N, size_t M, T* B, void* twiddles_large, int twl, int dir, hipStream_t rocfft_stream)
{
    dim3 grid((M - N) / LAUNCH_BOUNDS_BLUESTEIN_KERNEL + 1);
    dim3 threads(LAUNCH_BOUNDS_BLUESTEIN_KERNEL);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(chirp_device<T>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       rocfft_stream,
                       N,
                       M,
                       B,
                       (T*)twiddles_large,
                       twl,
                       dir);

    return rocfft_status_success;
}

ROCFFT_DEVICE_EXPORT void rocfft_internal_chirp(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t N = data->node->length[0];
    size_t M = data->node->lengthBlue;

    int twl = 0;

    if(data->node->large1D > (size_t)256 * 256 * 256 * 256)
        printf("large1D twiddle size too large error");
    else if(data->node->large1D > (size_t)256 * 256 * 256)
        twl = 4;
    else if(data->node->large1D > (size_t)256 * 256)
        twl = 3;
    // TODO- possibly using a smaller LargeTwdBase for chirp by large_twiddle_base
    else if(data->node->large1D > (size_t)256)
        twl = 2;
    else
        twl = 1;

    int dir = data->node->direction;

    hipStream_t rocfft_stream = data->rocfft_stream;

    if(data->node->precision == rocfft_precision_single)
        chirp_launch<float2>(N,
                             M,
                             (float2*)data->bufOut[0],
                             data->node->twiddles_large.data(),
                             twl,
                             dir,
                             rocfft_stream);
    else
        chirp_launch<double2>(N,
                              M,
                              (double2*)data->bufOut[0],
                              data->node->twiddles_large.data(),
                              twl,
                              dir,
                              rocfft_stream);
}

ROCFFT_DEVICE_EXPORT void rocfft_internal_mul(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t N = data->node->length[0];
    size_t M = data->node->lengthBlue;

    // TODO:: fix the local scheme with enum class and pass it
    //        into kernel as a template parameter
    int scheme = 0; // fft mul
    if(data->node->scheme == CS_KERNEL_PAD_MUL)
    {
        scheme = 1; // pad mul
    }
    else if(data->node->scheme == CS_KERNEL_RES_MUL)
    {
        scheme = 2; // res mul
    }
    CallbackType cbtype = data->get_callback_type();

    size_t cBytes;
    if(data->node->precision == rocfft_precision_single)
    {
        cBytes = sizeof(float) * 2;
    }
    else
    {
        cBytes = sizeof(double) * 2;
    }

    void* bufIn0  = data->bufIn[0];
    void* bufOut0 = data->bufOut[0];
    void* bufIn1  = data->bufIn[1];
    void* bufOut1 = data->bufOut[1];

    // TODO: Not all in/out interleaved/planar combinations support for all 3
    // schemes until we figure out the buffer offset for planar format.
    // At least, planar for CS_KERNEL_PAD_MUL input and CS_KERNEL_RES_MUL output
    // are good enough for current strategy(check TreeNode::ReviseLeafsArrayType).
    // That is why we add asserts below.

    size_t numof = 0;
    if(scheme == 0)
    {
        bufIn0  = ((char*)bufIn0 + M * cBytes);
        bufOut0 = ((char*)bufOut0 + 2 * M * cBytes);

        numof = M;
    }
    else if(scheme == 1)
    {
        bufOut0 = ((char*)bufOut0 + M * cBytes);

        numof = M;
    }
    else if(scheme == 2)
    {
        numof = N;
    }

    size_t count = data->node->batch;
    for(size_t i = 1; i < data->node->length.size(); i++)
        count *= data->node->length[i];
    count *= numof;

    int dir = data->node->direction;

    hipStream_t rocfft_stream = data->rocfft_stream;

    dim3 grid((count - 1) / LAUNCH_BOUNDS_BLUESTEIN_KERNEL + 1);
    dim3 threads(LAUNCH_BOUNDS_BLUESTEIN_KERNEL);

    if((data->node->inArrayType == rocfft_array_type_complex_interleaved
        || data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
       && (data->node->outArrayType == rocfft_array_type_complex_interleaved
           || data->node->outArrayType == rocfft_array_type_hermitian_interleaved))
    {
        if(data->node->precision == rocfft_precision_single)
        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(mul_device_I_I<float2, CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(mul_device_I_I<float2, CallbackType::NONE>),
                dim3(grid),
                dim3(threads),
                0,
                rocfft_stream,
                numof,
                count,
                N,
                M,
                (const float2*)bufIn0,
                (float2*)bufOut0,
                data->node->length.size(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                dir,
                scheme,
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
        else
        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(mul_device_I_I<double2, CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(mul_device_I_I<double2, CallbackType::NONE>),
                dim3(grid),
                dim3(threads),
                0,
                rocfft_stream,
                numof,
                count,
                N,
                M,
                (const double2*)bufIn0,
                (double2*)bufOut0,
                data->node->length.size(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                dir,
                scheme,
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
    }
    else if((data->node->inArrayType == rocfft_array_type_complex_planar
             || data->node->inArrayType == rocfft_array_type_hermitian_planar)
            && (data->node->outArrayType == rocfft_array_type_complex_interleaved
                || data->node->outArrayType == rocfft_array_type_hermitian_interleaved))
    {
        assert(scheme != 0);
        assert(scheme != 2);

        if(data->node->precision == rocfft_precision_single)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_P_I<float2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const real_type_t<float2>*)bufIn0,
                               (const real_type_t<float2>*)bufIn1,
                               (float2*)bufOut0,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_P_I<double2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const real_type_t<double2>*)bufIn0,
                               (const real_type_t<double2>*)bufIn1,
                               (double2*)bufOut0,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
    }
    else if((data->node->inArrayType == rocfft_array_type_complex_interleaved
             || data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
            && (data->node->outArrayType == rocfft_array_type_complex_planar
                || data->node->outArrayType == rocfft_array_type_hermitian_planar))
    {
        assert(scheme != 0);
        assert(scheme != 1);

        if(data->node->precision == rocfft_precision_single)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_I_P<float2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const float2*)bufIn0,
                               (real_type_t<float2>*)bufOut0,
                               (real_type_t<float2>*)bufOut1,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_I_P<double2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const double2*)bufIn0,
                               (real_type_t<double2>*)bufOut0,
                               (real_type_t<double2>*)bufOut1,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
    }
    else if((data->node->inArrayType == rocfft_array_type_complex_planar
             || data->node->inArrayType == rocfft_array_type_hermitian_planar)
            && (data->node->outArrayType == rocfft_array_type_complex_planar
                || data->node->outArrayType == rocfft_array_type_hermitian_planar))
    {
        assert(scheme != 0);
        assert(scheme != 1);
        assert(scheme != 2);

        if(data->node->precision == rocfft_precision_single)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_P_P<float2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const real_type_t<float2>*)bufIn0,
                               (const real_type_t<float2>*)bufIn1,
                               (real_type_t<float2>*)bufOut0,
                               (real_type_t<float2>*)bufOut1,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device_P_P<double2>),
                               dim3(grid),
                               dim3(threads),
                               0,
                               rocfft_stream,
                               numof,
                               count,
                               N,
                               M,
                               (const real_type_t<double2>*)bufIn0,
                               (const real_type_t<double2>*)bufIn1,
                               (real_type_t<double2>*)bufOut0,
                               (real_type_t<double2>*)bufOut1,
                               data->node->length.size(),
                               kargs_lengths(data->node->devKernArg),
                               kargs_stride_in(data->node->devKernArg),
                               kargs_stride_out(data->node->devKernArg),
                               dir,
                               scheme);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported array type in bluestein kernel launch");
    }
}
