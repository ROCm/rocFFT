
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "bluestein.h"
#include "kernel_launch.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename T>
rocfft_status chirp_launch(
    size_t N, size_t M, T* B, void* twiddles_large, int twl, int dir, hipStream_t rocfft_stream)
{
    dim3 grid((M - N) / 64 + 1);
    dim3 threads(64);

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

void rocfft_internal_chirp(const void* data_p, void* back_p)
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
    else if(data->node->large1D > (size_t)256)
        twl = 2;
    else
        twl = 1;

    int dir = data->node->direction;

    hipStream_t rocfft_stream = data->rocfft_stream;

    if(data->node->precision == rocfft_precision_single)
        chirp_launch<float2>(
            N, M, (float2*)data->bufOut[0], data->node->twiddles_large, twl, dir, rocfft_stream);
    else
        chirp_launch<double2>(
            N, M, (double2*)data->bufOut[0], data->node->twiddles_large, twl, dir, rocfft_stream);
}

template <typename T>
rocfft_status mul_launch(size_t      numof,
                         size_t      totalWI,
                         size_t      N,
                         size_t      M,
                         const T*    A,
                         T*          B,
                         size_t      dim,
                         size_t*     lengths,
                         size_t*     stride_in,
                         size_t*     stride_out,
                         int         dir,
                         int         scheme,
                         hipStream_t rocfft_stream)
{

    dim3 grid((totalWI - 1) / 64 + 1);
    dim3 threads(64);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_device<T>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       rocfft_stream,
                       numof,
                       totalWI,
                       N,
                       M,
                       A,
                       B,
                       dim,
                       lengths,
                       stride_in,
                       stride_out,
                       dir,
                       scheme);

    return rocfft_status_success;
}

void rocfft_internal_mul(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t N = data->node->length[0];
    size_t M = data->node->lengthBlue;

    int scheme = 0; // fft mul
    if(data->node->scheme == CS_KERNEL_PAD_MUL)
    {
        scheme = 1; // pad mul
    }
    else if(data->node->scheme == CS_KERNEL_RES_MUL)
    {
        scheme = 2; // res mul
    }

    size_t cBytes;
    if(data->node->precision == rocfft_precision_single)
    {
        cBytes = sizeof(float) * 2;
    }
    else
    {
        cBytes = sizeof(double) * 2;
    }

    void* bufIn  = data->bufIn[0];
    void* bufOut = data->bufOut[0];

    size_t numof = 0;
    if(scheme == 0)
    {
        bufIn  = ((char*)bufIn + M * cBytes);
        bufOut = ((char*)bufOut + 2 * M * cBytes);

        numof = M;
    }
    else if(scheme == 1)
    {
        bufOut = ((char*)bufOut + M * cBytes);

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

    if(data->node->precision == rocfft_precision_single)
        mul_launch<float2>(numof,
                           count,
                           N,
                           M,
                           (const float2*)bufIn,
                           (float2*)bufOut,
                           data->node->length.size(),
                           data->node->devKernArg,
                           data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,
                           data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,
                           dir,
                           scheme,
                           rocfft_stream);
    else
        mul_launch<double2>(numof,
                            count,
                            N,
                            M,
                            (const double2*)bufIn,
                            (double2*)bufOut,
                            data->node->length.size(),
                            data->node->devKernArg,
                            data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,
                            data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,
                            dir,
                            scheme,
                            rocfft_stream);
}
