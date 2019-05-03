/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "transpose.h"
#include "kernel_launch.h"
#include "rocfft_hip.h"
#include <iostream>

/* ============================================================================================
 */

/*! \brief FFT Transpose out-of-place API

    \details
    transpose matrix A of size (m row by n cols) to matrix B (n row by m cols)
    both A and B are in row major

    @param[in]
    m         size_t.
    @param[in]
    n         size_t.
    @param[in]
    A     pointer storing batch_count of A matrix on the GPU.
    @param[inout]
    B    pointer storing batch_count of B matrix on the GPU.
    @param[in]
    count
              size_t
              number of matrices processed
    ********************************************************************/

template <typename T, int TRANSPOSE_DIM_X, int TRANSPOSE_DIM_Y>
rocfft_status rocfft_transpose_outofplace_template(size_t      m,
                                                   size_t      n,
                                                   const T*    A,
                                                   T*          B,
                                                   void*       twiddles_large,
                                                   size_t      count,
                                                   size_t      dim,
                                                   size_t*     lengths,
                                                   size_t*     stride_in,
                                                   size_t*     stride_out,
                                                   int         twl,
                                                   int         dir,
                                                   int         scheme,
                                                   hipStream_t rocfft_stream)
{

    dim3 grid((n - 1) / TRANSPOSE_DIM_X + 1, ((m - 1) / TRANSPOSE_DIM_X + 1), count);
    dim3 threads(TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, 1);

    bool noCorner = false;

    if((n % TRANSPOSE_DIM_X == 0)
       && (m % TRANSPOSE_DIM_X == 0)) // working threads match problem sizes, no corner cases
    {
        noCorner = true;
    }

    if(scheme == 0)
    {
        if(twl == 2)
        {
            if(dir == -1)
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         2,
                                                                         -1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         2,
                                                                         -1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
            else
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         2,
                                                                         1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         2,
                                                                         1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
        }
        else if(twl == 3)
        {
            if(dir == -1)
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         3,
                                                                         -1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         3,
                                                                         -1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
            else
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         3,
                                                                         1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         3,
                                                                         1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
        }
        else if(twl == 4)
        {
            if(dir == -1)
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         4,
                                                                         -1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         4,
                                                                         -1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
            else
            {
                if(noCorner)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         4,
                                                                         1,
                                                                         true>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_kernel2<T,
                                                                         TRANSPOSE_DIM_X,
                                                                         TRANSPOSE_DIM_Y,
                                                                         true,
                                                                         4,
                                                                         1,
                                                                         false>),
                                       dim3(grid),
                                       dim3(threads),
                                       0,
                                       rocfft_stream,
                                       A,
                                       B,
                                       (T*)twiddles_large,
                                       dim,
                                       lengths,
                                       stride_in,
                                       stride_out);
            }
        }
        else
        {
            if(noCorner)
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        transpose_kernel2<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, false, 0, 0, true>),
                    dim3(grid),
                    dim3(threads),
                    0,
                    rocfft_stream,
                    A,
                    B,
                    (T*)twiddles_large,
                    dim,
                    lengths,
                    stride_in,
                    stride_out);
            else
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        transpose_kernel2<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, false, 0, 0, false>),
                    dim3(grid),
                    dim3(threads),
                    0,
                    rocfft_stream,
                    A,
                    B,
                    (T*)twiddles_large,
                    dim,
                    lengths,
                    stride_in,
                    stride_out);
        }
    }
    else
    {
        if(noCorner)
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    transpose_kernel2_scheme<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, true>),
                dim3(grid),
                dim3(threads),
                0,
                rocfft_stream,
                A,
                B,
                (T*)twiddles_large,
                dim,
                lengths,
                stride_in,
                stride_out,
                scheme);
        else
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    transpose_kernel2_scheme<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, false>),
                dim3(grid),
                dim3(threads),
                0,
                rocfft_stream,
                A,
                B,
                (T*)twiddles_large,
                dim,
                lengths,
                stride_in,
                stride_out,
                scheme);
    }

    return rocfft_status_success;
}

/* ============================================================================================
 */

void rocfft_internal_transpose_var2(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t m = data->node->length[1];
    size_t n = data->node->length[0];

    int scheme = 0;
    if(data->node->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
    {
        scheme = 1;
        m      = data->node->length[2];
        n      = data->node->length[0] * data->node->length[1];
    }
    else if(data->node->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
    {
        scheme = 2;
        m      = data->node->length[1] * data->node->length[2];
        n      = data->node->length[0];
    }

    // size_t ld_in = data->node->inStride[1];
    // size_t ld_out = data->node->outStride[1];

    /*
  if (ld_in < m )
      return rocfft_status_invalid_dimensions;
  else if (ld_out < n )
      return rocfft_status_invalid_dimensions;

  if(m == 0 || n == 0 ) return rocfft_status_success;
  */

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
        twl = 0;

    int dir = data->node->direction;

    size_t count = data->node->batch;

    size_t extraDimStart = 2;
    if(scheme != 0)
        extraDimStart = 3;

    hipStream_t rocfft_stream = data->rocfft_stream;

    for(size_t i = extraDimStart; i < data->node->length.size(); i++)
        count *= data->node->length[i];

    if(data->node->precision == rocfft_precision_single)
        rocfft_transpose_outofplace_template<float2, 64, 16>(
            m,
            n,
            (const float2*)data->bufIn[0],
            (float2*)data->bufOut[0],
            data->node->twiddles_large,
            count,
            data->node->length.size(),
            data->node->devKernArg,
            data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,
            data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,
            twl,
            dir,
            scheme,
            rocfft_stream);
    else
        rocfft_transpose_outofplace_template<double2, 32, 32>(
            m,
            n,
            (const double2*)data->bufIn[0],
            (double2*)data->bufOut[0],
            data->node->twiddles_large,
            count,
            data->node->length.size(),
            data->node->devKernArg,
            data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,
            data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,
            twl,
            dir,
            scheme,
            rocfft_stream);
    // double2 must use 32 otherwise exceed the shared memory (LDS) size
}
