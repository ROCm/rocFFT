/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <iostream>
#include "kernel_launch.h"
#include "./kernels/common.h"
#include "rocfft_hip.h"
#include "./kernels/transpose.h"

/* ============================================================================================ */

/*! \brief FFT Transpose out-of-place API

    \details
    transpose matrix A of size (m by n) to matrix B (n by m)

    @param[in]
    m         size_t.
    @param[in]
    n         size_t.
    @param[in]
    A     pointer storing batch_count of A matrix on the GPU.
    @param[in]
    ld_in
              size_t
              specifies the leading dimension for the matrix A
    @param[inout]
    B    pointer storing batch_count of B matrix on the GPU.
    @param[in]
    ld_out
              size_t
              specifies the leading dimension for the matrix B
    @param[in]
    batch_count
              size_t
              number of matrices processed
    ********************************************************************/


template<typename T, int TRANSPOSE_DIM_X, int TRANSPOSE_DIM_Y>
rocfft_status
rocfft_transpose_outofplace_template(size_t m, size_t n, const T* A, T* B, void *twiddles_large, size_t count, size_t dim, size_t *lengths, size_t *stride_in, size_t *stride_out, int twl, int dir, int scheme)
{

    dim3 grid((n-1)/TRANSPOSE_DIM_X + 1, ( (m-1)/TRANSPOSE_DIM_X + 1 ), count);
    dim3 threads(TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, 1);

    hipStream_t rocfft_stream = 0;

    if(scheme == 0)
    {
        hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel2<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y>), dim3(grid), dim3(threads), 0, rocfft_stream,
                A, B, (T *)twiddles_large, dim, lengths, stride_in, stride_out, twl, dir);
    }
    else
    {
        hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel2_scheme<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y>), dim3(grid), dim3(threads), 0, rocfft_stream,
                A, B, (T *)twiddles_large, dim, lengths, stride_in, stride_out, scheme);
    }

    return rocfft_status_success;

}

/* ============================================================================================ */


/*
extern "C"
rocfft_status
rocfft_transpose_complex_to_complex(rocfft_precision precision, size_t m, size_t n, const void* A, size_t ld_in, void* B, size_t ld_out, size_t batch_count)
{
    if( precision == rocfft_precision_single)
        return rocfft_transpose_outofplace_template<float2, 64, 16>(m, n, (const float2*)A, ld_in, (float2*)B, ld_out, batch_count);
    else
        return rocfft_transpose_outofplace_template<double2, 32, 32>(m, n, (const double2*)A, ld_in, (double2*)B, ld_out, batch_count);//double2 must use 32 otherwise exceed the shared memory (LDS) size
}
*/

void rocfft_internal_transpose_var2(const void *data_p, void *back_p)
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

    size_t m = data->node->length[1];
    size_t n = data->node->length[0];

    int scheme = 0;
    if(data->node->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
    {
        scheme = 1;
        m = data->node->length[2];
        n = data->node->length[0] * data->node->length[1];
    }
    else if(data->node->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
    {
        scheme = 2;
        m = data->node->length[1] * data->node->length[2];
        n = data->node->length[0];
    }

    //size_t ld_in = data->node->inStride[1]; 
    //size_t ld_out = data->node->outStride[1];

    /*
    if (ld_in < m )
        return rocfft_status_invalid_dimensions;
    else if (ld_out < n )
        return rocfft_status_invalid_dimensions;

    if(m == 0 || n == 0 ) return rocfft_status_success;
    */

    int twl = 0;

         if(data->node->large1D > (size_t)256*256*256*256) printf("large1D twiddle size too large error");
    else if(data->node->large1D > (size_t)256*256*256) twl = 4;
    else if(data->node->large1D > (size_t)256*256) twl = 3;
    else if(data->node->large1D > (size_t)256) twl = 2;
    else twl = 0;

    int dir = data->node->direction;

    size_t count = data->node->batch;

    size_t extraDimStart = 2;
    if(scheme != 0)
        extraDimStart = 3;

    for(size_t i=extraDimStart; i<data->node->length.size(); i++) count *= data->node->length[i];

    if( data->node->precision == rocfft_precision_single)
        rocfft_transpose_outofplace_template<float2, 64, 16>(m, n, (const float2 *)data->bufIn[0], (float2 *)data->bufOut[0], data->node->twiddles_large, count,
                data->node->length.size(), data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH, twl, dir, scheme);
    else
        rocfft_transpose_outofplace_template<double2, 32, 32>(m, n, (const double2 *)data->bufIn[0], (double2 *)data->bufOut[0], data->node->twiddles_large, count,
                data->node->length.size(), data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH, twl, dir, scheme);
            //double2 must use 32 otherwise exceed the shared memory (LDS) size

}



/* ============================================================================================ */

#define TRANSPOSE_CALL(NUM_Y,DRN,TWL,TTD)    \
hipLaunchKernel(HIP_KERNEL_NAME( transpose_var1<float2, DRN,TWL,TTD> ),\
                    dim3(data->gridParam.b_x, data->gridParam.b_y), dim3(data->gridParam.tpb_x, data->gridParam.tpb_x), 0, 0,\
                    (float2 *)data->node->twiddles_large, (float2 *)data->bufIn[0], (float2 *)data->bufOut[0],\
                    NUM_Y, data->node->inStride[1], data->node->outStride[1], data->node->iDist, data->node->oDist)

void rocfft_internal_transpose_var1_sp(const void *data_p, void *back_p)
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;
    if(data->node->transTileDir == TTD_IP_HOR)
    {
        if(data->node->large1D == 0)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 0, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 0, TTD_IP_HOR);
        }
        else if(data->node->large1D <= 16777216)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 3, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 3, TTD_IP_HOR);
        }
        else
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 4, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 4, TTD_IP_HOR);
        }
    }
    else
    {
        if(data->node->large1D == 0)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 0, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 0, TTD_IP_VER);
        }
        else if(data->node->large1D <= 16777216)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 3, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 3, TTD_IP_VER);
        }
        else
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 4, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 4, TTD_IP_VER);
        }
    }
}


