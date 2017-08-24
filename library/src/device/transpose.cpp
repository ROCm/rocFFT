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
    lda
              size_t
              specifies the leading dimension for the matrix A
    @param[inout]
    B    pointer storing batch_count of B matrix on the GPU.
    @param[in]
    ldb
              size_t
              specifies the leading dimension for the matrix B
    @param[in]
    batch_count
              size_t
              number of matrices processed
    ********************************************************************/


template<typename T, int TRANSPOSE_DIM_X, int TRANSPOSE_DIM_Y>
rocfft_status
rocfft_transpose_outofplace_template(size_t m, size_t n, const T* A, size_t lda, T* B, size_t ldb, size_t batch_count)
{

    if (lda < m )
        return rocfft_status_invalid_dimensions;
    else if (ldb < n )
        return rocfft_status_invalid_dimensions;

    if(m == 0 || n == 0 ) return rocfft_status_success;

    dim3 grid((m-1)/TRANSPOSE_DIM_X + 1, ( (n-1)/TRANSPOSE_DIM_X + 1 ), batch_count);
    dim3 threads(TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y, 1);

    hipStream_t rocfft_stream = 0;

    hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel2<T, TRANSPOSE_DIM_X, TRANSPOSE_DIM_Y>), dim3(grid), dim3(threads), 0, rocfft_stream, m, n, A, B, lda, ldb);

    return rocfft_status_success;

}

/* ============================================================================================ */


/*
rocfft_status
rocfft_transpose_complex_to_complex(rocfft_precision precision, size_t m, size_t n, const void* A, size_t lda, void* B, size_t ldb, size_t batch_count)
{
    if( precision == rocfft_precision_single)
        return rocfft_transpose_outofplace_template<float2, 64, 16>(m, n, (const float2*)A, lda, (float2*)B, ldb, batch_count);
    else
        return rocfft_transpose_outofplace_template<double2, 32, 32>(m, n, (const double2*)A, lda, (double2*)B, ldb, batch_count);//double2 must use 32 otherwise exceed the shared memory (LDS) size
}
*/

void rocfft_internal_transpose_var2(void *data_p, void *back_p)
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

    size_t m = data->node->length[1];
    size_t n = data->node->length[0];
    size_t lda = data->node->inStride[1]; 
    size_t ldb = data->node->outStride[1];
    size_t batch_count = data->node->batch;

    if( data->node->precision == rocfft_precision_single)
        rocfft_transpose_outofplace_template<float2, 64, 16>(m, n, (const float2 *)data->bufIn[0], lda, (float2 *)data->bufOut[0], ldb, batch_count);
    else
        rocfft_transpose_outofplace_template<double2, 32, 32>(m, n, (const double2 *)data->bufIn[0], lda, (double2 *)data->bufOut[0], ldb, batch_count);//double2 must use 32 otherwise exceed the shared memory (LDS) size

/*    (float2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);*/
}

