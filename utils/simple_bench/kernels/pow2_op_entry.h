/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_OP_ENTRY_HIP_H
#define POW2_OP_ENTRY_HIP_H

#include "common.h"
#include "pow2.h"

template <typename T, int dir>
__global__ void fft_2048_op_d2_s1(T*          twiddles,
                                  T*          buffer_i,
                                  T*          buffer_o,
                                  const ulong len,
                                  const ulong stride_i,
                                  const ulong stride_o,
                                  const ulong dist_i,
                                  const ulong dist_o)
{
    uint me    = hipThreadIdx_x;
    uint batch = hipBlockIdx_x;

    __shared__ real_type_t<T> lds[2048];

    T* lwb_i = buffer_i + (batch / len) * dist_i + (batch % len) * stride_i;
    T* lwb_o = buffer_o + (batch / len) * dist_o + (batch % len) * stride_o;

    fft_2048<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

template <typename T, int dir>
__global__ void fft_4096_op_d2_s1(T*          twiddles,
                                  T*          buffer_i,
                                  T*          buffer_o,
                                  const ulong len,
                                  const ulong stride_i,
                                  const ulong stride_o,
                                  const ulong dist_i,
                                  const ulong dist_o)
{
    uint me    = hipThreadIdx_x;
    uint batch = hipBlockIdx_x;

    __shared__ real_type_t<T> lds[4096];

    T* lwb_i = buffer_i + (batch / len) * dist_i + (batch % len) * stride_i;
    T* lwb_o = buffer_o + (batch / len) * dist_o + (batch % len) * stride_o;

    fft_4096<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

#endif // POW2_OP_ENTRY_HIP_H
