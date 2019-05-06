/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_IP_ENTRY_HIP_H
#define POW2_IP_ENTRY_HIP_H

#include "common.h"
#include "pow2.h"

template <typename T, int dir>
__global__ void
    fft_2048_ip_d2_s1(T* twiddles, T* buffer, const ulong len, const ulong stride, const ulong dist)
{
    uint me    = hipThreadIdx_x;
    uint batch = hipBlockIdx_x;

    __shared__ real_type_t<T> lds[2048];

    T* lwb = buffer + (batch / len) * dist + (batch % len) * stride;

    fft_2048<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

template <typename T, int dir>
__global__ void
    fft_4096_ip_d2_s1(T* twiddles, T* buffer, const ulong len, const ulong stride, const ulong dist)
{
    uint me    = hipThreadIdx_x;
    uint batch = hipBlockIdx_x;

    __shared__ real_type_t<T> lds[4096];

    T* lwb = buffer + (batch / len) * dist + (batch % len) * stride;

    fft_4096<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

#endif // POW2_IP_ENTRY_HIP_H
