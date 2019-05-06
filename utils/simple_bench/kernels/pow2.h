/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_HIP_H
#define POW2_HIP_H

#include "butterfly.h"
#include "common.h"

template <typename T, StrideBin sb, int dir>
__device__ void fft_2048(T*              twiddles,
                         T*              lwb_in,
                         T*              lwb_out,
                         real_type_t<T>* lds,
                         const uint      me,
                         const ulong     stride_in,
                         const ulong     stride_out)
{
    T X0, X1, X2, X3, X4, X5, X6, X7;

    if(sb == SB_UNIT)
    {
        X0 = lwb_in[me + 0];
        X1 = lwb_in[me + 256];
        X2 = lwb_in[me + 512];
        X3 = lwb_in[me + 768];
        X4 = lwb_in[me + 1024];
        X5 = lwb_in[me + 1280];
        X6 = lwb_in[me + 1536];
        X7 = lwb_in[me + 1792];
    }
    else
    {
        X0 = lwb_in[(me + 0) * stride_in];
        X1 = lwb_in[(me + 256) * stride_in];
        X2 = lwb_in[(me + 512) * stride_in];
        X3 = lwb_in[(me + 768) * stride_in];
        X4 = lwb_in[(me + 1024) * stride_in];
        X5 = lwb_in[(me + 1280) * stride_in];
        X6 = lwb_in[(me + 1536) * stride_in];
        X7 = lwb_in[(me + 1792) * stride_in];
    }

    if(dir == -1)
        FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
    else
        InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

    lds[me * 8 + 0] = X0.x;
    lds[me * 8 + 1] = X1.x;
    lds[me * 8 + 2] = X2.x;
    lds[me * 8 + 3] = X3.x;
    lds[me * 8 + 4] = X4.x;
    lds[me * 8 + 5] = X5.x;
    lds[me * 8 + 6] = X6.x;
    lds[me * 8 + 7] = X7.x;

    __syncthreads();

    X0.x = lds[me + 0];
    X1.x = lds[me + 256];
    X2.x = lds[me + 512];
    X3.x = lds[me + 768];
    X4.x = lds[me + 1024];
    X5.x = lds[me + 1280];
    X6.x = lds[me + 1536];
    X7.x = lds[me + 1792];

    __syncthreads();

    lds[me * 8 + 0] = X0.y;
    lds[me * 8 + 1] = X1.y;
    lds[me * 8 + 2] = X2.y;
    lds[me * 8 + 3] = X3.y;
    lds[me * 8 + 4] = X4.y;
    lds[me * 8 + 5] = X5.y;
    lds[me * 8 + 6] = X6.y;
    lds[me * 8 + 7] = X7.y;

    __syncthreads();

    X0.y = lds[me + 0];
    X1.y = lds[me + 256];
    X2.y = lds[me + 512];
    X3.y = lds[me + 768];
    X4.y = lds[me + 1024];
    X5.y = lds[me + 1280];
    X6.y = lds[me + 1536];
    X7.y = lds[me + 1792];

    __syncthreads();

    if(dir == -1)
    {
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 0, X1)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 1, X2)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 2, X3)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 3, X4)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 4, X5)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 5, X6)
        TWIDDLE_MUL_FWD(twiddles, 7 + 7 * (me % 8) + 6, X7)
    }
    else
    {
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 0, X1)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 1, X2)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 2, X3)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 3, X4)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 4, X5)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 5, X6)
        TWIDDLE_MUL_INV(twiddles, 7 + 7 * (me % 8) + 6, X7)
    }

    if(dir == -1)
        FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
    else
        InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

    lds[(me / 8) * 64 + (me % 8) + 0]  = X0.x;
    lds[(me / 8) * 64 + (me % 8) + 8]  = X1.x;
    lds[(me / 8) * 64 + (me % 8) + 16] = X2.x;
    lds[(me / 8) * 64 + (me % 8) + 24] = X3.x;
    lds[(me / 8) * 64 + (me % 8) + 32] = X4.x;
    lds[(me / 8) * 64 + (me % 8) + 40] = X5.x;
    lds[(me / 8) * 64 + (me % 8) + 48] = X6.x;
    lds[(me / 8) * 64 + (me % 8) + 56] = X7.x;

    __syncthreads();

    X0.x = lds[me + 0];
    X1.x = lds[me + 256];
    X2.x = lds[me + 512];
    X3.x = lds[me + 768];
    X4.x = lds[me + 1024];
    X5.x = lds[me + 1280];
    X6.x = lds[me + 1536];
    X7.x = lds[me + 1792];

    __syncthreads();

    lds[(me / 8) * 64 + (me % 8) + 0]  = X0.y;
    lds[(me / 8) * 64 + (me % 8) + 8]  = X1.y;
    lds[(me / 8) * 64 + (me % 8) + 16] = X2.y;
    lds[(me / 8) * 64 + (me % 8) + 24] = X3.y;
    lds[(me / 8) * 64 + (me % 8) + 32] = X4.y;
    lds[(me / 8) * 64 + (me % 8) + 40] = X5.y;
    lds[(me / 8) * 64 + (me % 8) + 48] = X6.y;
    lds[(me / 8) * 64 + (me % 8) + 56] = X7.y;

    __syncthreads();

    X0.y = lds[me + 0];
    X1.y = lds[me + 256];
    X2.y = lds[me + 512];
    X3.y = lds[me + 768];
    X4.y = lds[me + 1024];
    X5.y = lds[me + 1280];
    X6.y = lds[me + 1536];
    X7.y = lds[me + 1792];

    __syncthreads();

    if(dir == -1)
    {
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 0, X1)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 1, X2)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 2, X3)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 3, X4)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 4, X5)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 5, X6)
        TWIDDLE_MUL_FWD(twiddles, 63 + 7 * (me % 64) + 6, X7)
    }
    else
    {
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 0, X1)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 1, X2)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 2, X3)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 3, X4)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 4, X5)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 5, X6)
        TWIDDLE_MUL_INV(twiddles, 63 + 7 * (me % 64) + 6, X7)
    }

    if(dir == -1)
        FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
    else
        InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

    lds[(me / 64) * 512 + (me % 64) + 0]   = X0.x;
    lds[(me / 64) * 512 + (me % 64) + 64]  = X1.x;
    lds[(me / 64) * 512 + (me % 64) + 128] = X2.x;
    lds[(me / 64) * 512 + (me % 64) + 192] = X3.x;
    lds[(me / 64) * 512 + (me % 64) + 256] = X4.x;
    lds[(me / 64) * 512 + (me % 64) + 320] = X5.x;
    lds[(me / 64) * 512 + (me % 64) + 384] = X6.x;
    lds[(me / 64) * 512 + (me % 64) + 448] = X7.x;

    __syncthreads();

    X0.x = lds[(2 * me + 0) + 0];
    X1.x = lds[(2 * me + 0) + 512];
    X2.x = lds[(2 * me + 0) + 1024];
    X3.x = lds[(2 * me + 0) + 1536];

    X4.x = lds[(2 * me + 1) + 0];
    X5.x = lds[(2 * me + 1) + 512];
    X6.x = lds[(2 * me + 1) + 1024];
    X7.x = lds[(2 * me + 1) + 1536];

    __syncthreads();

    lds[(me / 64) * 512 + (me % 64) + 0]   = X0.y;
    lds[(me / 64) * 512 + (me % 64) + 64]  = X1.y;
    lds[(me / 64) * 512 + (me % 64) + 128] = X2.y;
    lds[(me / 64) * 512 + (me % 64) + 192] = X3.y;
    lds[(me / 64) * 512 + (me % 64) + 256] = X4.y;
    lds[(me / 64) * 512 + (me % 64) + 320] = X5.y;
    lds[(me / 64) * 512 + (me % 64) + 384] = X6.y;
    lds[(me / 64) * 512 + (me % 64) + 448] = X7.y;

    __syncthreads();

    X0.y = lds[(2 * me + 0) + 0];
    X1.y = lds[(2 * me + 0) + 512];
    X2.y = lds[(2 * me + 0) + 1024];
    X3.y = lds[(2 * me + 0) + 1536];

    X4.y = lds[(2 * me + 1) + 0];
    X5.y = lds[(2 * me + 1) + 512];
    X6.y = lds[(2 * me + 1) + 1024];
    X7.y = lds[(2 * me + 1) + 1536];

    __syncthreads();

    if(dir == -1)
    {
        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 0, X1)
        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 1, X2)
        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 2, X3)

        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 0, X5)
        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 1, X6)
        TWIDDLE_MUL_FWD(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 2, X7)
    }
    else
    {
        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 0, X1)
        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 1, X2)
        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 0) % 512) + 2, X3)

        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 0, X5)
        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 1, X6)
        TWIDDLE_MUL_INV(twiddles, 511 + 3 * ((2 * me + 1) % 512) + 2, X7)
    }

    if(dir == -1)
    {
        FwdRad4<T>(&X0, &X1, &X2, &X3);
        FwdRad4<T>(&X4, &X5, &X6, &X7);
    }
    else
    {
        InvRad4<T>(&X0, &X1, &X2, &X3);
        InvRad4<T>(&X4, &X5, &X6, &X7);
    }

    if(sb == SB_UNIT)
    {
        vector4_type_t<T>* lwbv = (vector4_type_t<T>*)lwb_out;
        lwbv[me + 0]            = lib_make_vector4<vector4_type_t<T>>(X0.x, X0.y, X4.x, X4.y);
        lwbv[me + 256]          = lib_make_vector4<vector4_type_t<T>>(X1.x, X1.y, X5.x, X5.y);
        lwbv[me + 512]          = lib_make_vector4<vector4_type_t<T>>(X2.x, X2.y, X6.x, X6.y);
        lwbv[me + 768]          = lib_make_vector4<vector4_type_t<T>>(X3.x, X3.y, X7.x, X7.y);
    }
    else
    {
        lwb_out[(me + 0) * stride_out]    = X0;
        lwb_out[(me + 256) * stride_out]  = X1;
        lwb_out[(me + 512) * stride_out]  = X2;
        lwb_out[(me + 768) * stride_out]  = X3;
        lwb_out[(me + 1024) * stride_out] = X4;
        lwb_out[(me + 1280) * stride_out] = X5;
        lwb_out[(me + 1536) * stride_out] = X6;
        lwb_out[(me + 1792) * stride_out] = X7;
    }
}

template <typename T, StrideBin sb, int dir>
__device__ void fft_4096(T*              twiddles,
                         T*              lwb_in,
                         T*              lwb_out,
                         real_type_t<T>* lds,
                         const uint      me,
                         const ulong     stride_in,
                         const ulong     stride_out)
{
    T X0, X1, X2, X3, X4, X5, X6, X7;
    T X8, X9, X10, X11, X12, X13, X14, X15;

    if(sb == SB_UNIT)
    {
        X0  = lwb_in[me + 0];
        X1  = lwb_in[me + 256];
        X2  = lwb_in[me + 512];
        X3  = lwb_in[me + 768];
        X4  = lwb_in[me + 1024];
        X5  = lwb_in[me + 1280];
        X6  = lwb_in[me + 1536];
        X7  = lwb_in[me + 1792];
        X8  = lwb_in[me + 2048];
        X9  = lwb_in[me + 2304];
        X10 = lwb_in[me + 2560];
        X11 = lwb_in[me + 2816];
        X12 = lwb_in[me + 3072];
        X13 = lwb_in[me + 3328];
        X14 = lwb_in[me + 3584];
        X15 = lwb_in[me + 3840];
    }
    else
    {
        X0  = lwb_in[(me + 0) * stride_in];
        X1  = lwb_in[(me + 256) * stride_in];
        X2  = lwb_in[(me + 512) * stride_in];
        X3  = lwb_in[(me + 768) * stride_in];
        X4  = lwb_in[(me + 1024) * stride_in];
        X5  = lwb_in[(me + 1280) * stride_in];
        X6  = lwb_in[(me + 1536) * stride_in];
        X7  = lwb_in[(me + 1792) * stride_in];
        X8  = lwb_in[(me + 2048) * stride_in];
        X9  = lwb_in[(me + 2304) * stride_in];
        X10 = lwb_in[(me + 2560) * stride_in];
        X11 = lwb_in[(me + 2816) * stride_in];
        X12 = lwb_in[(me + 3072) * stride_in];
        X13 = lwb_in[(me + 3328) * stride_in];
        X14 = lwb_in[(me + 3584) * stride_in];
        X15 = lwb_in[(me + 3840) * stride_in];
    }

    if(dir == -1)
    {
        FwdRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }
    else
    {
        InvRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }

    lds[me * 16 + 0]  = X0.x;
    lds[me * 16 + 1]  = X1.x;
    lds[me * 16 + 2]  = X2.x;
    lds[me * 16 + 3]  = X3.x;
    lds[me * 16 + 4]  = X4.x;
    lds[me * 16 + 5]  = X5.x;
    lds[me * 16 + 6]  = X6.x;
    lds[me * 16 + 7]  = X7.x;
    lds[me * 16 + 8]  = X8.x;
    lds[me * 16 + 9]  = X9.x;
    lds[me * 16 + 10] = X10.x;
    lds[me * 16 + 11] = X11.x;
    lds[me * 16 + 12] = X12.x;
    lds[me * 16 + 13] = X13.x;
    lds[me * 16 + 14] = X14.x;
    lds[me * 16 + 15] = X15.x;

    __syncthreads();

    X0.x  = lds[me + 0];
    X1.x  = lds[me + 256];
    X2.x  = lds[me + 512];
    X3.x  = lds[me + 768];
    X4.x  = lds[me + 1024];
    X5.x  = lds[me + 1280];
    X6.x  = lds[me + 1536];
    X7.x  = lds[me + 1792];
    X8.x  = lds[me + 2048];
    X9.x  = lds[me + 2304];
    X10.x = lds[me + 2560];
    X11.x = lds[me + 2816];
    X12.x = lds[me + 3072];
    X13.x = lds[me + 3328];
    X14.x = lds[me + 3584];
    X15.x = lds[me + 3840];

    __syncthreads();

    lds[me * 16 + 0]  = X0.y;
    lds[me * 16 + 1]  = X1.y;
    lds[me * 16 + 2]  = X2.y;
    lds[me * 16 + 3]  = X3.y;
    lds[me * 16 + 4]  = X4.y;
    lds[me * 16 + 5]  = X5.y;
    lds[me * 16 + 6]  = X6.y;
    lds[me * 16 + 7]  = X7.y;
    lds[me * 16 + 8]  = X8.y;
    lds[me * 16 + 9]  = X9.y;
    lds[me * 16 + 10] = X10.y;
    lds[me * 16 + 11] = X11.y;
    lds[me * 16 + 12] = X12.y;
    lds[me * 16 + 13] = X13.y;
    lds[me * 16 + 14] = X14.y;
    lds[me * 16 + 15] = X15.y;

    __syncthreads();

    X0.y  = lds[me + 0];
    X1.y  = lds[me + 256];
    X2.y  = lds[me + 512];
    X3.y  = lds[me + 768];
    X4.y  = lds[me + 1024];
    X5.y  = lds[me + 1280];
    X6.y  = lds[me + 1536];
    X7.y  = lds[me + 1792];
    X8.y  = lds[me + 2048];
    X9.y  = lds[me + 2304];
    X10.y = lds[me + 2560];
    X11.y = lds[me + 2816];
    X12.y = lds[me + 3072];
    X13.y = lds[me + 3328];
    X14.y = lds[me + 3584];
    X15.y = lds[me + 3840];

    __syncthreads();

    if(dir == -1)
    {
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 0, X1)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 1, X2)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 2, X3)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 3, X4)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 4, X5)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 5, X6)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 6, X7)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 7, X8)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 8, X9)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 9, X10)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 10, X11)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 11, X12)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 12, X13)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 13, X14)
        TWIDDLE_MUL_FWD(twiddles, 15 + 15 * (me % 16) + 14, X15)
    }
    else
    {
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 0, X1)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 1, X2)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 2, X3)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 3, X4)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 4, X5)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 5, X6)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 6, X7)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 7, X8)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 8, X9)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 9, X10)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 10, X11)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 11, X12)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 12, X13)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 13, X14)
        TWIDDLE_MUL_INV(twiddles, 15 + 15 * (me % 16) + 14, X15)
    }

    if(dir == -1)
    {
        FwdRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }
    else
    {
        InvRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }

    lds[(me / 16) * 256 + (me % 16) + 0]   = X0.x;
    lds[(me / 16) * 256 + (me % 16) + 16]  = X1.x;
    lds[(me / 16) * 256 + (me % 16) + 32]  = X2.x;
    lds[(me / 16) * 256 + (me % 16) + 48]  = X3.x;
    lds[(me / 16) * 256 + (me % 16) + 64]  = X4.x;
    lds[(me / 16) * 256 + (me % 16) + 80]  = X5.x;
    lds[(me / 16) * 256 + (me % 16) + 96]  = X6.x;
    lds[(me / 16) * 256 + (me % 16) + 112] = X7.x;
    lds[(me / 16) * 256 + (me % 16) + 128] = X8.x;
    lds[(me / 16) * 256 + (me % 16) + 144] = X9.x;
    lds[(me / 16) * 256 + (me % 16) + 160] = X10.x;
    lds[(me / 16) * 256 + (me % 16) + 176] = X11.x;
    lds[(me / 16) * 256 + (me % 16) + 192] = X12.x;
    lds[(me / 16) * 256 + (me % 16) + 208] = X13.x;
    lds[(me / 16) * 256 + (me % 16) + 224] = X14.x;
    lds[(me / 16) * 256 + (me % 16) + 240] = X15.x;

    __syncthreads();

    X0.x  = lds[me + 0];
    X1.x  = lds[me + 256];
    X2.x  = lds[me + 512];
    X3.x  = lds[me + 768];
    X4.x  = lds[me + 1024];
    X5.x  = lds[me + 1280];
    X6.x  = lds[me + 1536];
    X7.x  = lds[me + 1792];
    X8.x  = lds[me + 2048];
    X9.x  = lds[me + 2304];
    X10.x = lds[me + 2560];
    X11.x = lds[me + 2816];
    X12.x = lds[me + 3072];
    X13.x = lds[me + 3328];
    X14.x = lds[me + 3584];
    X15.x = lds[me + 3840];

    __syncthreads();

    lds[(me / 16) * 256 + (me % 16) + 0]   = X0.y;
    lds[(me / 16) * 256 + (me % 16) + 16]  = X1.y;
    lds[(me / 16) * 256 + (me % 16) + 32]  = X2.y;
    lds[(me / 16) * 256 + (me % 16) + 48]  = X3.y;
    lds[(me / 16) * 256 + (me % 16) + 64]  = X4.y;
    lds[(me / 16) * 256 + (me % 16) + 80]  = X5.y;
    lds[(me / 16) * 256 + (me % 16) + 96]  = X6.y;
    lds[(me / 16) * 256 + (me % 16) + 112] = X7.y;
    lds[(me / 16) * 256 + (me % 16) + 128] = X8.y;
    lds[(me / 16) * 256 + (me % 16) + 144] = X9.y;
    lds[(me / 16) * 256 + (me % 16) + 160] = X10.y;
    lds[(me / 16) * 256 + (me % 16) + 176] = X11.y;
    lds[(me / 16) * 256 + (me % 16) + 192] = X12.y;
    lds[(me / 16) * 256 + (me % 16) + 208] = X13.y;
    lds[(me / 16) * 256 + (me % 16) + 224] = X14.y;
    lds[(me / 16) * 256 + (me % 16) + 240] = X15.y;

    __syncthreads();

    X0.y  = lds[me + 0];
    X1.y  = lds[me + 256];
    X2.y  = lds[me + 512];
    X3.y  = lds[me + 768];
    X4.y  = lds[me + 1024];
    X5.y  = lds[me + 1280];
    X6.y  = lds[me + 1536];
    X7.y  = lds[me + 1792];
    X8.y  = lds[me + 2048];
    X9.y  = lds[me + 2304];
    X10.y = lds[me + 2560];
    X11.y = lds[me + 2816];
    X12.y = lds[me + 3072];
    X13.y = lds[me + 3328];
    X14.y = lds[me + 3584];
    X15.y = lds[me + 3840];

    __syncthreads();

    if(dir == -1)
    {
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 0, X1)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 1, X2)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 2, X3)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 3, X4)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 4, X5)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 5, X6)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 6, X7)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 7, X8)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 8, X9)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 9, X10)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 10, X11)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 11, X12)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 12, X13)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 13, X14)
        TWIDDLE_MUL_FWD(twiddles, 255 + 15 * (me % 256) + 14, X15)
    }
    else
    {
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 0, X1)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 1, X2)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 2, X3)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 3, X4)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 4, X5)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 5, X6)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 6, X7)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 7, X8)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 8, X9)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 9, X10)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 10, X11)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 11, X12)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 12, X13)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 13, X14)
        TWIDDLE_MUL_INV(twiddles, 255 + 15 * (me % 256) + 14, X15)
    }

    if(dir == -1)
    {
        FwdRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }
    else
    {
        InvRad16<T>(
            &X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
    }

    if(sb == SB_UNIT)
    {
        lwb_out[me + 0]    = X0;
        lwb_out[me + 256]  = X1;
        lwb_out[me + 512]  = X2;
        lwb_out[me + 768]  = X3;
        lwb_out[me + 1024] = X4;
        lwb_out[me + 1280] = X5;
        lwb_out[me + 1536] = X6;
        lwb_out[me + 1792] = X7;
        lwb_out[me + 2048] = X8;
        lwb_out[me + 2304] = X9;
        lwb_out[me + 2560] = X10;
        lwb_out[me + 2816] = X11;
        lwb_out[me + 3072] = X12;
        lwb_out[me + 3328] = X13;
        lwb_out[me + 3584] = X14;
        lwb_out[me + 3840] = X15;
    }
    else
    {
        lwb_out[(me + 0) * stride_out]    = X0;
        lwb_out[(me + 256) * stride_out]  = X1;
        lwb_out[(me + 512) * stride_out]  = X2;
        lwb_out[(me + 768) * stride_out]  = X3;
        lwb_out[(me + 1024) * stride_out] = X4;
        lwb_out[(me + 1280) * stride_out] = X5;
        lwb_out[(me + 1536) * stride_out] = X6;
        lwb_out[(me + 1792) * stride_out] = X7;
        lwb_out[(me + 2048) * stride_out] = X8;
        lwb_out[(me + 2304) * stride_out] = X9;
        lwb_out[(me + 2560) * stride_out] = X10;
        lwb_out[(me + 2816) * stride_out] = X11;
        lwb_out[(me + 3072) * stride_out] = X12;
        lwb_out[(me + 3328) * stride_out] = X13;
        lwb_out[(me + 3584) * stride_out] = X14;
        lwb_out[(me + 3840) * stride_out] = X15;
    }
}

#endif // POW2_HIP_H
