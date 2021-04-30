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

#ifndef BLUESTEIN_H
#define BLUESTEIN_H

#include "callback.h"
#include "common.h"
#include "rocfft_hip.h"

static const unsigned int LAUNCH_BOUNDS_BLUESTEIN_KERNEL = 64;

template <typename T>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_BLUESTEIN_KERNEL) chirp_device(
    const size_t N, const size_t M, T* output, T* twiddles_large, const int twl, const int dir)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    T val = lib_make_vector2<T>(0, 0);

    if(twl == 1)
        val = TWLstep1(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 2)
        val = TWLstep2(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 3)
        val = TWLstep3(twiddles_large, (tx * tx) % (2 * N));
    else if(twl == 4)
        val = TWLstep4(twiddles_large, (tx * tx) % (2 * N));

    val.y *= (real_type_t<T>)(dir);

    if(tx == 0)
    {
        output[tx]     = val;
        output[tx + M] = val;
    }
    else if(tx < N)
    {
        output[tx]     = val;
        output[tx + M] = val;

        output[M - tx]     = val;
        output[M - tx + M] = val;
    }
    else if(tx <= (M - N))
    {
        output[tx]     = lib_make_vector2<T>(0, 0);
        output[tx + M] = lib_make_vector2<T>(0, 0);
    }
}

// mul_device takes care of fft_mul, pad_mul, and res_mul, which
// are 3 steps in Bluestein algorithm. And In the below, we have
// 4 similar functions to support interleaved and planar format.

template <typename T, CallbackType cbtype>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_BLUESTEIN_KERNEL)
    mul_device_I_I(const size_t  numof,
                   const size_t  totalWI,
                   const size_t  N,
                   const size_t  M,
                   const T*      input,
                   T*            output,
                   const size_t  dim,
                   const size_t* lengths,
                   const size_t* stride_in,
                   const size_t* stride_out,
                   const int     dir,
                   const int     scheme,
                   void* __restrict__ load_cb_fn,
                   void* __restrict__ load_cb_data,
                   uint32_t load_cb_lds_bytes,
                   void* __restrict__ store_cb_fn,
                   void* __restrict__ store_cb_data)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx          = tx % numof;
    size_t iIdx = tx * stride_in[0];
    size_t oIdx = tx * stride_out[0];

    auto load_cb  = get_load_cb<T, cbtype>(load_cb_fn);
    auto store_cb = get_store_cb<T, cbtype>(store_cb_fn);
    if(scheme == 0)
    {
        // FFT_MUL is in the middle of bluestein and should never be
        // the first/last kernel to read/write global memory.  So we
        // don't need to run callbacks.
        output += oOffset;

        T out          = output[oIdx];
        output[oIdx].x = input[iIdx].x * out.x - input[iIdx].y * out.y;
        output[oIdx].y = input[iIdx].x * out.y + input[iIdx].y * out.x;
    }
    else if(scheme == 1)
    {
        // PAD_MUL is the first non-chirp step of bluestein and
        // should never be the last kernel to write global memory.
        // So we should never need to run a "store" callback.

        T* chirp = output;

        iIdx += iOffset;

        oIdx += M;
        oIdx += oOffset;

        if(tx < N)
        {
            // callback might modify input, but otherwise it's const
            T in_elem      = load_cb(const_cast<T*>(input), iIdx, load_cb_data, nullptr);
            output[oIdx].x = in_elem.x * chirp[tx].x + in_elem.y * chirp[tx].y;
            output[oIdx].y = -in_elem.x * chirp[tx].y + in_elem.y * chirp[tx].x;
        }
        else
        {
            output[oIdx] = lib_make_vector2<T>(0, 0);
        }
    }
    else if(scheme == 2)
    {
        // RES_MUL is the last step of bluestein and
        // should never be the first kernel to read global memory.
        // So we should never need to run a "load" callback.

        const T* chirp = input;

        iIdx += 2 * M;
        iIdx += iOffset;

        oIdx += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        T              out_elem;

        out_elem.x = MI * (input[iIdx].x * chirp[tx].x + input[iIdx].y * chirp[tx].y);
        out_elem.y = MI * (-input[iIdx].x * chirp[tx].y + input[iIdx].y * chirp[tx].x);
        store_cb(output, oIdx, out_elem, store_cb_data, nullptr);
    }
}

template <typename T>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_BLUESTEIN_KERNEL)
    mul_device_P_I(const size_t          numof,
                   const size_t          totalWI,
                   const size_t          N,
                   const size_t          M,
                   const real_type_t<T>* inputRe,
                   const real_type_t<T>* inputIm,
                   T*                    output,
                   const size_t          dim,
                   const size_t*         lengths,
                   const size_t*         stride_in,
                   const size_t*         stride_out,
                   const int             dir,
                   const int             scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx          = tx % numof;
    size_t iIdx = tx * stride_in[0];
    size_t oIdx = tx * stride_out[0];

    if(scheme == 0)
    {
        output += oOffset;

        T out          = output[oIdx];
        output[oIdx].x = inputRe[iIdx] * out.x - inputIm[iIdx] * out.y;
        output[oIdx].y = inputRe[iIdx] * out.y + inputIm[iIdx] * out.x;
    }
    else if(scheme == 1)
    {
        T* chirp = output;

        inputRe += iOffset;
        inputIm += iOffset;

        output += M;
        output += oOffset;

        if(tx < N)
        {
            output[oIdx].x = inputRe[iIdx] * chirp[tx].x + inputIm[iIdx] * chirp[tx].y;
            output[oIdx].y = -inputRe[iIdx] * chirp[tx].y + inputIm[iIdx] * chirp[tx].x;
        }
        else
        {
            output[oIdx] = lib_make_vector2<T>(0, 0);
        }
    }
    else if(scheme == 2)
    {
        const real_type_t<T>* chirpRe = inputRe;
        const real_type_t<T>* chirpIm = inputIm;

        inputRe += 2 * M;
        inputRe += iOffset;

        inputIm += 2 * M;
        inputIm += iOffset;

        output += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        output[oIdx].x    = MI * (inputRe[iIdx] * chirpRe[tx] + inputIm[iIdx] * chirpIm[tx]);
        output[oIdx].y    = MI * (-inputRe[iIdx] * chirpIm[tx] + inputIm[iIdx] * chirpRe[tx]);
    }
}

template <typename T>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_BLUESTEIN_KERNEL)
    mul_device_I_P(const size_t    numof,
                   const size_t    totalWI,
                   const size_t    N,
                   const size_t    M,
                   const T*        input,
                   real_type_t<T>* outputRe,
                   real_type_t<T>* outputIm,
                   const size_t    dim,
                   const size_t*   lengths,
                   const size_t*   stride_in,
                   const size_t*   stride_out,
                   const int       dir,
                   const int       scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx          = tx % numof;
    size_t iIdx = tx * stride_in[0];
    size_t oIdx = tx * stride_out[0];

    if(scheme == 0)
    {
        outputRe += oOffset;
        outputIm += oOffset;

        T out          = lib_make_vector2<T>(outputRe[oIdx], outputIm[oIdx]);
        outputRe[oIdx] = input[iIdx].x * out.x - input[iIdx].y * out.y;
        outputIm[oIdx] = input[iIdx].x * out.y + input[iIdx].y * out.x;
    }
    else if(scheme == 1)
    {
        real_type_t<T>* chirpRe = outputRe;
        real_type_t<T>* chirpIm = outputIm;

        input += iOffset;

        outputRe += M;
        outputRe += oOffset;

        outputIm += M;
        outputIm += oOffset;

        if(tx < N)
        {
            outputRe[oIdx] = input[iIdx].x * chirpRe[tx] + input[iIdx].y * chirpIm[tx];
            outputIm[oIdx] = -input[iIdx].x * chirpIm[tx] + input[iIdx].y * chirpRe[tx];
        }
        else
        {
            outputRe[oIdx] = 0;
            outputIm[oIdx] = 0;
        }
    }
    else if(scheme == 2)
    {
        const T* chirp = input;

        input += 2 * M;
        input += iOffset;

        outputRe += oOffset;
        outputIm += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        outputRe[oIdx]    = MI * (input[iIdx].x * chirp[tx].x + input[iIdx].y * chirp[tx].y);
        outputIm[oIdx]    = MI * (-input[iIdx].x * chirp[tx].y + input[iIdx].y * chirp[tx].x);
    }
}

template <typename T>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_BLUESTEIN_KERNEL)
    mul_device_P_P(const size_t          numof,
                   const size_t          totalWI,
                   const size_t          N,
                   const size_t          M,
                   const real_type_t<T>* inputRe,
                   const real_type_t<T>* inputIm,
                   real_type_t<T>*       outputRe,
                   real_type_t<T>*       outputIm,
                   const size_t          dim,
                   const size_t*         lengths,
                   const size_t*         stride_in,
                   const size_t*         stride_out,
                   const int             dir,
                   const int             scheme)
{
    size_t tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if(tx >= totalWI)
        return;

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = tx / numof;

    for(size_t i = dim; i > 1; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 1; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[1];
    oOffset += counter_mod * stride_out[1];

    tx          = tx % numof;
    size_t iIdx = tx * stride_in[0];
    size_t oIdx = tx * stride_out[0];

    if(scheme == 0)
    {
        outputRe += oOffset;
        outputIm += oOffset;

        T out          = lib_make_vector2<T>(outputRe[oIdx], outputIm[oIdx]);
        outputRe[oIdx] = inputRe[iIdx] * out.x - inputIm[iIdx] * out.y;
        outputIm[oIdx] = inputRe[iIdx] * out.y + inputIm[iIdx] * out.x;
    }
    else if(scheme == 1)
    {
        real_type_t<T>* chirpRe = outputRe;
        real_type_t<T>* chirpIm = outputIm;

        inputRe += iOffset;
        inputIm += iOffset;

        outputRe += M;
        outputRe += oOffset;

        outputIm += M;
        outputIm += oOffset;

        if(tx < N)
        {
            outputRe[oIdx] = inputRe[iIdx] * chirpRe[tx] + inputIm[iIdx] * chirpIm[tx];
            outputIm[oIdx] = -inputRe[iIdx] * chirpIm[tx] + inputIm[iIdx] * chirpRe[tx];
        }
        else
        {
            outputRe[tx] = 0;
            outputIm[tx] = 0;
        }
    }
    else if(scheme == 2)
    {
        const real_type_t<T>* chirpRe = inputRe;
        const real_type_t<T>* chirpIm = inputIm;

        inputRe += 2 * M;
        inputRe += iOffset;

        inputIm += 2 * M;
        inputIm += iOffset;

        outputRe += oOffset;
        outputIm += oOffset;

        real_type_t<T> MI = 1.0 / (real_type_t<T>)M;
        outputRe[oIdx]    = MI * (inputRe[iIdx] * chirpRe[tx] + inputIm[iIdx] * chirpIm[tx]);
        outputIm[oIdx]    = MI * (-inputRe[iIdx] * chirpIm[tx] + inputIm[iIdx] * chirpRe[tx]);
    }
}

#endif // BLUESTEIN_H
