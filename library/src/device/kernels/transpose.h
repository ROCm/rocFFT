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

#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "common.h"
#include "rocfft_hip.h"

#define TRANSPOSE_TWIDDLE_MUL()                                                                   \
    if(WITH_TWL)                                                                                  \
    {                                                                                             \
        if(TWL == 2)                                                                              \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep2, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep2, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
        else if(TWL == 3)                                                                         \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
        else if(TWL == 4)                                                                         \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    shared_A[tx1][ty1 + i] = tmp; // the transpose taking place here

//-----------------------------------------------------------------------------
// To support planar format with template, we have the below simple conventions.
// And it might be moved to somewhere to share.

template <typename PRECISION>
struct planar
{
    real_type_t<PRECISION>* R; // points to real part array
    real_type_t<PRECISION>* I; // points to imag part array
};

// the default interleaved format
using cmplx_float  = float2;
using cmplx_double = double2;

// the planar format
using cmplx_float_planar  = planar<float2>;
using cmplx_double_planar = planar<double2>;

template <class T>
struct cmplx_type;

template <>
struct cmplx_type<cmplx_float>
{
    typedef float2 type;
};

template <>
struct cmplx_type<double2>
{
    typedef double2 type;
};

template <>
struct cmplx_type<cmplx_float_planar>
{
    typedef float2 type;
};

template <>
struct cmplx_type<cmplx_double_planar>
{
    typedef double2 type;
};

template <class T>
using cmplx_type_t = typename cmplx_type<T>::type;

template <typename T>
struct Handler
{
};

template <>
struct Handler<cmplx_float>
{
    static __host__ __device__ inline float2 read(cmplx_float const* in, size_t idx)
    {
        return in[idx];
    }

    static __host__ __device__ inline void write(cmplx_float* out, size_t idx, float2 v)
    {
        out[idx] = v;
    }
};

template <>
struct Handler<cmplx_double>
{
    static __host__ __device__ inline double2 read(cmplx_double const* in, size_t idx)
    {
        return in[idx];
    }

    static __host__ __device__ inline void write(cmplx_double* out, size_t idx, double2 v)
    {
        out[idx] = v;
    }
};

template <>
struct Handler<cmplx_float_planar>
{
    static __host__ __device__ inline float2 read(cmplx_float_planar const* in, size_t idx)
    {
        float2 t;
        t.x = in->R[idx];
        t.y = in->I[idx];
        return t;
    }

    static __host__ __device__ inline void write(cmplx_float_planar* out, size_t idx, float2 v)
    {
        out->R[idx] = v.x;
        out->I[idx] = v.y;
    }
};

template <>
struct Handler<cmplx_double_planar>
{
    static __host__ __device__ inline double2 read(cmplx_double_planar const* in, size_t idx)
    {
        double2 t;
        t.x = in->R[idx];
        t.y = in->I[idx];
        return t;
    }

    static __host__ __device__ inline void write(cmplx_double_planar* out, size_t idx, double2 v)
    {
        out->R[idx] = v.x;
        out->I[idx] = v.y;
    }
};

//-----------------------------------------------------------------------------

// - transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
//   input, output are in device memory
//   shared memory of size DIM_X*DIM_X is allocated size_ternally as working space
// - Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
//   DIM_X is divisible by DIM_Y
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   WITH_TWL,
          int    TWL,
          int    DIR,
          bool   ALL>
__device__ void transpose_tile_device(const T_I*   input,
                                      T_O*         output,
                                      size_t       in_offset,
                                      size_t       out_offset,
                                      const size_t m,
                                      const size_t n,
                                      size_t       gx,
                                      size_t       gy,
                                      size_t       ld_in,
                                      size_t       ld_out,
                                      T*           twiddles_large)
{
    __shared__ T shared_A[DIM_X][DIM_X];

    size_t tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;

    if(ALL)
    {
#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            //T tmp = input[tx1 + (ty1 + i) * ld_in];
            T tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            TRANSPOSE_TWIDDLE_MUL();
        }

        __syncthreads();

#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            // reconfigure the threads
            //output[tx1 + (i + ty1) * ld_out] = shared_A[ty1 + i][tx1];
            Handler<T_O>::write(
                output, out_offset + tx1 + (i + ty1) * ld_out, shared_A[ty1 + i][tx1]);
        }
    }
    else
    {
        for(size_t i = 0; i < m; i += DIM_Y)
        {
            if(tx1 < n && (ty1 + i) < m)
            {
                //T tmp = input[tx1 + (ty1 + i) * ld_in];
                T tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
                TRANSPOSE_TWIDDLE_MUL();
            }
        }

        __syncthreads();

        for(size_t i = 0; i < n; i += DIM_Y)
        {
            // reconfigure the threads
            if(tx1 < m && (ty1 + i) < n)
            {
                //output[tx1 + (i + ty1) * ld_out] = shared_A[ty1 + i][tx1];
                Handler<T_O>::write(
                    output, out_offset + tx1 + (i + ty1) * ld_out, shared_A[ty1 + i][tx1]);
            }
        }
    }
}

// - transpose input of size m * n to output of size n * m
//   input, output are in device memory
// - 2D grid and 2D thread block (DIM_X, DIM_Y)
// - Assume DIM_X by DIM_Y threads are transposing a tile DIM_X * DIM_X
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   WITH_TWL,
          int    TWL,
          int    DIR,
          bool   ALL>
__global__ void transpose_kernel2(const T_I* input,
                                  T_O*       output,
                                  T*         twiddles_large,
                                  size_t     dim,
                                  size_t*    lengths,
                                  size_t*    stride_in,
                                  size_t*    stride_out)
{
    size_t ld_in  = stride_in[1];
    size_t ld_out = stride_out[1];

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = hipBlockIdx_z;

    for(size_t i = dim; i > 2; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 2; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[2];
    oOffset += counter_mod * stride_out[2];

    //input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in + iOffset;
    //output += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X + oOffset;
    iOffset += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in;
    oOffset += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X;

    if(ALL)
    {
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL>(
            input,
            output,
            iOffset,
            oOffset,
            DIM_X,
            DIM_X,
            hipBlockIdx_x * DIM_X,
            hipBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            twiddles_large);
    }
    else
    {
        size_t m  = lengths[1];
        size_t n  = lengths[0];
        size_t mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
        size_t nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL>(
            input,
            output,
            iOffset,
            oOffset,
            mm,
            nn,
            hipBlockIdx_x * DIM_X,
            hipBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            twiddles_large);
    }
}

template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y, bool ALL>
__global__ void transpose_kernel2_scheme(const T_I*   input,
                                         T_O*         output,
                                         T*           twiddles_large,
                                         size_t       dim,
                                         size_t*      lengths,
                                         size_t*      stride_in,
                                         size_t*      stride_out,
                                         const size_t scheme)
{
    size_t ld_in  = scheme == 1 ? stride_in[2] : stride_in[1];
    size_t ld_out = scheme == 1 ? stride_out[1] : stride_out[2];

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = hipBlockIdx_z;

    for(size_t i = dim; i > 3; i--)
    {
        size_t currentLength = 1;
        for(size_t j = 3; j < i; j++)
        {
            currentLength *= lengths[j];
        }

        iOffset += (counter_mod / currentLength) * stride_in[i];
        oOffset += (counter_mod / currentLength) * stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset += counter_mod * stride_in[3];
    oOffset += counter_mod * stride_out[3];

    //input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in + iOffset;
    //output += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X + oOffset;
    iOffset += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in;
    oOffset += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X;

    if(ALL)
    {
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, false, 0, 0, ALL>(input,
                                                                           output,
                                                                           iOffset,
                                                                           oOffset,
                                                                           DIM_X,
                                                                           DIM_X,
                                                                           hipBlockIdx_x * DIM_X,
                                                                           hipBlockIdx_y * DIM_X,
                                                                           ld_in,
                                                                           ld_out,
                                                                           twiddles_large);
    }
    else
    {
        size_t m  = scheme == 1 ? lengths[2] : lengths[1] * lengths[2];
        size_t n  = scheme == 1 ? lengths[0] * lengths[1] : lengths[0];
        size_t mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
        size_t nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, false, 0, 0, ALL>(input,
                                                                           output,
                                                                           iOffset,
                                                                           oOffset,
                                                                           mm,
                                                                           nn,
                                                                           hipBlockIdx_x * DIM_X,
                                                                           hipBlockIdx_y * DIM_X,
                                                                           ld_in,
                                                                           ld_out,
                                                                           twiddles_large);
    }
}

#endif // TRANSPOSE_H
