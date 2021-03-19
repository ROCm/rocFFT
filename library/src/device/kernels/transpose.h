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

#include "array_format.h"
#include "common.h"
#include "rocfft_hip.h"

#define MAX_LAUNCH_BOUNDS_TRANSPOSE_KERNEL 1024

#define TRANSPOSE_TWIDDLE_MUL(tmp)                                                                \
    if(WITH_TWL)                                                                                  \
    {                                                                                             \
        if(TWL == 1)                                                                              \
        {                                                                                         \
            if(DIR == -1)                                                                         \
            {                                                                                     \
                TWIDDLE_STEP_MUL_FWD(TWLstep1, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                TWIDDLE_STEP_MUL_INV(TWLstep1, twiddles_large, (gx + tx1) * (gy + ty1 + i), tmp); \
            }                                                                                     \
        }                                                                                         \
        else if(TWL == 2)                                                                         \
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
    }

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
          bool   ALL,
          bool   UNIT_STRIDE_0>
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
                                      size_t       stride_0_in,
                                      size_t       stride_0_out,
                                      T*           twiddles_large)
{
    __shared__ T shared[DIM_X][DIM_X];

    size_t tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;

    if(ALL)
    {
#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            T tmp;
            if(UNIT_STRIDE_0)
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            }
            else
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
            }
            TRANSPOSE_TWIDDLE_MUL(tmp);
            shared[tx1][ty1 + i] = tmp; // the transpose taking place here
        }

        __syncthreads();

        // From generated assembly it looks like the compiler has difficulty interleaving
        // long latency instructions in the store loop. The workaround is to load LDS
        // data to a register array *and* separate LDS read and VMEM write into two loops,
        // so that compiler easily sees the oppportunity for latency hiding
        //
        // Similar workaround is also applied to transpose_tile_device_scheme
        T val[DIM_X / DIM_Y];
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            val[j] = shared[ty1 + i][tx1];
        }
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            // reconfigure the threads
            if(UNIT_STRIDE_0)
            {
                Handler<T_O>::write(output, out_offset + tx1 + (i + ty1) * ld_out, val[j]);
            }
            else
            {
                Handler<T_O>::write(
                    output, out_offset + tx1 * stride_0_out + (i + ty1) * ld_out, val[j]);
            }
        }
    }
    else
    {
        // For edge tiles, fully unroll the loop nevertheless and use predicates
        // to control whether memory instructions should be issued
        T val[DIM_X / DIM_Y];
#pragma unroll
        for(size_t i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < n && (ty1 + i) < m && i < m)
            {
                if(UNIT_STRIDE_0)
                {
                    val[j] = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
                }
                else
                {
                    val[j] = Handler<T_I>::read(input,
                                                in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
                }
                TRANSPOSE_TWIDDLE_MUL(val[j]);
            }
        }
#pragma unroll
        for(size_t i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < n && (ty1 + i) < m && i < m)
            {
                shared[tx1][ty1 + i] = val[j]; // the transpose taking place here
            }
        }
        __syncthreads();
#pragma unroll
        for(size_t i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < m && (ty1 + i) < n && i < n)
            {
                val[j] = shared[ty1 + i][tx1]; // the transpose taking place here
            }
        }
#pragma unroll
        for(size_t i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            // reconfigure the threads
            if(tx1 < m && (ty1 + i) < n && i < n)
            {
                if(UNIT_STRIDE_0)
                {
                    Handler<T_O>::write(output, out_offset + tx1 + (i + ty1) * ld_out, val[j]);
                }
                else
                {
                    Handler<T_O>::write(
                        output, out_offset + tx1 * stride_0_out + (i + ty1) * ld_out, val[j]);
                }
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
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
__global__ void __launch_bounds__(MAX_LAUNCH_BOUNDS_TRANSPOSE_KERNEL)
    transpose_kernel2(const T_I* input,
                      T_O*       output,
                      T*         twiddles_large,
                      size_t*    lengths,
                      size_t*    stride_in,
                      size_t*    stride_out)
{
    size_t ld_in  = stride_in[1];
    size_t ld_out = stride_out[1];

    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = hipBlockIdx_z;

    iOffset += counter_mod * stride_in[2];
    oOffset += counter_mod * stride_out[2];

    size_t tileBlockIdx_x, tileBlockIdx_y;
    if(DIAGONAL) // diagonal reordering
    {
        //TODO: template and simplify index calc for square case if necessary
        size_t bid     = hipBlockIdx_x + gridDim.x * hipBlockIdx_y;
        tileBlockIdx_y = bid % hipGridDim_y;
        tileBlockIdx_x = ((bid / hipGridDim_y) + tileBlockIdx_y) % hipGridDim_x;
    }
    else
    {
        tileBlockIdx_x = hipBlockIdx_x;
        tileBlockIdx_y = hipBlockIdx_y;
    }

    iOffset += tileBlockIdx_x * DIM_X * stride_in[0] + tileBlockIdx_y * DIM_X * ld_in;
    oOffset += tileBlockIdx_x * DIM_X * ld_out + tileBlockIdx_y * DIM_X * stride_out[0];

    if(ALL)
    {
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL, UNIT_STRIDE_0>(
            input,
            output,
            iOffset,
            oOffset,
            DIM_X,
            DIM_X,
            tileBlockIdx_x * DIM_X,
            tileBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            stride_in[0],
            stride_out[0],
            twiddles_large);
    }
    else
    {
        size_t m  = lengths[1];
        size_t n  = lengths[0];
        size_t mm = min(m - tileBlockIdx_y * DIM_X, DIM_X); // the corner case along m
        size_t nn = min(n - tileBlockIdx_x * DIM_X, DIM_X); // the corner case along n
        transpose_tile_device<T, T_I, T_O, DIM_X, DIM_Y, WITH_TWL, TWL, DIR, ALL, UNIT_STRIDE_0>(
            input,
            output,
            iOffset,
            oOffset,
            mm,
            nn,
            tileBlockIdx_x * DIM_X,
            tileBlockIdx_y * DIM_X,
            ld_in,
            ld_out,
            stride_in[0],
            stride_out[0],
            twiddles_large);
    }
}

// tiled transpose device function for transpose_scheme
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   ALL,
          bool   UNIT_STRIDE_0>
__device__ void transpose_tile_device_scheme(const T_I*   input,
                                             T_O*         output,
                                             size_t       in_offset,
                                             size_t       out_offset,
                                             const size_t m,
                                             const size_t n,
                                             size_t       ld_in,
                                             size_t       ld_out,
                                             size_t       stride_0_in,
                                             size_t       stride_0_out)
{
    __shared__ T shared[DIM_X][DIM_X];

    size_t tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;

    if(ALL)
    {
#pragma unroll
        for(int i = 0; i < DIM_X; i += DIM_Y)
        {
            T tmp;
            if(UNIT_STRIDE_0)
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
            }
            else
            {
                tmp = Handler<T_I>::read(input, in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
            }
            shared[tx1][ty1 + i] = tmp; // the transpose taking place here
        }

        __syncthreads();
        T val[DIM_X / DIM_Y];
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            val[j] = shared[ty1 + i][tx1];
        }
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            // reconfigure the threads
            if(UNIT_STRIDE_0)
            {
                Handler<T_O>::write(output, out_offset + tx1 + (i + ty1) * ld_out, val[j]);
            }
            else
            {
                Handler<T_O>::write(
                    output, out_offset + tx1 * stride_0_out + (i + ty1) * ld_out, val[j]);
            }
        }
    }
    else
    {
        T val[DIM_X / DIM_Y];
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < n && (ty1 + i) < m && i < m)
            {
                if(UNIT_STRIDE_0)
                {
                    val[j] = Handler<T_I>::read(input, in_offset + tx1 + (ty1 + i) * ld_in);
                }
                else
                {
                    val[j] = Handler<T_I>::read(input,
                                                in_offset + tx1 * stride_0_in + (ty1 + i) * ld_in);
                }
            }
        }
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < n && (ty1 + i) < m && i < m)
            {
                shared[tx1][ty1 + i] = val[j]; // the transpose taking place here
            }
        }
        __syncthreads();
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            if(tx1 < m && (ty1 + i) < n && i < n)
            {
                val[j] = shared[ty1 + i][tx1];
            }
        }
#pragma unroll
        for(int i = 0, j = 0; i < DIM_X; i += DIM_Y, j++)
        {
            // reconfigure the threads
            if(tx1 < m && (ty1 + i) < n && i < n)
            {
                if(UNIT_STRIDE_0)
                {
                    Handler<T_O>::write(output, out_offset + tx1 + (i + ty1) * ld_out, val[j]);
                }
                else
                {
                    Handler<T_O>::write(
                        output, out_offset + tx1 * stride_0_out + (i + ty1) * ld_out, val[j]);
                }
            }
        }
    }
}

// global function for transpose scheme
template <typename T,
          typename T_I,
          typename T_O,
          size_t DIM_X,
          size_t DIM_Y,
          bool   ALL,
          bool   UNIT_STRIDE_0,
          bool   DIAGONAL>
__global__ void __launch_bounds__(MAX_LAUNCH_BOUNDS_TRANSPOSE_KERNEL)
    transpose_kernel2_scheme(const T_I* input,
                             T_O*       output,
                             T*         twiddles_large,
                             size_t*    lengths,
                             size_t*    stride_in,
                             size_t*    stride_out,
                             size_t     ld_in,
                             size_t     ld_out,
                             size_t     m,
                             size_t     n)
{
    size_t iOffset = 0;
    size_t oOffset = 0;

    size_t counter_mod = hipBlockIdx_z;

    iOffset += counter_mod * stride_in[3];
    oOffset += counter_mod * stride_out[3];

    size_t tileBlockIdx_x, tileBlockIdx_y;
    if(DIAGONAL) // diagonal reordering
    {
        //TODO: template and simplify index calc for square case if necessary
        size_t bid     = hipBlockIdx_x + gridDim.x * hipBlockIdx_y;
        tileBlockIdx_y = bid % hipGridDim_y;
        tileBlockIdx_x = ((bid / hipGridDim_y) + tileBlockIdx_y) % hipGridDim_x;
    }
    else
    {
        tileBlockIdx_x = hipBlockIdx_x;
        tileBlockIdx_y = hipBlockIdx_y;
    }

    iOffset += tileBlockIdx_x * DIM_X * stride_in[0] + tileBlockIdx_y * DIM_X * ld_in;
    oOffset += tileBlockIdx_x * DIM_X * ld_out + tileBlockIdx_y * DIM_X * stride_out[0];

    if(ALL)
    {
        transpose_tile_device_scheme<T, T_I, T_O, DIM_X, DIM_Y, ALL, UNIT_STRIDE_0>(input,
                                                                                    output,
                                                                                    iOffset,
                                                                                    oOffset,
                                                                                    DIM_X,
                                                                                    DIM_X,
                                                                                    ld_in,
                                                                                    ld_out,
                                                                                    stride_in[0],
                                                                                    stride_out[0]);
    }
    else
    {
        size_t mm = min(m - tileBlockIdx_y * DIM_X, DIM_X); // the partial case along m
        size_t nn = min(n - tileBlockIdx_x * DIM_X, DIM_X); // the partial case along n
        transpose_tile_device_scheme<T, T_I, T_O, DIM_X, DIM_Y, ALL, UNIT_STRIDE_0>(
            input, output, iOffset, oOffset, mm, nn, ld_in, ld_out, stride_in[0], stride_out[0]);
    }
}

#endif // TRANSPOSE_H
