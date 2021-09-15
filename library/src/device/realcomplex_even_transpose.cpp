// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "./kernels/array_format.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"

#include <numeric>

__device__ size_t output_row_base(size_t        dim,
                                  size_t        output_batch_start,
                                  const size_t* outStride,
                                  const size_t  col)
{
    if(dim == 2)
        return output_batch_start + outStride[1] * col;
    else if(dim == 3)
        return output_batch_start + outStride[2] * col;
    return 0;
}

// R2C post-process kernel, 2D and 3D, transposed output.
// lengths counts in complex elements
template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y, CallbackType cbtype>
__global__ static void __launch_bounds__(DIM_X* DIM_Y)
    real_post_process_kernel_transpose(size_t        dim,
                                       const T_I*    input0,
                                       size_t        idist,
                                       T_O*          output0,
                                       size_t        odist,
                                       const void*   twiddles0,
                                       const size_t* lengths,
                                       const size_t* inStride,
                                       const size_t* outStride,
                                       void* __restrict__ load_cb_fn,
                                       void* __restrict__ load_cb_data,
                                       uint32_t load_cb_lds_bytes,
                                       void* __restrict__ store_cb_fn,
                                       void* __restrict__ store_cb_data)
{
    size_t idist1D            = inStride[1];
    size_t input_batch_start  = idist * blockIdx.z;
    size_t output_batch_start = odist * blockIdx.z;
    auto   twiddles           = static_cast<const T*>(twiddles0);

    // allocate 2 tiles so we can butterfly the values together.
    // left tile grabs values from towards the beginnings of the rows
    // right tile grabs values from towards the ends
    __shared__ T leftTile[DIM_X][DIM_Y];
    __shared__ T rightTile[DIM_X][DIM_Y];

    // take fastest dimension and partition it into lengths that will go into each tile
    const size_t len0 = lengths[0];
    // size of a complete tile for this problem - ignore the first
    // element and middle element (if there is one).  those are
    // treated specially
    const size_t tile_size = (len0 - 1) / 2 < DIM_X ? (len0 - 1) / 2 : DIM_X;

    // first column to read into the left tile, offset by one because
    // first element is already handled
    const size_t left_col_start = blockIdx.x * tile_size + 1;
    const size_t middle         = (len0 + 1) / 2;

    // number of columns to actually read into the tile (can be less
    // than tile size if we're out of data)
    size_t cols_to_read = tile_size;
    if(left_col_start + tile_size >= middle)
        cols_to_read = middle - left_col_start;

    // maximum number of rows in the problem
    const size_t row_limit = dim == 2 ? lengths[1] : lengths[1] * lengths[2];

    // start+end of range this thread will work on
    const size_t row_start = blockIdx.y * DIM_Y;
    size_t       row_end   = DIM_Y + row_start;
    if(row_end > row_limit)
        row_end = row_limit;

    const size_t lds_row = threadIdx.y;
    const size_t lds_col = threadIdx.x;
    // TODO: currently assumes idist2D has no extra padding
    const size_t input_row_base = (row_start + lds_row) * idist1D;

    if(row_start + lds_row < row_end && lds_col < cols_to_read)
    {
        auto v                     = Handler<T_I, cbtype>::read(input0,
                                            input_batch_start + input_row_base + left_col_start
                                                + lds_col,
                                            load_cb_fn,
                                            load_cb_data);
        leftTile[lds_col][lds_row] = v;

        auto v2                     = Handler<T_I, cbtype>::read(input0,
                                             input_batch_start + input_row_base
                                                 + (len0 - (left_col_start + cols_to_read - 1))
                                                 + lds_col,
                                             load_cb_fn,
                                             load_cb_data);
        rightTile[lds_col][lds_row] = v2;
    }

    // handle first + middle element (if there is a middle)
    T first_elem;
    T middle_elem;
    if(blockIdx.x == 0 && threadIdx.x == 0 && row_start + lds_row < row_end)
    {
        first_elem = Handler<T_I, cbtype>::read(
            input0, input_batch_start + input_row_base, load_cb_fn, load_cb_data);

        if(len0 % 2 == 0)
        {
            middle_elem = Handler<T_I, cbtype>::read(
                input0, input_batch_start + input_row_base + len0 / 2, load_cb_fn, load_cb_data);
        }
    }

    __syncthreads();

    // write first + middle
    if(blockIdx.x == 0 && threadIdx.x == 0 && row_start + lds_row < row_end)
    {
        T tmp;
        tmp.x = first_elem.x - first_elem.y;
        tmp.y = 0.0;
        Handler<T_O, cbtype>::write(output0,
                                    output_row_base(dim, output_batch_start, outStride, len0)
                                        + row_start + lds_row,
                                    tmp,
                                    store_cb_fn,
                                    store_cb_data);
        T tmp2;
        tmp2.x = first_elem.x + first_elem.y;
        tmp2.y = 0.0;
        Handler<T_O, cbtype>::write(output0,
                                    output_row_base(dim, output_batch_start, outStride, 0)
                                        + row_start + lds_row,
                                    tmp2,
                                    store_cb_fn,
                                    store_cb_data);

        if(len0 % 2 == 0)
        {

            tmp.x = middle_elem.x;
            tmp.y = -middle_elem.y;

            Handler<T_O, cbtype>::write(output0,
                                        output_row_base(dim, output_batch_start, outStride, middle)
                                            + row_start + lds_row,
                                        tmp,
                                        store_cb_fn,
                                        store_cb_data);
        }
    }

    // butterfly the two tiles we've collected (offset col by one
    // because first element is special)
    if(row_start + lds_row < row_end && lds_col < cols_to_read)
    {
        size_t col = blockIdx.x * tile_size + 1 + threadIdx.x;

        const T p = leftTile[lds_col][lds_row];
        const T q = rightTile[cols_to_read - lds_col - 1][lds_row];
        const T u = 0.5 * (p + q);
        const T v = 0.5 * (p - q);

        auto twd_p = twiddles[col];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        // write left side
        T tmp;
        tmp.x                 = u.x + v.x * twd_p.y + u.y * twd_p.x;
        tmp.y                 = v.y + u.y * twd_p.y - v.x * twd_p.x;
        auto output_left_base = output_row_base(dim, output_batch_start, outStride, col);
        Handler<T_O, cbtype>::write(
            output0, output_left_base + row_start + lds_row, tmp, store_cb_fn, store_cb_data);

        // write right side
        T tmp2;
        tmp2.x                 = u.x - v.x * twd_p.y - u.y * twd_p.x;
        tmp2.y                 = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        auto output_right_base = output_row_base(dim, output_batch_start, outStride, len0 - col);
        Handler<T_O, cbtype>::write(
            output0, output_right_base + row_start + lds_row, tmp2, store_cb_fn, store_cb_data);
    }
}

// Entrance function for r2c post-processing kernel, fused with transpose
ROCFFT_DEVICE_EXPORT void r2c_1d_post_transpose(const void* data_p, void*)
{
    auto data = reinterpret_cast<const DeviceCallIn*>(data_p);

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    const void* bufIn0  = data->bufIn[0];
    void*       bufOut0 = data->bufOut[0];
    void*       bufOut1 = data->bufOut[1];

    size_t count = data->node->batch;
    size_t m     = data->node->length[1];
    size_t n     = data->node->length[0];
    size_t dim   = data->node->length.size();

    auto cbtype = data->get_callback_type();

    // we're allocating one thread per tile element.  16x16 seems to
    // hit a sweet spot for performance, where it's enough threads to
    // be useful, but not too many.
    //
    static const size_t DIM_X = 16;
    static const size_t DIM_Y = 16;

    // grid X dimension handles 2 tiles at a time, so allocate enough
    // blocks to go halfway across 'n'
    //
    // grid Y dimension needs enough blocks to handle the second
    // dimension - multiply by the third dimension to get enough
    // blocks, if we're doing 3D
    //
    // grid Z counts number of batches
    dim3 grid((n - 1) / DIM_X / 2 + 1,
              ((m - 1) / DIM_Y + 1) * (dim > 2 ? data->node->length[2] : 1),
              count);
    // one thread per element in a tile
    dim3 threads(DIM_X, DIM_Y, 1);

    // rc input should always be interleaved by this point - we
    // should have done a transform just before this operation which
    // outputs interleaved
    assert(is_complex_interleaved(data->node->inArrayType));
    if(data->node->precision == rocfft_precision_single)
    {
        if(is_complex_planar(data->node->outArrayType))
        {
            cmplx_planar_device_buffer<float2> out_planar(bufOut0, bufOut1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_float,
                                                                   cmplx_float,
                                                                   cmplx_float_planar,
                                                                   DIM_X,
                                                                   DIM_Y,
                                                                   CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_float*>(bufIn0),
                idist,
                out_planar.devicePtr(),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
        else
        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        real_post_process_kernel_transpose<cmplx_float,
                                                           cmplx_float,
                                                           cmplx_float,
                                                           DIM_X,
                                                           DIM_Y,
                                                           CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_float,
                                                                         cmplx_float,
                                                                         cmplx_float,
                                                                         DIM_X,
                                                                         DIM_Y,
                                                                         CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_float*>(bufIn0),
                idist,
                static_cast<cmplx_float*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
    }
    else
    {
        if(is_complex_planar(data->node->outArrayType))
        {
            cmplx_planar_device_buffer<double2> out_planar(bufOut0, bufOut1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_double,
                                                                   cmplx_double,
                                                                   cmplx_double_planar,
                                                                   DIM_X,
                                                                   DIM_Y,
                                                                   CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_double*>(bufIn0),
                idist,
                out_planar.devicePtr(),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
        else
        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        real_post_process_kernel_transpose<cmplx_double,
                                                           cmplx_double,
                                                           cmplx_double,
                                                           DIM_X,
                                                           DIM_Y,
                                                           CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(real_post_process_kernel_transpose<cmplx_double,
                                                                         cmplx_double,
                                                                         cmplx_double,
                                                                         DIM_X,
                                                                         DIM_Y,
                                                                         CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_double*>(bufIn0),
                idist,
                static_cast<cmplx_double*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
    }
}

// C2R pre-process kernel, 2D and 3D, transposed input.
// lengths counts in complex elements
template <typename T, typename T_I, typename T_O, size_t DIM_X, size_t DIM_Y, CallbackType cbtype>
__global__ static void __launch_bounds__(DIM_X* DIM_Y)
    transpose_real_pre_process_kernel(size_t        dim,
                                      const T_I*    input0,
                                      size_t        idist,
                                      T_O*          output0,
                                      size_t        odist,
                                      const void*   twiddles0,
                                      const size_t* lengths,
                                      const size_t* inStride,
                                      const size_t* outStride,
                                      void* __restrict__ load_cb_fn,
                                      void* __restrict__ load_cb_data,
                                      uint32_t load_cb_lds_bytes,
                                      void* __restrict__ store_cb_fn,
                                      void* __restrict__ store_cb_data)
{
    size_t idist1D            = dim == 2 ? inStride[1] : inStride[2];
    size_t odist1D            = outStride[1];
    size_t input_batch_start  = idist * blockIdx.z;
    size_t output_batch_start = odist * blockIdx.z;
    auto   twiddles           = static_cast<const T*>(twiddles0);

    // allocate 2 tiles so we can butterfly the values together.
    // top tile grabs values from towards the tops of the columns
    // bottom tile grabs values from towards the bottoms
    __shared__ T topTile[DIM_X][DIM_Y];
    __shared__ T bottomTile[DIM_X][DIM_Y];

    // take middle dimension and partition it into lengths that will go into each tile
    // note that last row effectively gets thrown away
    const size_t len1 = dim == 2 ? lengths[1] - 1 : lengths[2] - 1;
    // size of a complete tile for this problem - ignore the first
    // element and middle element (if there is one).  those are
    // treated specially
    const size_t tile_size = (len1 - 1) / 2 < DIM_Y ? (len1 - 1) / 2 : DIM_Y;

    // first column to read into the left tile, offset by one because
    // first element is already handled
    const size_t top_row_start = blockIdx.y * tile_size + 1;

    // middle row
    const size_t middle = (len1 + 1) / 2;

    // number of rows to actually read into the tile (can be less
    // than tile size if we're out of data)
    size_t rows_to_read = tile_size;
    // read towards the middle, but not past
    if(top_row_start + tile_size >= middle)
        rows_to_read = middle - top_row_start;

    const size_t col_start = blockIdx.x * DIM_X;
    size_t       col_end   = DIM_X + col_start;
    // TODO: currently assumes idist2D has no extra padding
    const size_t col_limit = dim == 2 ? lengths[0] : lengths[0] * lengths[1];
    if(col_end > col_limit)
        col_end = col_limit;

    const size_t lds_row        = threadIdx.y;
    const size_t lds_col        = threadIdx.x;
    const size_t input_col_base = col_start;

    if(col_start + lds_col < col_end && lds_row < rows_to_read)
    {
        auto v                    = Handler<T_I, cbtype>::read(input0,
                                            input_batch_start + input_col_base + lds_col
                                                + (top_row_start + lds_row) * idist1D,
                                            load_cb_fn,
                                            load_cb_data);
        topTile[lds_col][lds_row] = v;

        auto v2 = Handler<T_I, cbtype>::read(input0,
                                             input_batch_start + input_col_base + lds_col
                                                 + (len1 - (top_row_start + lds_row)) * idist1D,
                                             load_cb_fn,
                                             load_cb_data);
        // TODO: reads values-to-butterfly into same col/row in LDS.
        // r2c kernel writes LDS in same order as input.  these
        // probably should be made consistent
        bottomTile[lds_col][lds_row] = v2;
    }

    // handle first + last + middle element (if there is a middle)
    T first_elem;
    T middle_elem;
    T last_elem;
    if(blockIdx.y == 0 && threadIdx.y == 0 && col_start + lds_col < col_end)
    {
        first_elem = Handler<T_I, cbtype>::read(
            input0, input_batch_start + col_start + lds_col, load_cb_fn, load_cb_data);
        if(len1 % 2 == 0)
        {
            middle_elem = Handler<T_I, cbtype>::read(input0,
                                                     input_batch_start + col_start + lds_col
                                                         + middle * idist1D,
                                                     load_cb_fn,
                                                     load_cb_data);
        }
        last_elem
            = Handler<T_I, cbtype>::read(input0,
                                         input_batch_start + col_start + lds_col + len1 * idist1D,
                                         load_cb_fn,
                                         load_cb_data);
    }

    __syncthreads();

    // write first + middle
    if(blockIdx.y == 0 && threadIdx.y == 0 && col_start + lds_col < col_end)
    {
        T tmp;
        tmp.x = first_elem.x - first_elem.y + last_elem.x + last_elem.y;
        tmp.y = first_elem.x + first_elem.y - last_elem.x + last_elem.y;
        Handler<T_O, cbtype>::write(output0,
                                    output_batch_start + outStride[1] * (col_start + lds_col),
                                    tmp,
                                    store_cb_fn,
                                    store_cb_data);

        if(len1 % 2 == 0)
        {

            tmp.x = 2.0 * middle_elem.x;
            tmp.y = -2.0 * middle_elem.y;

            Handler<T_O, cbtype>::write(output0,
                                        output_batch_start + outStride[1] * (col_start + lds_col)
                                            + middle,
                                        tmp,
                                        store_cb_fn,
                                        store_cb_data);
        }
    }

    // butterfly the two tiles we've collected (offset col by one
    // because first element is special)
    if(col_start + lds_col < col_end && lds_row < rows_to_read)
    {
        const T p = topTile[lds_col][lds_row];
        const T q = bottomTile[lds_col][lds_row];
        const T u = p + q;
        const T v = p - q;

        auto twd_p = twiddles[top_row_start + lds_row];

        // write top side
        T tmp;
        tmp.x = u.x + v.x * twd_p.y - u.y * twd_p.x;
        tmp.y = v.y + u.y * twd_p.y + v.x * twd_p.x;
        Handler<T_O, cbtype>::write(output0,
                                    output_batch_start + top_row_start + lds_row
                                        + (col_start + lds_col) * odist1D,
                                    tmp,
                                    store_cb_fn,
                                    store_cb_data);

        // write bottom side
        T tmp2;
        tmp2.x = u.x - v.x * twd_p.y + u.y * twd_p.x;
        tmp2.y = -v.y + u.y * twd_p.y + v.x * twd_p.x;
        Handler<T_O, cbtype>::write(output0,
                                    output_batch_start + (len1 - (top_row_start + lds_row))
                                        + (col_start + lds_col) * odist1D,
                                    tmp2,
                                    store_cb_fn,
                                    store_cb_data);
    }
}

// Entrance function for c2r pre-processing kernel, fused with transpose
ROCFFT_DEVICE_EXPORT void transpose_c2r_1d_pre(const void* data_p, void*)
{
    auto data = reinterpret_cast<const DeviceCallIn*>(data_p);

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    const void* bufIn0  = data->bufIn[0];
    const void* bufIn1  = data->bufIn[1];
    void*       bufOut0 = data->bufOut[0];

    size_t count = data->node->batch;
    size_t m     = data->node->length[1];
    size_t n     = data->node->length[0];
    size_t dim   = data->node->length.size();

    auto cbtype = data->get_callback_type();

    // we're allocating one thread per tile element.  32x16 seems to
    // hit a sweet spot for performance, where it's enough threads to
    // be useful, but not too many.
    static const size_t DIM_X = 32;
    static const size_t DIM_Y = 16;

    // grid X dimension needs enough blocks to handle the first
    // dimension - multiply by the second dimension to get enough
    // blocks, if we're doing 3D
    if(dim > 2)
    {
        n *= data->node->length[1];
        m = data->node->length[2];
    }
    //
    // grid Y dimension handles 2 tiles at a time, so allocate enough
    // blocks to go halfway across 'm'
    //
    // grid Z counts number of batches
    dim3 grid((n - 1) / DIM_X + 1, (m - 1) / DIM_Y / 2 + 1, count);
    // one thread per element in a tile
    dim3 threads(DIM_X, DIM_Y, 1);
    // printf("GRID (%d,%d,%d) (%d,%d,%d), n=%zu m=%zu\n",
    //        grid.x,
    //        grid.y,
    //        grid.z,
    //        threads.x,
    //        threads.y,
    //        threads.z,
    //        n,
    //        m);

    // c2r output should also be interleaved, as we expect to follow
    // with a transform that needs interleaved input
    assert(is_complex_interleaved(data->node->outArrayType));
    if(data->node->precision == rocfft_precision_single)
    {
        if(is_complex_planar(data->node->inArrayType))
        {
            cmplx_planar_device_buffer<float2> in_planar(bufIn0, bufIn1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(transpose_real_pre_process_kernel<cmplx_float,
                                                                  cmplx_float_planar,
                                                                  cmplx_float,
                                                                  DIM_X,
                                                                  DIM_Y,
                                                                  CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                in_planar.devicePtr(),
                idist,
                static_cast<cmplx_float*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
        else
        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        transpose_real_pre_process_kernel<cmplx_float,
                                                          cmplx_float,
                                                          cmplx_float,
                                                          DIM_X,
                                                          DIM_Y,
                                                          CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(transpose_real_pre_process_kernel<cmplx_float,
                                                                        cmplx_float,
                                                                        cmplx_float,
                                                                        DIM_X,
                                                                        DIM_Y,
                                                                        CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_float*>(bufIn0),
                idist,
                static_cast<cmplx_float*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
    }
    else
    {
        if(is_complex_planar(data->node->inArrayType))
        {
            cmplx_planar_device_buffer<double2> in_planar(bufIn0, bufIn1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(transpose_real_pre_process_kernel<cmplx_double,
                                                                  cmplx_double_planar,
                                                                  cmplx_double,
                                                                  DIM_X,
                                                                  DIM_Y,
                                                                  CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                in_planar.devicePtr(),
                idist,
                static_cast<cmplx_double*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
        else

        {
            hipLaunchKernelGGL(
                cbtype == CallbackType::USER_LOAD_STORE
                    ? HIP_KERNEL_NAME(
                        transpose_real_pre_process_kernel<cmplx_double,
                                                          cmplx_double,
                                                          cmplx_double,
                                                          DIM_X,
                                                          DIM_Y,
                                                          CallbackType::USER_LOAD_STORE>)
                    : HIP_KERNEL_NAME(transpose_real_pre_process_kernel<cmplx_double,
                                                                        cmplx_double,
                                                                        cmplx_double,
                                                                        DIM_X,
                                                                        DIM_Y,
                                                                        CallbackType::NONE>),
                grid,
                threads,
                0,
                data->rocfft_stream,
                dim,
                static_cast<const cmplx_double*>(bufIn0),
                idist,
                static_cast<cmplx_double*>(bufOut0),
                odist,
                data->node->twiddles.data(),
                kargs_lengths(data->node->devKernArg),
                kargs_stride_in(data->node->devKernArg),
                kargs_stride_out(data->node->devKernArg),
                data->callbacks.load_cb_fn,
                data->callbacks.load_cb_data,
                data->callbacks.load_cb_lds_bytes,
                data->callbacks.store_cb_fn,
                data->callbacks.store_cb_data);
        }
    }
}
