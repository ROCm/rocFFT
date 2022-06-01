// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "kernel_launch.h"
#include "kernels/callback.h"

static const size_t APPLY_REAL_CALLBACK_THREADS = 64;
template <typename Treal>
__global__ void __launch_bounds__(APPLY_REAL_CALLBACK_THREADS)
    apply_real_callback_kernel(unsigned int dim,
                               unsigned int lengths0,
                               unsigned int lengths1,
                               unsigned int lengths2,
                               unsigned int stride_in0,
                               unsigned int stride_in1,
                               unsigned int stride_in2,
                               unsigned int stride_in3,
                               Treal* __restrict__ input,
                               void* __restrict__ load_cb_fn,
                               void* __restrict__ load_cb_data,
                               uint32_t load_cb_lds_bytes,
                               void* __restrict__ store_cb_fn,
                               void* __restrict__ store_cb_data)
{
    const size_t idx_0 = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int lengths[3]   = {lengths0, lengths1, lengths2};
    const unsigned int stride_in[4] = {stride_in0, stride_in1, stride_in2, stride_in3};

    // offsets
    size_t offset_in = 0;
    size_t remaining;
    size_t index_along_d;
    remaining = blockIdx.y;
    for(int d = 1; d < dim; ++d)
    {
        index_along_d = remaining % lengths[d];
        remaining     = remaining / lengths[d];
        offset_in     = offset_in + index_along_d * stride_in[d];
    }
    // remaining should be 1 at this point, since batch goes into blockIdx.z
    size_t batch = blockIdx.z;
    offset_in    = offset_in + batch * stride_in[dim];

    if(idx_0 < lengths[0])
    {
        auto load_cb  = get_load_cb<Treal, CallbackType::USER_LOAD_STORE>(load_cb_fn);
        auto store_cb = get_store_cb<Treal, CallbackType::USER_LOAD_STORE>(store_cb_fn);

        const auto inputIdx = offset_in + idx_0 * stride_in[0];

        auto elem = load_cb(input, inputIdx, load_cb_data, nullptr);
        store_cb(input, inputIdx, elem, store_cb_data, nullptr);
    }
}

ROCFFT_DEVICE_EXPORT void apply_real_callback(const void* data_p, void* back)
{
    auto data = static_cast<const DeviceCallIn*>(data_p);

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

    void* input_buffer = data->bufIn[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    size_t dim            = data->node->length.size();

    for(int i = 1; i < dim; i++)
    {
        high_dimension *= data->node->length[i];
    }

    rocfft_precision precision = data->node->precision;

    size_t blocks = (input_size - 1) / APPLY_REAL_CALLBACK_THREADS + 1;

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(APPLY_REAL_CALLBACK_THREADS, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    // explode lengths/strides out to pass to the kernel
    std::array<size_t, 3> kern_lengths{1, 1, 1};
    std::array<size_t, 4> kern_stride_in{1, 1, 1, 1};

    std::copy(data->node->length.begin(), data->node->length.end(), kern_lengths.begin());
    std::copy(data->node->inStride.begin(), data->node->inStride.end(), kern_stride_in.begin());
    kern_stride_in[dim] = data->node->iDist;

    if(precision == rocfft_precision_single)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_real_callback_kernel<float>),
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           dim,
                           kern_lengths[0],
                           kern_lengths[1],
                           kern_lengths[2],
                           kern_stride_in[0],
                           kern_stride_in[1],
                           kern_stride_in[2],
                           kern_stride_in[3],
                           static_cast<float*>(input_buffer),
                           data->callbacks.load_cb_fn,
                           data->callbacks.load_cb_data,
                           data->callbacks.load_cb_lds_bytes,
                           data->callbacks.store_cb_fn,
                           data->callbacks.store_cb_data);
    }
    else
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_real_callback_kernel<double>),
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           dim,
                           kern_lengths[0],
                           kern_lengths[1],
                           kern_lengths[2],
                           kern_stride_in[0],
                           kern_stride_in[1],
                           kern_stride_in[2],
                           kern_stride_in[3],
                           static_cast<double*>(input_buffer),
                           data->callbacks.load_cb_fn,
                           data->callbacks.load_cb_data,
                           data->callbacks.load_cb_lds_bytes,
                           data->callbacks.store_cb_fn,
                           data->callbacks.store_cb_data);
    }
}