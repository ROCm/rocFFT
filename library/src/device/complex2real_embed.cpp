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

#include "./kernels/common.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename Tcomplex, CallbackType cbtype, unsigned int dim>
__global__ static void complex2real_kernel(unsigned int           lengths0,
                                           unsigned int           lengths1,
                                           unsigned int           lengths2,
                                           unsigned int           stride_in0,
                                           unsigned int           stride_in1,
                                           unsigned int           stride_in2,
                                           unsigned int           stride_in3,
                                           unsigned int           stride_out0,
                                           unsigned int           stride_out1,
                                           unsigned int           stride_out2,
                                           unsigned int           stride_out3,
                                           const Tcomplex*        input,
                                           real_type_t<Tcomplex>* output,
                                           void* __restrict__ load_cb_fn,
                                           void* __restrict__ load_cb_data,
                                           uint32_t load_cb_lds_bytes,
                                           void* __restrict__ store_cb_fn,
                                           void* __restrict__ store_cb_data)
{
    size_t idx_0 = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int lengths[3]    = {lengths0, lengths1, lengths2};
    const unsigned int stride_in[4]  = {stride_in0, stride_in1, stride_in2, stride_in3};
    const unsigned int stride_out[4] = {stride_out0, stride_out1, stride_out2, stride_out3};

    // offsets
    size_t offset_in  = 0;
    size_t offset_out = 0;
    size_t remaining;
    size_t index_along_d;
    remaining = blockIdx.y;
    for(int d = 1; d < dim; ++d)
    {
        index_along_d = remaining % lengths[d];
        remaining     = remaining / lengths[d];
        offset_in     = offset_in + index_along_d * stride_in[d];
        offset_out    = offset_out + index_along_d * stride_out[d];
    }
    // remaining should be 1 at this point, since batch goes into blockIdx.z
    size_t batch = blockIdx.z;
    offset_in    = offset_in + batch * stride_in[dim];
    offset_out   = offset_out + batch * stride_out[dim];

    if(idx_0 < lengths[0])
    {
        const auto inputIdx  = offset_in + idx_0 * stride_in[0];
        const auto outputIdx = offset_out + idx_0 * stride_out[0];

        // we would do real2complex at the end of a C2R transform, so
        // it would never be the first kernel to read from global
        // memory.  don't bother going through load callback to read
        // global memory.

        auto store_cb = get_store_cb<real_type_t<Tcomplex>, cbtype>(store_cb_fn);
        store_cb(output, outputIdx, input[inputIdx].x, store_cb_data, nullptr);
    }
}

#define COMPLEX2REAL_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                                         \
    decltype(&complex2real_kernel<TFLOAT, CallbackType::USER_LOAD_STORE, DIM>) kernel_func; \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)                          \
        kernel_func = complex2real_kernel<TFLOAT, CallbackType::USER_LOAD_STORE, DIM>;      \
    else                                                                                    \
        kernel_func = complex2real_kernel<TFLOAT, CallbackType::NONE, DIM>;                 \
    hipLaunchKernelGGL(kernel_func,                                                         \
                       grid,                                                                \
                       threads,                                                             \
                       0,                                                                   \
                       rocfft_stream,                                                       \
                       kern_lengths[0],                                                     \
                       kern_lengths[1],                                                     \
                       kern_lengths[2],                                                     \
                       kern_stride_in[0],                                                   \
                       kern_stride_in[1],                                                   \
                       kern_stride_in[2],                                                   \
                       kern_stride_in[3],                                                   \
                       kern_stride_out[0],                                                  \
                       kern_stride_out[1],                                                  \
                       kern_stride_out[2],                                                  \
                       kern_stride_out[3],                                                  \
                       static_cast<TFLOAT*>(input_buffer),                                  \
                       static_cast<real_type_t<TFLOAT>*>(output_buffer),                    \
                       data->callbacks.load_cb_fn,                                          \
                       data->callbacks.load_cb_data,                                        \
                       data->callbacks.load_cb_lds_bytes,                                   \
                       data->callbacks.store_cb_fn,                                         \
                       data->callbacks.store_cb_data);

// assign real2complex function pointer given a float type
#define COMPLEX2REAL_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                   \
    {                                              \
        COMPLEX2REAL_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                              \
    else if(dim == 2)                              \
    {                                              \
        COMPLEX2REAL_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                              \
    else if(dim == 3)                              \
    {                                              \
        COMPLEX2REAL_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                              \
    else                                           \
        throw std::runtime_error("invalid dimension in complex2real");

/// \brief auxiliary function
///   Convert a complex vector into a real one by only taking the real part of the complex
///   vector.
ROCFFT_DEVICE_EXPORT void complex2real(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0];

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    size_t dim            = data->node->length.size();
    if(dim > 1)
    {
        for(int i = 1; i < dim; i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (input_size - 1) / LAUNCH_BOUNDS_R2C_C2R_KERNEL + 1;

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_size*batch);
    // hipMemcpy(tmp, input_buffer, sizeof(float2)*input_size*batch,
    //           hipMemcpyDeviceToHost);
    // for(size_t j=0; j< (data->node->length.size() == 2 ? (data->node->length[1]) :
    //                     1); j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", i, j,
    //                tmp[j*data->node->length[0]+i].x, tmp[j*data->node->length[0]+i].y);
    //     }
    // }
    // free(tmp);

    // explode lengths/strides out to pass to the kernel
    std::array<size_t, 3> kern_lengths{1, 1, 1};
    std::array<size_t, 4> kern_stride_in{1, 1, 1, 1};
    std::array<size_t, 4> kern_stride_out{1, 1, 1, 1};

    std::copy(data->node->length.begin(), data->node->length.end(), kern_lengths.begin());
    std::copy(data->node->inStride.begin(), data->node->inStride.end(), kern_stride_in.begin());
    kern_stride_in[dim] = data->node->iDist;
    std::copy(data->node->outStride.begin(), data->node->outStride.end(), kern_stride_out.begin());
    kern_stride_out[dim] = data->node->oDist;

    if(precision == rocfft_precision_single)
    {
        COMPLEX2REAL_KERNEL_LAUNCH(float2);
    }
    else
    {
        COMPLEX2REAL_KERNEL_LAUNCH(double2);
    }
}

template <typename T, CallbackType cbtype, unsigned int dim>
__global__ static void hermitian2complex_kernel(const unsigned int hermitian_size,
                                                unsigned int       dim_0,
                                                unsigned int       dim_1,
                                                unsigned int       dim_2,
                                                unsigned int       stride_in0,
                                                unsigned int       stride_in1,
                                                unsigned int       stride_in2,
                                                unsigned int       stride_in3,
                                                unsigned int       stride_out0,
                                                unsigned int       stride_out1,
                                                unsigned int       stride_out2,
                                                unsigned int       stride_out3,
                                                T* __restrict__ input,
                                                T* __restrict__ output,
                                                void* __restrict__ load_cb_fn,
                                                void* __restrict__ load_cb_data,
                                                uint32_t load_cb_lds_bytes,
                                                void* __restrict__ store_cb_fn,
                                                void* __restrict__ store_cb_data)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int stride_in[4]  = {stride_in0, stride_in1, stride_in2, stride_in3};
    const unsigned int stride_out[4] = {stride_out0, stride_out1, stride_out2, stride_out3};

    // start with batch offset
    size_t input_offset   = blockIdx.z * stride_in[dim];
    size_t outputs_offset = blockIdx.z * stride_out[dim]; // straight copy
    size_t outputc_offset = blockIdx.z * stride_out[dim]; // conjugate copy

    // straight copy indices
    size_t is0 = tid;
    size_t is1 = blockIdx.y % dim_1;
    size_t is2 = blockIdx.y / dim_1;

    // conjugate copy indices
    size_t ic0 = (is0 == 0) ? 0 : dim_0 - is0;
    size_t ic1 = (is1 == 0) ? 0 : dim_1 - is1;
    size_t ic2 = (is2 == 0) ? 0 : dim_2 - is2;

    input_offset += is2 * stride_in2 + is1 * stride_in1 + is0 * stride_in0;
    // notice for 1D,
    // blockIdx.y == 0 and
    // thus has no effect for
    // input_offset
    outputs_offset += is2 * stride_out2 + is1 * stride_out1 + is0 * stride_out0;
    outputc_offset += ic2 * stride_out2 + ic1 * stride_out1 + ic0 * stride_out0;

    auto load_cb = get_load_cb<T, cbtype>(load_cb_fn);

    // we would do hermitian2complex at the start of a C2R transform,
    // so it would never be the last kernel to write to global
    // memory.  don't bother going through the store callback to
    // write global memory.

    T* outputs = output + outputs_offset;
    T* outputc = output + outputc_offset;

    if((is0 == 0) || (is0 * 2 == dim_0)) // simply write the element to output
    {
        outputs[0] = load_cb(input, input_offset, load_cb_data, nullptr);
        return;
    }

    if(is0 < hermitian_size)
    {
        T res      = load_cb(input, input_offset, load_cb_data, nullptr);
        outputs[0] = res;
        res.y      = -res.y;
        outputc[0] = res;
    }
}

template <typename T, unsigned int dim>
__global__ static void hermitian2complex_planar_kernel(const unsigned int hermitian_size,
                                                       unsigned int       dim_0,
                                                       unsigned int       dim_1,
                                                       unsigned int       dim_2,
                                                       unsigned int       stride_in0,
                                                       unsigned int       stride_in1,
                                                       unsigned int       stride_in2,
                                                       unsigned int       stride_in3,
                                                       unsigned int       stride_out0,
                                                       unsigned int       stride_out1,
                                                       unsigned int       stride_out2,
                                                       unsigned int       stride_out3,
                                                       real_type_t<T>* __restrict__ inputRe,
                                                       real_type_t<T>* __restrict__ inputIm,
                                                       T* __restrict__ output)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int stride_in[4]  = {stride_in0, stride_in1, stride_in2, stride_in3};
    const unsigned int stride_out[4] = {stride_out0, stride_out1, stride_out2, stride_out3};

    // start with batch offset
    size_t input_offset   = blockIdx.z * stride_in[dim];
    size_t outputs_offset = blockIdx.z * stride_out[dim]; // straight copy
    size_t outputc_offset = blockIdx.z * stride_out[dim]; // conjugate copy

    // straight copy indices
    size_t is0 = tid;
    size_t is1 = blockIdx.y % dim_1;
    size_t is2 = blockIdx.y / dim_1;

    // conjugate copy indices
    size_t ic0 = (is0 == 0) ? 0 : dim_0 - is0;
    size_t ic1 = (is1 == 0) ? 0 : dim_1 - is1;
    size_t ic2 = (is2 == 0) ? 0 : dim_2 - is2;

    input_offset += is2 * stride_in2 + is1 * stride_in1 + is0 * stride_in0;
    // notice for 1D,
    // blockIdx.y == 0 and
    // thus has no effect for
    // input_offset
    outputs_offset += is2 * stride_out2 + is1 * stride_out1 + is0 * stride_out0;
    outputc_offset += ic2 * stride_out2 + ic1 * stride_out1 + ic0 * stride_out0;

    inputRe += input_offset;
    inputIm += input_offset;
    T* outputs = output + outputs_offset;
    T* outputc = output + outputc_offset;

    if((is0 == 0) || (is0 * 2 == dim_0)) // simply write the element to output
    {
        outputs[0].x = inputRe[0];
        outputs[0].y = inputIm[0];
        return;
    }

    if(is0 < hermitian_size)
    {
        outputs[0].x = inputRe[0];
        outputs[0].y = inputIm[0];
        outputc[0].x = inputRe[0];
        outputc[0].y = -inputIm[0];
    }
}

#define HERM2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                                         \
    decltype(&hermitian2complex_kernel<TFLOAT, CallbackType::NONE, DIM>) kernel_func;       \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)                          \
        kernel_func = hermitian2complex_kernel<TFLOAT, CallbackType::USER_LOAD_STORE, DIM>; \
    else                                                                                    \
        kernel_func = hermitian2complex_kernel<TFLOAT, CallbackType::NONE, DIM>;            \
    hipLaunchKernelGGL(kernel_func,                                                         \
                       grid,                                                                \
                       threads,                                                             \
                       0,                                                                   \
                       rocfft_stream,                                                       \
                       hermitian_size,                                                      \
                       dim_0,                                                               \
                       dim_1,                                                               \
                       dim_2,                                                               \
                       kern_stride_in[0],                                                   \
                       kern_stride_in[1],                                                   \
                       kern_stride_in[2],                                                   \
                       kern_stride_in[3],                                                   \
                       kern_stride_out[0],                                                  \
                       kern_stride_out[1],                                                  \
                       kern_stride_out[2],                                                  \
                       kern_stride_out[3],                                                  \
                       static_cast<TFLOAT*>(input_buffer),                                  \
                       static_cast<TFLOAT*>(output_buffer),                                 \
                       data->callbacks.load_cb_fn,                                          \
                       data->callbacks.load_cb_data,                                        \
                       data->callbacks.load_cb_lds_bytes,                                   \
                       data->callbacks.store_cb_fn,                                         \
                       data->callbacks.store_cb_data);

// assign hermitian2complex function pointer given a float type
#define HERM2COMPLEX_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                   \
    {                                              \
        HERM2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                              \
    else if(dim == 2)                              \
    {                                              \
        HERM2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                              \
    else if(dim == 3)                              \
    {                                              \
        HERM2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                              \
    else                                           \
        throw std::runtime_error("invalid dimension in hermitian2complex");

#define HERM2COMPLEX_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                \
    decltype(&hermitian2complex_planar_kernel<TFLOAT, DIM>) kernel_func;  \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)        \
        kernel_func = hermitian2complex_planar_kernel<TFLOAT, DIM>;       \
    else                                                                  \
        kernel_func = hermitian2complex_planar_kernel<TFLOAT, DIM>;       \
    hipLaunchKernelGGL(kernel_func,                                       \
                       grid,                                              \
                       threads,                                           \
                       0,                                                 \
                       rocfft_stream,                                     \
                       hermitian_size,                                    \
                       dim_0,                                             \
                       dim_1,                                             \
                       dim_2,                                             \
                       kern_stride_in[0],                                 \
                       kern_stride_in[1],                                 \
                       kern_stride_in[2],                                 \
                       kern_stride_in[3],                                 \
                       kern_stride_out[0],                                \
                       kern_stride_out[1],                                \
                       kern_stride_out[2],                                \
                       kern_stride_out[3],                                \
                       static_cast<real_type_t<TFLOAT>*>(data->bufIn[0]), \
                       static_cast<real_type_t<TFLOAT>*>(data->bufIn[1]), \
                       static_cast<TFLOAT*>(output_buffer));

// assign hermitian2complex_planar function pointer given a float type
#define HERM2COMPLEX_PLANAR_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                          \
    {                                                     \
        HERM2COMPLEX_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                                     \
    else if(dim == 2)                                     \
    {                                                     \
        HERM2COMPLEX_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                                     \
    else if(dim == 3)                                     \
    {                                                     \
        HERM2COMPLEX_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                                     \
    else                                                  \
        throw std::runtime_error("invalid dimension in hermitian2complex_planar");

/// \brief auxiliary function
///   Read from input_buffer of hermitian structure into an output_buffer of regular
///   complex structure by padding 0.
ROCFFT_DEVICE_EXPORT void hermitian2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t dim_0          = data->node->length[0]; // dim_0 is the innermost dimension
    size_t hermitian_size = dim_0 / 2 + 1;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    size_t dim            = data->node->length.size();
    if(dim > 1)
    {
        for(int i = 1; i < dim; i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;

    size_t blocks = (hermitian_size - 1) / LAUNCH_BOUNDS_R2C_C2R_KERNEL + 1;

    if(dim > 3)
        throw std::runtime_error("Error: dimension larger than 3, which is not handled");

    size_t dim_1 = 1, dim_2 = 1;
    if(data->node->length.size() >= 2)
        dim_1 = data->node->length[1];
    if(data->node->length.size() == 3)
        dim_2 = data->node->length[2];

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    //   float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
    //   hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
    //             hipMemcpyDeviceToHost);
    //   printf("herm size %d, dim0 %d, dim1 %d, dim2 %d\n", hermitian_size, dim_0,
    //          dim_1, dim_2);
    //   printf("input_stride %d output_stride %d input_distance %d output_distance
    // %d\n", input_stride, output_stride, input_distance, output_distance);
    //   //for(size_t j=0;j<data->node->length[1]; j++)
    //   size_t j = 0;
    //   {
    //       for(size_t i=0; i<input_stride; i++)
    //       {
    //           printf("kernel input[%zu][%zu]=(%f, %f) \n", j, i, tmp[j*input_stride
    //                                                                  + i].x, tmp[j*input_stride + i].y);
    //       }
    //   }

    // explode strides out to pass to the kernel
    std::array<size_t, 4> kern_stride_in{1, 1, 1, 1};
    std::array<size_t, 4> kern_stride_out{1, 1, 1, 1};

    std::copy(data->node->inStride.begin(), data->node->inStride.end(), kern_stride_in.begin());
    kern_stride_in[dim] = data->node->iDist;
    std::copy(data->node->outStride.begin(), data->node->outStride.end(), kern_stride_out.begin());
    kern_stride_out[dim] = data->node->oDist;

    if(data->node->inArrayType == rocfft_array_type_hermitian_interleaved)
    {
        if(precision == rocfft_precision_single)
        {
            HERM2COMPLEX_KERNEL_LAUNCH(float2);
        }
        else
        {
            HERM2COMPLEX_KERNEL_LAUNCH(double2);
        }
    }
    else if(data->node->inArrayType == rocfft_array_type_hermitian_planar)
    {
        if(precision == rocfft_precision_single)
        {
            HERM2COMPLEX_PLANAR_KERNEL_LAUNCH(float2);
        }
        else
        {
            HERM2COMPLEX_PLANAR_KERNEL_LAUNCH(double2);
        }
    }

    // float2* tmpo; tmpo = (float2*)malloc(sizeof(float2)*output_distance*batch);
    // hipMemcpy(tmpo, output_buffer, sizeof(float2)*output_distance*batch,
    //           hipMemcpyDeviceToHost);

    // {
    //     for(size_t i=0; i<output_stride; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmpo[j*output_stride + i].x, tmpo[j*output_stride + i].y);
    //     }
    // }
}
