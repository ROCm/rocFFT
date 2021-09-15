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
#include <numeric>

template <typename Tcomplex, CallbackType cbtype, unsigned int dim>
__global__ static void __launch_bounds__(LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    real2complex_kernel(unsigned int lengths0,
                        unsigned int lengths1,
                        unsigned int lengths2,
                        unsigned int stride_in0,
                        unsigned int stride_in1,
                        unsigned int stride_in2,
                        unsigned int stride_in3,
                        unsigned int stride_out0,
                        unsigned int stride_out1,
                        unsigned int stride_out2,
                        unsigned int stride_out3,
                        const real_type_t<Tcomplex>* __restrict__ input,
                        Tcomplex* __restrict__ output,
                        void* __restrict__ load_cb_fn,
                        void* __restrict__ load_cb_data,
                        uint32_t load_cb_lds_bytes,
                        void* __restrict__ store_cb_fn,
                        void* __restrict__ store_cb_data)
{
    unsigned int idx_0 = blockIdx.x * blockDim.x + threadIdx.x;

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
        // we would do real2complex at the beginning of an R2C
        // transform, so it would never be the last kernel to write
        // to global memory.  don't bother going through the store cb
        // to write global memory.
        auto load_cb = get_load_cb<real_type_t<Tcomplex>, cbtype>(load_cb_fn);

        const auto inputIdx  = offset_in + idx_0 * stride_in[0];
        const auto outputIdx = offset_out + idx_0 * stride_out[0];

        // callback is allowed to modify input, though it's const for us
        output[outputIdx].x
            = load_cb(const_cast<real_type_t<Tcomplex>*>(input), inputIdx, load_cb_data, nullptr);
        output[outputIdx].y = 0.0;
    }
}

#define REAL2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                                    \
    decltype(&real2complex_kernel<TFLOAT, CallbackType::NONE, DIM>) kernel_func;       \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)                     \
        kernel_func = real2complex_kernel<TFLOAT, CallbackType::USER_LOAD_STORE, DIM>; \
    else                                                                               \
        kernel_func = real2complex_kernel<TFLOAT, CallbackType::NONE, DIM>;            \
    hipLaunchKernelGGL(kernel_func,                                                    \
                       grid,                                                           \
                       threads,                                                        \
                       0,                                                              \
                       rocfft_stream,                                                  \
                       kern_lengths[0],                                                \
                       kern_lengths[1],                                                \
                       kern_lengths[2],                                                \
                       kern_stride_in[0],                                              \
                       kern_stride_in[1],                                              \
                       kern_stride_in[2],                                              \
                       kern_stride_in[3],                                              \
                       kern_stride_out[0],                                             \
                       kern_stride_out[1],                                             \
                       kern_stride_out[2],                                             \
                       kern_stride_out[3],                                             \
                       static_cast<real_type_t<TFLOAT>*>(input_buffer),                \
                       static_cast<TFLOAT*>(output_buffer),                            \
                       data->callbacks.load_cb_fn,                                     \
                       data->callbacks.load_cb_data,                                   \
                       data->callbacks.load_cb_lds_bytes,                              \
                       data->callbacks.store_cb_fn,                                    \
                       data->callbacks.store_cb_data);

// assign real2complex function pointer given a float type
#define REAL2COMPLEX_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                   \
    {                                              \
        REAL2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                              \
    else if(dim == 2)                              \
    {                                              \
        REAL2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                              \
    else if(dim == 3)                              \
    {                                              \
        REAL2COMPLEX_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                              \
    else                                           \
        throw std::runtime_error("invalid dimension in real2complex");

/// \brief auxiliary function
///    convert a real vector into a complex one by padding the imaginary part with  0.
ROCFFT_DEVICE_EXPORT void real2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

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

    // TODO: verify with API that high_dimension and batch aren't too big.

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

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
        REAL2COMPLEX_KERNEL_LAUNCH(float2);
    }
    else
    {
        REAL2COMPLEX_KERNEL_LAUNCH(double2);
    }

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*output_distance*batch);
    // hipMemcpy(tmp, output_buffer, sizeof(float2)*output_distance*batch,
    //           hipMemcpyDeviceToHost);

    // for(size_t j=0;j<data->node->length[1]; j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
    //     }
    // }
}

// The complex to hermitian simple copy kernel for interleaved format
template <typename Tcomplex, CallbackType cbtype, unsigned int dim>
__global__ void __launch_bounds__(LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    complex2hermitian_kernel(unsigned int lengths0,
                             unsigned int lengths1,
                             unsigned int lengths2,
                             unsigned int stride_in0,
                             unsigned int stride_in1,
                             unsigned int stride_in2,
                             unsigned int stride_in3,
                             unsigned int stride_out0,
                             unsigned int stride_out1,
                             unsigned int stride_out2,
                             unsigned int stride_out3,
                             const Tcomplex* __restrict__ input,
                             Tcomplex* __restrict__ output,
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

    // only read and write the first [length0/2+1] elements due to conjugate redundancy
    if(idx_0 < (1 + lengths[0] / 2))
    {
        // we would do complex2hermitian at the end of an R2C
        // transform, so it would never be the first kernel to read
        // from global memory.  don't bother going through the load
        // callback to read global memory.

        const auto inputIdx  = offset_in + idx_0 * stride_in[0];
        const auto outputIdx = offset_out + idx_0 * stride_out[0];

        auto store_cb = get_store_cb<Tcomplex, cbtype>(store_cb_fn);
        store_cb(output, outputIdx, input[inputIdx], store_cb_data, nullptr);
    }
}

// The planar overload function of the above interleaved one
template <typename Tcomplex, unsigned int dim>
__global__ static void __launch_bounds__(LAUNCH_BOUNDS_R2C_C2R_KERNEL)
    complex2hermitian_planar_kernel(unsigned int lengths0,
                                    unsigned int lengths1,
                                    unsigned int lengths2,
                                    unsigned int stride_in0,
                                    unsigned int stride_in1,
                                    unsigned int stride_in2,
                                    unsigned int stride_in3,
                                    unsigned int stride_out0,
                                    unsigned int stride_out1,
                                    unsigned int stride_out2,
                                    unsigned int stride_out3,
                                    const Tcomplex* __restrict__ input,
                                    real_type_t<Tcomplex>* __restrict__ outputRe,
                                    real_type_t<Tcomplex>* __restrict__ outputIm)
{
    size_t idx_0 = blockIdx.x * LAUNCH_BOUNDS_R2C_C2R_KERNEL + threadIdx.x;

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

    // only read and write the first [length0/2+1] elements due to conjugate redundancy
    if(idx_0 < (1 + lengths[0] / 2))
    {
        const auto inputIdx  = offset_in + idx_0 * stride_in[0];
        const auto outputIdx = offset_out + idx_0 * stride_out[0];

        outputRe[outputIdx] = input[inputIdx].x;
        outputIm[outputIdx] = input[inputIdx].y;
    }
}

#define COMPLEX2HERM_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                                         \
    decltype(&complex2hermitian_kernel<TFLOAT, CallbackType::NONE, DIM>) kernel_func;       \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)                          \
        kernel_func = complex2hermitian_kernel<TFLOAT, CallbackType::USER_LOAD_STORE, DIM>; \
    else                                                                                    \
        kernel_func = complex2hermitian_kernel<TFLOAT, CallbackType::NONE, DIM>;            \
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
                       static_cast<TFLOAT*>(output_buffer),                                 \
                       data->callbacks.load_cb_fn,                                          \
                       data->callbacks.load_cb_data,                                        \
                       data->callbacks.load_cb_lds_bytes,                                   \
                       data->callbacks.store_cb_fn,                                         \
                       data->callbacks.store_cb_data);

// assign complex2hermitian function pointer given a float type
#define COMPLEX2HERM_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                   \
    {                                              \
        COMPLEX2HERM_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                              \
    else if(dim == 2)                              \
    {                                              \
        COMPLEX2HERM_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                              \
    else if(dim == 3)                              \
    {                                              \
        COMPLEX2HERM_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                              \
    else                                           \
        throw std::runtime_error("invalid dimension in complex2hermitian");

#define COMPLEX2HERM_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, DIM)                 \
    decltype(&complex2hermitian_planar_kernel<TFLOAT, DIM>) kernel_func;   \
    if(data->get_callback_type() == CallbackType::USER_LOAD_STORE)         \
        kernel_func = complex2hermitian_planar_kernel<TFLOAT, DIM>;        \
    else                                                                   \
        kernel_func = complex2hermitian_planar_kernel<TFLOAT, DIM>;        \
    hipLaunchKernelGGL(kernel_func,                                        \
                       grid,                                               \
                       threads,                                            \
                       0,                                                  \
                       rocfft_stream,                                      \
                       kern_lengths[0],                                    \
                       kern_lengths[1],                                    \
                       kern_lengths[2],                                    \
                       kern_stride_in[0],                                  \
                       kern_stride_in[1],                                  \
                       kern_stride_in[2],                                  \
                       kern_stride_in[3],                                  \
                       kern_stride_out[0],                                 \
                       kern_stride_out[1],                                 \
                       kern_stride_out[2],                                 \
                       kern_stride_out[3],                                 \
                       static_cast<TFLOAT*>(input_buffer),                 \
                       static_cast<real_type_t<TFLOAT>*>(data->bufOut[0]), \
                       static_cast<real_type_t<TFLOAT>*>(data->bufOut[1]));

// assign complex2hermitian_planar function pointer given a float type
#define COMPLEX2HERM_PLANAR_KERNEL_LAUNCH(TFLOAT)         \
    if(dim == 1)                                          \
    {                                                     \
        COMPLEX2HERM_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 1); \
    }                                                     \
    else if(dim == 2)                                     \
    {                                                     \
        COMPLEX2HERM_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 2); \
    }                                                     \
    else if(dim == 3)                                     \
    {                                                     \
        COMPLEX2HERM_PLANAR_KERNEL_LAUNCH_DIM(TFLOAT, 3); \
    }                                                     \
    else                                                  \
        throw std::runtime_error("invalid dimension in complex2hermitian_planar");

/// \brief auxiliary function
///   read from input_buffer and store the first  [1 + input_size/2] elements to
///   the output_buffer
ROCFFT_DEVICE_EXPORT void complex2hermitian(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0]; // input_size is the innermost dimension

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

    // TODO: verify with API that high_dimension and batch aren't too big.

    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1);

    hipStream_t rocfft_stream = data->rocfft_stream;

    // float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
    // hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
    //           hipMemcpyDeviceToHost);

    // for(size_t j=0;j<data->node->length[1]; j++)
    // {
    //     for(size_t i=0; i<data->node->length[0]; i++)
    //     {
    //         printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
    //                tmp[j*data->node->outStride[1] + i].x, tmp[j*data->node->outStride[1] + i].y);
    //     }
    // }

    // explode lengths/strides out to pass to the kernel
    std::array<size_t, 3> kern_lengths{1, 1, 1};
    std::array<size_t, 4> kern_stride_in{1, 1, 1, 1};
    std::array<size_t, 4> kern_stride_out{1, 1, 1, 1};

    std::copy(data->node->length.begin(), data->node->length.end(), kern_lengths.begin());
    std::copy(data->node->inStride.begin(), data->node->inStride.end(), kern_stride_in.begin());
    kern_stride_in[dim] = data->node->iDist;
    std::copy(data->node->outStride.begin(), data->node->outStride.end(), kern_stride_out.begin());
    kern_stride_out[dim] = data->node->oDist;

    // TODO: check the input type
    if(data->node->outArrayType == rocfft_array_type_hermitian_interleaved)
    {
        if(precision == rocfft_precision_single)
        {
            COMPLEX2HERM_KERNEL_LAUNCH(float2);
        }
        else
        {
            COMPLEX2HERM_KERNEL_LAUNCH(double2);
        }
    }
    else if(data->node->outArrayType == rocfft_array_type_hermitian_planar)
    {
        if(precision == rocfft_precision_single)
        {
            COMPLEX2HERM_PLANAR_KERNEL_LAUNCH(float2);
        }
        else
        {
            COMPLEX2HERM_PLANAR_KERNEL_LAUNCH(double2);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported output format in complex2hermitian kernel");
    }
}
