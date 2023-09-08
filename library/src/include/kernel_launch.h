/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef KERNEL_LAUNCH_SINGLE
#define KERNEL_LAUNCH_SINGLE

#define FN_PRFX(X) rocfft_internal_##X
#ifndef __clang__
#include "error.h"
#endif
#include "../../../shared/array_predicate.h"
#include "../../../shared/rocfft_hip.h"
#include "../device/kernels/callback.h"
#include "kargs.h"
#include "rocfft/rocfft.h"
#include "tree_node.h"
#include <iostream>

using log_func_t = void (*)(const char* msg);

// Shim for logging occupancy when launch the kernel
//
// NB: to avoid link to logging.h in device lib, we need to pass log_func as a function ptr
#define hipLaunchKernelGGL_shim(                                                                   \
    log_func, kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)                       \
    hipLaunchKernelGGL(                                                                            \
        (kernelName), (numBlocks), (numThreads), (memPerBlock), (streamId), __VA_ARGS__);          \
    if(log_func != nullptr)                                                                        \
    {                                                                                              \
        int         max_blocks_per_sm;                                                             \
        hipError_t  ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,         \
                                                                      kernelName,                 \
                                                                      numThreads.x * numThreads.y \
                                                                          * numThreads.z,         \
                                                                      memPerBlock);               \
        std::string s;                                                                             \
        if(ret == hipSuccess)                                                                      \
            s = "Kernel occupancy: " + std::to_string(max_blocks_per_sm) + "\n";                   \
        else                                                                                       \
            s = "Can not retrieve occupancy info.\n";                                              \
        log_func(s.c_str());                                                                       \
    }

// FIXME: documentation
struct DeviceCallIn
{
    TreeNode* node;
    void*     bufIn[2];
    void*     bufOut[2];

    // some kernels require a temp buffer that's neither input nor output
    void* bufTemp = nullptr;

    hipStream_t     rocfft_stream;
    GridParam       gridParam;
    hipDeviceProp_t deviceProp;

    UserCallbacks callbacks;

    CallbackType get_callback_type() const
    {
        if(callbacks.load_cb_fn || callbacks.store_cb_fn)
            return CallbackType::USER_LOAD_STORE;
        else
            return CallbackType::NONE;
    }

    log_func_t log_func;
};

// FIXME: documentation
struct DeviceCallOut
{

    int err;
};

/* Naming convention

dfn – device function caller (just a prefix, though actually GPU kernel
function)

sp (dp) – single (double) precision

ip – in-place

op - out-of-place

ci – complex-interleaved (format of input buffer)

ci – complex-interleaved (format of output buffer)

stoc – stockham fft kernel
bcc - block column column

1(2) – one (two) dimension data from kernel viewpoint, but 2D may transform into
1D. e.g  64*128(2D) = 8192(1D)

1024, 64_128 – length of fft on each dimension

*/

/*
    TODO:
        - compress the below code
        - refactor the code to support batched r2c/c2r
 */

// base args for out-of-place includes extra strides for output
#define KERNEL_BASE_ARGS_IP(PRECISION)                                                    \
    const PRECISION* __restrict__, const size_t, const size_t* __restrict__,              \
        const size_t* __restrict__, const size_t, const unsigned int, void* __restrict__, \
        void* __restrict__, uint32_t, void* __restrict__, void* __restrict__

#define KERNEL_BASE_ARGS_OP(PRECISION)                                                            \
    const PRECISION* __restrict__, const size_t, const size_t* __restrict__,                      \
        const size_t* __restrict__, const size_t* __restrict__, const size_t, const unsigned int, \
        void* __restrict__, void* __restrict__, uint32_t, void* __restrict__, void* __restrict__

#define GET_KERNEL_FUNC_CBTYPE(FWD, BACK, PRECISION, CBTYPE, DRTYPE)                              \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                             \
    {                                                                                             \
        if(data->node->direction == -1)                                                           \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::Real2C_POST)                                   \
                kernel_func = FWD<PRECISION, SB_UNIT, EmbeddedType::Real2C_POST, CBTYPE, DRTYPE>; \
            else                                                                                  \
                kernel_func = FWD<PRECISION, SB_UNIT, EmbeddedType::NONE, CBTYPE, DRTYPE>;        \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::C2Real_PRE)                                    \
                kernel_func = BACK<PRECISION, SB_UNIT, EmbeddedType::C2Real_PRE, CBTYPE, DRTYPE>; \
            else                                                                                  \
                kernel_func = BACK<PRECISION, SB_UNIT, EmbeddedType::NONE, CBTYPE, DRTYPE>;       \
        }                                                                                         \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        if(data->node->direction == -1)                                                           \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::Real2C_POST)                                   \
                kernel_func                                                                       \
                    = FWD<PRECISION, SB_NONUNIT, EmbeddedType::Real2C_POST, CBTYPE, DRTYPE>;      \
            else                                                                                  \
                kernel_func = FWD<PRECISION, SB_NONUNIT, EmbeddedType::NONE, CBTYPE, DRTYPE>;     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::C2Real_PRE)                                    \
                kernel_func                                                                       \
                    = BACK<PRECISION, SB_NONUNIT, EmbeddedType::C2Real_PRE, CBTYPE, DRTYPE>;      \
            else                                                                                  \
                kernel_func = BACK<PRECISION, SB_NONUNIT, EmbeddedType::NONE, CBTYPE, DRTYPE>;    \
        }                                                                                         \
    }

#define GET_KERNEL_FUNC(FWD, BACK, PRECISION, BASE_ARGS, ...)         \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr; \
    GET_KERNEL_FUNC_CBTYPE(                                           \
        FWD, BACK, PRECISION, CallbackType::NONE, DirectRegType::TRY_ENABLE_IF_SUPPORT)

#define GET_KERNEL_FUNC_CB(FWD, BACK, PRECISION, BASE_ARGS, ...)                            \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                       \
    if(data->get_callback_type() == CallbackType::NONE)                                     \
        GET_KERNEL_FUNC_CBTYPE(                                                             \
            FWD, BACK, PRECISION, CallbackType::NONE, DirectRegType::TRY_ENABLE_IF_SUPPORT) \
    else                                                                                    \
        GET_KERNEL_FUNC_CBTYPE(FWD,                                                         \
                               BACK,                                                        \
                               PRECISION,                                                   \
                               CallbackType::USER_LOAD_STORE,                               \
                               DirectRegType::TRY_ENABLE_IF_SUPPORT)

// SBRC has COL_DIM, TRANSPOSE_TYPE template args and is always out-of-place, and ebtype NONE
#define GET_KERNEL_FUNC_CBTYPE_SBRC(FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, CBTYPE, DRTYPE) \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                              \
    {                                                                                              \
        if(data->node->direction == -1)                                                            \
        {                                                                                          \
            kernel_func = FWD<PRECISION,                                                           \
                              SB_UNIT,                                                             \
                              COL_DIM,                                                             \
                              TRANSPOSE_TYPE,                                                      \
                              EmbeddedType::NONE,                                                  \
                              CBTYPE,                                                              \
                              DRTYPE>;                                                             \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            kernel_func = BACK<PRECISION,                                                          \
                               SB_UNIT,                                                            \
                               COL_DIM,                                                            \
                               TRANSPOSE_TYPE,                                                     \
                               EmbeddedType::NONE,                                                 \
                               CBTYPE,                                                             \
                               DRTYPE>;                                                            \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(data->node->direction == -1)                                                            \
        {                                                                                          \
            kernel_func = FWD<PRECISION,                                                           \
                              SB_NONUNIT,                                                          \
                              COL_DIM,                                                             \
                              TRANSPOSE_TYPE,                                                      \
                              EmbeddedType::NONE,                                                  \
                              CBTYPE,                                                              \
                              DRTYPE>;                                                             \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            kernel_func = BACK<PRECISION,                                                          \
                               SB_NONUNIT,                                                         \
                               COL_DIM,                                                            \
                               TRANSPOSE_TYPE,                                                     \
                               EmbeddedType::NONE,                                                 \
                               CBTYPE,                                                             \
                               DRTYPE>;                                                            \
        }                                                                                          \
    }

#define GET_KERNEL_FUNC_SBRC_CB(FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                          \
    if(data->get_callback_type() == CallbackType::NONE)                                        \
    {                                                                                          \
        if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)                    \
            GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                   \
                                        BACK,                                                  \
                                        PRECISION,                                             \
                                        COL_DIM,                                               \
                                        TRANSPOSE_TYPE,                                        \
                                        CallbackType::NONE,                                    \
                                        DirectRegType::TRY_ENABLE_IF_SUPPORT)                  \
        else                                                                                   \
            GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                   \
                                        BACK,                                                  \
                                        PRECISION,                                             \
                                        COL_DIM,                                               \
                                        TRANSPOSE_TYPE,                                        \
                                        CallbackType::NONE,                                    \
                                        DirectRegType::FORCE_OFF_OR_NOT_SUPPORT)               \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)                    \
            GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                   \
                                        BACK,                                                  \
                                        PRECISION,                                             \
                                        COL_DIM,                                               \
                                        TRANSPOSE_TYPE,                                        \
                                        CallbackType::USER_LOAD_STORE,                         \
                                        DirectRegType::TRY_ENABLE_IF_SUPPORT)                  \
        else                                                                                   \
            GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                   \
                                        BACK,                                                  \
                                        PRECISION,                                             \
                                        COL_DIM,                                               \
                                        TRANSPOSE_TYPE,                                        \
                                        CallbackType::USER_LOAD_STORE,                         \
                                        DirectRegType::FORCE_OFF_OR_NOT_SUPPORT)               \
    }

#define GET_KERNEL_FUNC_SBRC(FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                       \
    if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)                     \
        GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                    \
                                    BACK,                                                   \
                                    PRECISION,                                              \
                                    COL_DIM,                                                \
                                    TRANSPOSE_TYPE,                                         \
                                    CallbackType::NONE,                                     \
                                    DirectRegType::TRY_ENABLE_IF_SUPPORT)                   \
    else                                                                                    \
        GET_KERNEL_FUNC_CBTYPE_SBRC(FWD,                                                    \
                                    BACK,                                                   \
                                    PRECISION,                                              \
                                    COL_DIM,                                                \
                                    TRANSPOSE_TYPE,                                         \
                                    CallbackType::NONE,                                     \
                                    DirectRegType::FORCE_OFF_OR_NOT_SUPPORT)

// SBCR has C2Real_PRE but no Real2C_POST
#define GET_KERNEL_FUNC_CBTYPE_SBCR(FWD, BACK, PRECISION, CBTYPE, DRTYPE, INTRINSIC)              \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                             \
    {                                                                                             \
        if(data->node->direction == -1)                                                           \
        {                                                                                         \
            kernel_func = FWD<PRECISION, SB_UNIT, EmbeddedType::NONE, CBTYPE, DRTYPE, INTRINSIC>; \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::C2Real_PRE)                                    \
                kernel_func = BACK<PRECISION,                                                     \
                                   SB_UNIT,                                                       \
                                   EmbeddedType::C2Real_PRE,                                      \
                                   CBTYPE,                                                        \
                                   DRTYPE,                                                        \
                                   INTRINSIC>;                                                    \
            else                                                                                  \
                kernel_func                                                                       \
                    = BACK<PRECISION, SB_UNIT, EmbeddedType::NONE, CBTYPE, DRTYPE, INTRINSIC>;    \
        }                                                                                         \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
        if(data->node->direction == -1)                                                           \
        {                                                                                         \
            kernel_func                                                                           \
                = FWD<PRECISION, SB_NONUNIT, EmbeddedType::NONE, CBTYPE, DRTYPE, INTRINSIC>;      \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(data->node->ebtype == EmbeddedType::C2Real_PRE)                                    \
                kernel_func = BACK<PRECISION,                                                     \
                                   SB_NONUNIT,                                                    \
                                   EmbeddedType::C2Real_PRE,                                      \
                                   CBTYPE,                                                        \
                                   DRTYPE,                                                        \
                                   INTRINSIC>;                                                    \
            else                                                                                  \
                kernel_func                                                                       \
                    = BACK<PRECISION, SB_NONUNIT, EmbeddedType::NONE, CBTYPE, DRTYPE, INTRINSIC>; \
        }                                                                                         \
    }

// SBCR is always out-of-place, and intrisic is either disable_ or enable_ both
#define GET_KERNEL_FUNC_SBCR_CB(FWD, BACK, PRECISION, BASE_ARGS, ...)             \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;             \
    if(data->get_callback_type() == CallbackType::NONE)                           \
    {                                                                             \
        if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)       \
        {                                                                         \
            if(data->node->intrinsicMode == IntrinsicAccessType::ENABLE_BOTH)     \
                GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                  \
                                            BACK,                                 \
                                            PRECISION,                            \
                                            CallbackType::NONE,                   \
                                            DirectRegType::TRY_ENABLE_IF_SUPPORT, \
                                            IntrinsicAccessType::ENABLE_BOTH)     \
            else                                                                  \
                GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                  \
                                            BACK,                                 \
                                            PRECISION,                            \
                                            CallbackType::NONE,                   \
                                            DirectRegType::TRY_ENABLE_IF_SUPPORT, \
                                            IntrinsicAccessType::DISABLE_BOTH)    \
        }                                                                         \
        else                                                                      \
            GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                      \
                                        BACK,                                     \
                                        PRECISION,                                \
                                        CallbackType::NONE,                       \
                                        DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,  \
                                        IntrinsicAccessType::DISABLE_BOTH)        \
    }                                                                             \
    else                                                                          \
    {                                                                             \
        if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)       \
            GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                      \
                                        BACK,                                     \
                                        PRECISION,                                \
                                        CallbackType::USER_LOAD_STORE,            \
                                        DirectRegType::TRY_ENABLE_IF_SUPPORT,     \
                                        IntrinsicAccessType::DISABLE_BOTH)        \
        else                                                                      \
            GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                      \
                                        BACK,                                     \
                                        PRECISION,                                \
                                        CallbackType::USER_LOAD_STORE,            \
                                        DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,  \
                                        IntrinsicAccessType::DISABLE_BOTH)        \
    }

#define GET_KERNEL_FUNC_SBCR(FWD, BACK, PRECISION, BASE_ARGS, ...)            \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;         \
    if(data->node->dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT)       \
    {                                                                         \
        if(data->node->intrinsicMode == IntrinsicAccessType::ENABLE_BOTH)     \
            GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                  \
                                        BACK,                                 \
                                        PRECISION,                            \
                                        CallbackType::NONE,                   \
                                        DirectRegType::TRY_ENABLE_IF_SUPPORT, \
                                        IntrinsicAccessType::ENABLE_BOTH)     \
        else                                                                  \
            GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                  \
                                        BACK,                                 \
                                        PRECISION,                            \
                                        CallbackType::NONE,                   \
                                        DirectRegType::TRY_ENABLE_IF_SUPPORT, \
                                        IntrinsicAccessType::DISABLE_BOTH)    \
    }                                                                         \
    else                                                                      \
        GET_KERNEL_FUNC_CBTYPE_SBCR(FWD,                                      \
                                    BACK,                                     \
                                    PRECISION,                                \
                                    CallbackType::NONE,                       \
                                    DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,  \
                                    IntrinsicAccessType::DISABLE_BOTH)

#define POWX_SMALL_GENERATOR(FUNCTION_NAME,                                        \
                             IP_FWD_KERN_NAME,                                     \
                             IP_BACK_KERN_NAME,                                    \
                             OP_FWD_KERN_NAME,                                     \
                             OP_BACK_KERN_NAME,                                    \
                             PRECISION)                                            \
    ROCFFT_DEVICE_EXPORT void FUNCTION_NAME(const void* data_p, void* back_p)      \
    {                                                                              \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                       \
        hipStream_t   rocfft_stream = data->rocfft_stream;                         \
                                                                                   \
        if(data->node->placement == rocfft_placement_inplace)                      \
        {                                                                          \
            if(array_type_is_interleaved(data->node->inArrayType)                  \
               && array_type_is_interleaved(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_CB(IP_FWD_KERN_NAME,                               \
                                   IP_BACK_KERN_NAME,                              \
                                   PRECISION,                                      \
                                   KERNEL_BASE_ARGS_IP,                            \
                                   PRECISION* __restrict__);                       \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0]);               \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC(IP_FWD_KERN_NAME,                                  \
                                IP_BACK_KERN_NAME,                                 \
                                PRECISION,                                         \
                                KERNEL_BASE_ARGS_IP,                               \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__);             \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1]);  \
            }                                                                      \
        }                                                                          \
        else /* out of place */                                                    \
        {                                                                          \
            if(array_type_is_interleaved(data->node->inArrayType)                  \
               && array_type_is_interleaved(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_CB(OP_FWD_KERN_NAME,                               \
                                   OP_BACK_KERN_NAME,                              \
                                   PRECISION,                                      \
                                   KERNEL_BASE_ARGS_OP,                            \
                                   PRECISION* __restrict__,                        \
                                   PRECISION* __restrict__);                       \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0],                \
                                        (PRECISION*)data->bufOut[0]);              \
            }                                                                      \
            else if(array_type_is_interleaved(data->node->inArrayType)             \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                  \
                                OP_BACK_KERN_NAME,                                 \
                                PRECISION,                                         \
                                KERNEL_BASE_ARGS_OP,                               \
                                PRECISION* __restrict__,                           \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__);             \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0],                \
                                        (real_type_t<PRECISION>*)data->bufOut[0],  \
                                        (real_type_t<PRECISION>*)data->bufOut[1]); \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_interleaved(data->node->outArrayType))        \
            {                                                                      \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                  \
                                OP_BACK_KERN_NAME,                                 \
                                PRECISION,                                         \
                                KERNEL_BASE_ARGS_OP,                               \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__,              \
                                PRECISION* __restrict__);                          \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1],   \
                                        (PRECISION*)data->bufOut[0]);              \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                  \
                                OP_BACK_KERN_NAME,                                 \
                                PRECISION,                                         \
                                KERNEL_BASE_ARGS_OP,                               \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__,              \
                                real_type_t<PRECISION>* __restrict__);             \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        data->node->batch,                         \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1],   \
                                        (real_type_t<PRECISION>*)data->bufOut[0],  \
                                        (real_type_t<PRECISION>*)data->bufOut[1]); \
            }                                                                      \
        }                                                                          \
    }

#define POWX_LARGE_SBCC_GENERATOR(FUNCTION_NAME,                                   \
                                  IP_FWD_KERN_NAME,                                \
                                  IP_BACK_KERN_NAME,                               \
                                  OP_FWD_KERN_NAME,                                \
                                  OP_BACK_KERN_NAME,                               \
                                  PRECISION)                                       \
    ROCFFT_DEVICE_EXPORT void FUNCTION_NAME(const void* data_p, void* back_p)      \
    {                                                                              \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                       \
        hipStream_t   rocfft_stream = data->rocfft_stream;                         \
                                                                                   \
        const size_t batch = data->node->batch;                                    \
                                                                                   \
        if(data->node->placement == rocfft_placement_inplace)                      \
        {                                                                          \
            if(array_type_is_interleaved(data->node->inArrayType)                  \
               && array_type_is_interleaved(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC_CB(IP_FWD_KERN_NAME,                          \
                                        IP_BACK_KERN_NAME,                         \
                                        PRECISION,                                 \
                                        KERNEL_BASE_ARGS_IP_SBCC,                  \
                                        PRECISION* __restrict__);                  \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0]);               \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC(IP_FWD_KERN_NAME,                             \
                                     IP_BACK_KERN_NAME,                            \
                                     PRECISION,                                    \
                                     KERNEL_BASE_ARGS_IP_SBCC,                     \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__);        \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1]);  \
            }                                                                      \
        }                                                                          \
        else                                                                       \
        {                                                                          \
            if(array_type_is_interleaved(data->node->inArrayType)                  \
               && array_type_is_interleaved(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC_CB(OP_FWD_KERN_NAME,                          \
                                        OP_BACK_KERN_NAME,                         \
                                        PRECISION,                                 \
                                        KERNEL_BASE_ARGS_OP_SBCC,                  \
                                        PRECISION* __restrict__,                   \
                                        PRECISION* __restrict__);                  \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0],                \
                                        (PRECISION*)data->bufOut[0]);              \
            }                                                                      \
            else if(array_type_is_interleaved(data->node->inArrayType)             \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                             \
                                     OP_BACK_KERN_NAME,                            \
                                     PRECISION,                                    \
                                     KERNEL_BASE_ARGS_OP_SBCC,                     \
                                     PRECISION* __restrict__,                      \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__);        \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (PRECISION*)data->bufIn[0],                \
                                        (real_type_t<PRECISION>*)data->bufOut[0],  \
                                        (real_type_t<PRECISION>*)data->bufOut[1]); \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_interleaved(data->node->outArrayType))        \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                             \
                                     OP_BACK_KERN_NAME,                            \
                                     PRECISION,                                    \
                                     KERNEL_BASE_ARGS_OP_SBCC,                     \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     PRECISION* __restrict__);                     \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1],   \
                                        (PRECISION*)data->bufOut[0]);              \
            }                                                                      \
            else if(array_type_is_planar(data->node->inArrayType)                  \
                    && array_type_is_planar(data->node->outArrayType))             \
            {                                                                      \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                             \
                                     OP_BACK_KERN_NAME,                            \
                                     PRECISION,                                    \
                                     KERNEL_BASE_ARGS_OP_SBCC,                     \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__,         \
                                     real_type_t<PRECISION>* __restrict__);        \
                hipLaunchKernelGGL_shim(data->log_func,                            \
                                        kernel_func,                               \
                                        dim3(data->gridParam.b_x),                 \
                                        dim3(data->gridParam.wgs_x),               \
                                        data->gridParam.lds_bytes,                 \
                                        rocfft_stream,                             \
                                        (PRECISION*)data->node->twiddles,          \
                                        (PRECISION*)data->node->twiddles_large,    \
                                        data->node->length.size(),                 \
                                        kargs_lengths(data->node->devKernArg),     \
                                        kargs_stride_in(data->node->devKernArg),   \
                                        kargs_stride_out(data->node->devKernArg),  \
                                        batch,                                     \
                                        data->node->lds_padding,                   \
                                        data->callbacks.load_cb_fn,                \
                                        data->callbacks.load_cb_data,              \
                                        data->callbacks.load_cb_lds_bytes,         \
                                        data->callbacks.store_cb_fn,               \
                                        data->callbacks.store_cb_data,             \
                                        (real_type_t<PRECISION>*)data->bufIn[0],   \
                                        (real_type_t<PRECISION>*)data->bufIn[1],   \
                                        (real_type_t<PRECISION>*)data->bufOut[0],  \
                                        (real_type_t<PRECISION>*)data->bufOut[1]); \
            }                                                                      \
        }                                                                          \
    }

#define POWX_LARGE_SBRC_GENERATOR(                                                    \
    FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION, COL_DIM, TRANSPOSE_TYPE) \
    ROCFFT_DEVICE_EXPORT void FUNCTION_NAME(const void* data_p, void* back_p)         \
    {                                                                                 \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                          \
        hipStream_t   rocfft_stream = data->rocfft_stream;                            \
                                                                                      \
        const size_t batch = data->node->batch;                                       \
                                                                                      \
        if(array_type_is_interleaved(data->node->inArrayType)                         \
           && array_type_is_interleaved(data->node->outArrayType))                    \
        {                                                                             \
            GET_KERNEL_FUNC_SBRC_CB(FWD_KERN_NAME,                                    \
                                    BACK_KERN_NAME,                                   \
                                    PRECISION,                                        \
                                    COL_DIM,                                          \
                                    TRANSPOSE_TYPE,                                   \
                                    KERNEL_BASE_ARGS_OP,                              \
                                    PRECISION* __restrict__,                          \
                                    PRECISION* __restrict__);                         \
            hipLaunchKernelGGL_shim(data->log_func,                                   \
                                    kernel_func,                                      \
                                    dim3(data->gridParam.b_x),                        \
                                    dim3(data->gridParam.wgs_x),                      \
                                    data->gridParam.lds_bytes,                        \
                                    rocfft_stream,                                    \
                                    (PRECISION*)data->node->twiddles,                 \
                                    data->node->length.size(),                        \
                                    kargs_lengths(data->node->devKernArg),            \
                                    kargs_stride_in(data->node->devKernArg),          \
                                    kargs_stride_out(data->node->devKernArg),         \
                                    batch,                                            \
                                    data->node->lds_padding,                          \
                                    data->callbacks.load_cb_fn,                       \
                                    data->callbacks.load_cb_data,                     \
                                    data->callbacks.load_cb_lds_bytes,                \
                                    data->callbacks.store_cb_fn,                      \
                                    data->callbacks.store_cb_data,                    \
                                    (PRECISION*)data->bufIn[0],                       \
                                    (PRECISION*)data->bufOut[0]);                     \
        }                                                                             \
        else if(array_type_is_interleaved(data->node->inArrayType)                    \
                && array_type_is_planar(data->node->outArrayType))                    \
        {                                                                             \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                       \
                                 BACK_KERN_NAME,                                      \
                                 PRECISION,                                           \
                                 COL_DIM,                                             \
                                 TRANSPOSE_TYPE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                 \
                                 PRECISION* __restrict__,                             \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 real_type_t<PRECISION>* __restrict__);               \
            hipLaunchKernelGGL_shim(data->log_func,                                   \
                                    kernel_func,                                      \
                                    dim3(data->gridParam.b_x),                        \
                                    dim3(data->gridParam.wgs_x),                      \
                                    data->gridParam.lds_bytes,                        \
                                    rocfft_stream,                                    \
                                    (PRECISION*)data->node->twiddles,                 \
                                    data->node->length.size(),                        \
                                    kargs_lengths(data->node->devKernArg),            \
                                    kargs_stride_in(data->node->devKernArg),          \
                                    kargs_stride_out(data->node->devKernArg),         \
                                    batch,                                            \
                                    data->node->lds_padding,                          \
                                    data->callbacks.load_cb_fn,                       \
                                    data->callbacks.load_cb_data,                     \
                                    data->callbacks.load_cb_lds_bytes,                \
                                    data->callbacks.store_cb_fn,                      \
                                    data->callbacks.store_cb_data,                    \
                                    (PRECISION*)data->bufIn[0],                       \
                                    (real_type_t<PRECISION>*)data->bufOut[0],         \
                                    (real_type_t<PRECISION>*)data->bufOut[1]);        \
        }                                                                             \
        else if(array_type_is_planar(data->node->inArrayType)                         \
                && array_type_is_interleaved(data->node->outArrayType))               \
        {                                                                             \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                       \
                                 BACK_KERN_NAME,                                      \
                                 PRECISION,                                           \
                                 COL_DIM,                                             \
                                 TRANSPOSE_TYPE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                 \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 PRECISION* __restrict__);                            \
            hipLaunchKernelGGL_shim(data->log_func,                                   \
                                    kernel_func,                                      \
                                    dim3(data->gridParam.b_x),                        \
                                    dim3(data->gridParam.wgs_x),                      \
                                    data->gridParam.lds_bytes,                        \
                                    rocfft_stream,                                    \
                                    (PRECISION*)data->node->twiddles,                 \
                                    data->node->length.size(),                        \
                                    kargs_lengths(data->node->devKernArg),            \
                                    kargs_stride_in(data->node->devKernArg),          \
                                    kargs_stride_out(data->node->devKernArg),         \
                                    batch,                                            \
                                    data->node->lds_padding,                          \
                                    data->callbacks.load_cb_fn,                       \
                                    data->callbacks.load_cb_data,                     \
                                    data->callbacks.load_cb_lds_bytes,                \
                                    data->callbacks.store_cb_fn,                      \
                                    data->callbacks.store_cb_data,                    \
                                    (real_type_t<PRECISION>*)data->bufIn[0],          \
                                    (real_type_t<PRECISION>*)data->bufIn[1],          \
                                    (PRECISION*)data->bufOut[0]);                     \
        }                                                                             \
        else if(array_type_is_planar(data->node->inArrayType)                         \
                && array_type_is_planar(data->node->outArrayType))                    \
        {                                                                             \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                       \
                                 BACK_KERN_NAME,                                      \
                                 PRECISION,                                           \
                                 COL_DIM,                                             \
                                 TRANSPOSE_TYPE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                 \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 real_type_t<PRECISION>* __restrict__,                \
                                 real_type_t<PRECISION>* __restrict__);               \
            hipLaunchKernelGGL_shim(data->log_func,                                   \
                                    kernel_func,                                      \
                                    dim3(data->gridParam.b_x),                        \
                                    dim3(data->gridParam.wgs_x),                      \
                                    data->gridParam.lds_bytes,                        \
                                    rocfft_stream,                                    \
                                    (PRECISION*)data->node->twiddles,                 \
                                    data->node->length.size(),                        \
                                    kargs_lengths(data->node->devKernArg),            \
                                    kargs_stride_in(data->node->devKernArg),          \
                                    kargs_stride_out(data->node->devKernArg),         \
                                    batch,                                            \
                                    data->node->lds_padding,                          \
                                    data->callbacks.load_cb_fn,                       \
                                    data->callbacks.load_cb_data,                     \
                                    data->callbacks.load_cb_lds_bytes,                \
                                    data->callbacks.store_cb_fn,                      \
                                    data->callbacks.store_cb_data,                    \
                                    (real_type_t<PRECISION>*)data->bufIn[0],          \
                                    (real_type_t<PRECISION>*)data->bufIn[1],          \
                                    (real_type_t<PRECISION>*)data->bufOut[0],         \
                                    (real_type_t<PRECISION>*)data->bufOut[1]);        \
        }                                                                             \
    }

#define POWX_LARGE_SBCR_GENERATOR(FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION) \
    ROCFFT_DEVICE_EXPORT void FUNCTION_NAME(const void* data_p, void* back_p)              \
    {                                                                                      \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                               \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                 \
                                                                                           \
        const size_t batch = data->node->batch;                                            \
                                                                                           \
        if(array_type_is_interleaved(data->node->inArrayType)                              \
           && array_type_is_interleaved(data->node->outArrayType))                         \
        {                                                                                  \
            GET_KERNEL_FUNC_SBCR_CB(FWD_KERN_NAME,                                         \
                                    BACK_KERN_NAME,                                        \
                                    PRECISION,                                             \
                                    KERNEL_BASE_ARGS_OP,                                   \
                                    PRECISION* __restrict__,                               \
                                    PRECISION* __restrict__);                              \
            hipLaunchKernelGGL_shim(data->log_func,                                        \
                                    kernel_func,                                           \
                                    dim3(data->gridParam.b_x),                             \
                                    dim3(data->gridParam.wgs_x),                           \
                                    data->gridParam.lds_bytes,                             \
                                    rocfft_stream,                                         \
                                    (PRECISION*)data->node->twiddles,                      \
                                    data->node->length.size(),                             \
                                    kargs_lengths(data->node->devKernArg),                 \
                                    kargs_stride_in(data->node->devKernArg),               \
                                    kargs_stride_out(data->node->devKernArg),              \
                                    batch,                                                 \
                                    data->node->lds_padding,                               \
                                    data->callbacks.load_cb_fn,                            \
                                    data->callbacks.load_cb_data,                          \
                                    data->callbacks.load_cb_lds_bytes,                     \
                                    data->callbacks.store_cb_fn,                           \
                                    data->callbacks.store_cb_data,                         \
                                    (PRECISION*)data->bufIn[0],                            \
                                    (PRECISION*)data->bufOut[0]);                          \
        }                                                                                  \
        else if(array_type_is_interleaved(data->node->inArrayType)                         \
                && array_type_is_planar(data->node->outArrayType))                         \
        {                                                                                  \
            GET_KERNEL_FUNC_SBCR(FWD_KERN_NAME,                                            \
                                 BACK_KERN_NAME,                                           \
                                 PRECISION,                                                \
                                 KERNEL_BASE_ARGS_OP,                                      \
                                 PRECISION* __restrict__,                                  \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 real_type_t<PRECISION>* __restrict__);                    \
            hipLaunchKernelGGL_shim(data->log_func,                                        \
                                    kernel_func,                                           \
                                    dim3(data->gridParam.b_x),                             \
                                    dim3(data->gridParam.wgs_x),                           \
                                    data->gridParam.lds_bytes,                             \
                                    rocfft_stream,                                         \
                                    (PRECISION*)data->node->twiddles,                      \
                                    data->node->length.size(),                             \
                                    kargs_lengths(data->node->devKernArg),                 \
                                    kargs_stride_in(data->node->devKernArg),               \
                                    kargs_stride_out(data->node->devKernArg),              \
                                    batch,                                                 \
                                    data->node->lds_padding,                               \
                                    data->callbacks.load_cb_fn,                            \
                                    data->callbacks.load_cb_data,                          \
                                    data->callbacks.load_cb_lds_bytes,                     \
                                    data->callbacks.store_cb_fn,                           \
                                    data->callbacks.store_cb_data,                         \
                                    (PRECISION*)data->bufIn[0],                            \
                                    (real_type_t<PRECISION>*)data->bufOut[0],              \
                                    (real_type_t<PRECISION>*)data->bufOut[1]);             \
        }                                                                                  \
        else if(array_type_is_planar(data->node->inArrayType)                              \
                && array_type_is_interleaved(data->node->outArrayType))                    \
        {                                                                                  \
            GET_KERNEL_FUNC_SBCR(FWD_KERN_NAME,                                            \
                                 BACK_KERN_NAME,                                           \
                                 PRECISION,                                                \
                                 KERNEL_BASE_ARGS_OP,                                      \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 PRECISION* __restrict__);                                 \
            hipLaunchKernelGGL_shim(data->log_func,                                        \
                                    kernel_func,                                           \
                                    dim3(data->gridParam.b_x),                             \
                                    dim3(data->gridParam.wgs_x),                           \
                                    data->gridParam.lds_bytes,                             \
                                    rocfft_stream,                                         \
                                    (PRECISION*)data->node->twiddles,                      \
                                    data->node->length.size(),                             \
                                    kargs_lengths(data->node->devKernArg),                 \
                                    kargs_stride_in(data->node->devKernArg),               \
                                    kargs_stride_out(data->node->devKernArg),              \
                                    batch,                                                 \
                                    data->node->lds_padding,                               \
                                    data->callbacks.load_cb_fn,                            \
                                    data->callbacks.load_cb_data,                          \
                                    data->callbacks.load_cb_lds_bytes,                     \
                                    data->callbacks.store_cb_fn,                           \
                                    data->callbacks.store_cb_data,                         \
                                    (real_type_t<PRECISION>*)data->bufIn[0],               \
                                    (real_type_t<PRECISION>*)data->bufIn[1],               \
                                    (PRECISION*)data->bufOut[0]);                          \
        }                                                                                  \
        else if(array_type_is_planar(data->node->inArrayType)                              \
                && array_type_is_planar(data->node->outArrayType))                         \
        {                                                                                  \
            GET_KERNEL_FUNC_SBCR(FWD_KERN_NAME,                                            \
                                 BACK_KERN_NAME,                                           \
                                 PRECISION,                                                \
                                 KERNEL_BASE_ARGS_OP,                                      \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 real_type_t<PRECISION>* __restrict__,                     \
                                 real_type_t<PRECISION>* __restrict__);                    \
            hipLaunchKernelGGL_shim(data->log_func,                                        \
                                    kernel_func,                                           \
                                    dim3(data->gridParam.b_x),                             \
                                    dim3(data->gridParam.wgs_x),                           \
                                    data->gridParam.lds_bytes,                             \
                                    rocfft_stream,                                         \
                                    (PRECISION*)data->node->twiddles,                      \
                                    data->node->length.size(),                             \
                                    kargs_lengths(data->node->devKernArg),                 \
                                    kargs_stride_in(data->node->devKernArg),               \
                                    kargs_stride_out(data->node->devKernArg),              \
                                    batch,                                                 \
                                    data->node->lds_padding,                               \
                                    data->callbacks.load_cb_fn,                            \
                                    data->callbacks.load_cb_data,                          \
                                    data->callbacks.load_cb_lds_bytes,                     \
                                    data->callbacks.store_cb_fn,                           \
                                    data->callbacks.store_cb_data,                         \
                                    (real_type_t<PRECISION>*)data->bufIn[0],               \
                                    (real_type_t<PRECISION>*)data->bufIn[1],               \
                                    (real_type_t<PRECISION>*)data->bufOut[0],              \
                                    (real_type_t<PRECISION>*)data->bufOut[1]);             \
        }                                                                                  \
    }

#endif // KERNEL_LAUNCH_SINGLE
