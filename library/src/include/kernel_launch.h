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

#ifndef KERNEL_LAUNCH_SINGLE
#define KERNEL_LAUNCH_SINGLE

#define FN_PRFX(X) rocfft_internal_##X
#ifndef __clang__
#include "error.h"
#endif
#include "../../../shared/array_predicate.h"
#include "../device/kernels/callback.h"
#include "kargs.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include "tree_node.h"
#include <iostream>

// FIXME: documentation
struct DeviceCallIn
{
    TreeNode* node;
    void*     bufIn[2];
    void*     bufOut[2];

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

void rocfft_internal_mul(const void* data_p, void* back_p);
void rocfft_internal_chirp(const void* data_p, void* back_p);
void rocfft_internal_transpose_var2(const void* data_p, void* back_p);

/*
   data->node->devKernArg : points to the internal length device pointer
   data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH : points to the intenal in
   stride device pointer
   data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH : points to the internal out
   stride device pointer, only used in outof place kernels
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

#define GET_KERNEL_FUNC_CBTYPE(FWD, BACK, PRECISION, EBTYPE, CBTYPE)   \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)  \
    {                                                                  \
        if(data->node->direction == -1)                                \
            kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE>;     \
        else                                                           \
            kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE>;    \
    }                                                                  \
    else                                                               \
    {                                                                  \
        if(data->node->direction == -1)                                \
            kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE>;  \
        else                                                           \
            kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE>; \
    }

#define GET_KERNEL_FUNC(FWD, BACK, PRECISION, EBTYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr; \
    GET_KERNEL_FUNC_CBTYPE(FWD, BACK, PRECISION, EBTYPE, CallbackType::NONE)

#define GET_KERNEL_FUNC_CB(FWD, BACK, PRECISION, EBTYPE, BASE_ARGS, ...)         \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;            \
    if(data->get_callback_type() == CallbackType::NONE)                          \
        GET_KERNEL_FUNC_CBTYPE(FWD, BACK, PRECISION, EBTYPE, CallbackType::NONE) \
    else                                                                         \
        GET_KERNEL_FUNC_CBTYPE(FWD, BACK, PRECISION, EBTYPE, CallbackType::USER_LOAD_STORE)

// SBCC adds large twiddles
#define KERNEL_BASE_ARGS_IP_SBCC(PRECISION) \
    const PRECISION* __restrict__, KERNEL_BASE_ARGS_IP(PRECISION)
#define KERNEL_BASE_ARGS_OP_SBCC(PRECISION) \
    const PRECISION* __restrict__, KERNEL_BASE_ARGS_OP(PRECISION)

#define GET_KERNEL_FUNC_CBTYPE_SBCC(FWD, BACK, PRECISION, EBTYPE, CBTYPE)               \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                   \
    {                                                                                   \
        if(data->node->direction == -1)                                                 \
        {                                                                               \
            if(data->node->large1D)                                                     \
            {                                                                           \
                if(data->node->largeTwdBase == 4)                                       \
                    kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 4>;     \
                else if(data->node->largeTwdBase == 5)                                  \
                    kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 5>;     \
                else if(data->node->largeTwdBase == 6)                                  \
                    kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 6>;     \
                else                                                                    \
                    kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true>;        \
            }                                                                           \
            else                                                                        \
                kernel_func = FWD<PRECISION, SB_UNIT, EBTYPE, CBTYPE, false>;           \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            if(data->node->large1D)                                                     \
            {                                                                           \
                if(data->node->largeTwdBase == 4)                                       \
                    kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 4>;    \
                else if(data->node->largeTwdBase == 5)                                  \
                    kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 5>;    \
                else if(data->node->largeTwdBase == 6)                                  \
                    kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true, 6>;    \
                else                                                                    \
                    kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE, true>;       \
            }                                                                           \
            else                                                                        \
                kernel_func = BACK<PRECISION, SB_UNIT, EBTYPE, CBTYPE, false>;          \
        }                                                                               \
    }                                                                                   \
    else                                                                                \
    {                                                                                   \
        if(data->node->direction == -1)                                                 \
        {                                                                               \
            if(data->node->large1D)                                                     \
            {                                                                           \
                if(data->node->largeTwdBase == 4)                                       \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 4>;  \
                else if(data->node->largeTwdBase == 5)                                  \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 5>;  \
                else if(data->node->largeTwdBase == 6)                                  \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 6>;  \
                else                                                                    \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true>;     \
            }                                                                           \
            else                                                                        \
                kernel_func = FWD<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, false>;        \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            if(data->node->large1D)                                                     \
            {                                                                           \
                if(data->node->largeTwdBase == 4)                                       \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 4>; \
                else if(data->node->largeTwdBase == 5)                                  \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 5>; \
                else if(data->node->largeTwdBase == 6)                                  \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true, 6>; \
                else                                                                    \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, true>;    \
            }                                                                           \
            else                                                                        \
                kernel_func = BACK<PRECISION, SB_NONUNIT, EBTYPE, CBTYPE, false>;       \
        }                                                                               \
    }
#define GET_KERNEL_FUNC_SBCC_CB(FWD, BACK, PRECISION, EBTYPE, BASE_ARGS, ...)         \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                 \
    if(data->get_callback_type() == CallbackType::NONE)                               \
        GET_KERNEL_FUNC_CBTYPE_SBCC(FWD, BACK, PRECISION, EBTYPE, CallbackType::NONE) \
    else                                                                              \
        GET_KERNEL_FUNC_CBTYPE_SBCC(FWD, BACK, PRECISION, EBTYPE, CallbackType::USER_LOAD_STORE)
#define GET_KERNEL_FUNC_SBCC(FWD, BACK, PRECISION, EBTYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;      \
    GET_KERNEL_FUNC_CBTYPE_SBCC(FWD, BACK, PRECISION, EBTYPE, CallbackType::NONE)

// SBRC has COL_DIM, TRANSPOSE_TYPE template args and is always out-of-place
#define GET_KERNEL_FUNC_CBTYPE_SBRC(FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CBTYPE) \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                              \
    {                                                                                              \
        if(data->node->direction == -1)                                                            \
            kernel_func = FWD<PRECISION, SB_UNIT, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CBTYPE>;        \
        else                                                                                       \
            kernel_func = BACK<PRECISION, SB_UNIT, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CBTYPE>;       \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(data->node->direction == -1)                                                            \
            kernel_func = FWD<PRECISION, SB_NONUNIT, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CBTYPE>;     \
        else                                                                                       \
            kernel_func = BACK<PRECISION, SB_NONUNIT, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CBTYPE>;    \
    }
#define GET_KERNEL_FUNC_SBRC_CB(                                                       \
    FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, BASE_ARGS, ...)             \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                  \
    if(data->get_callback_type() == CallbackType::NONE)                                \
        GET_KERNEL_FUNC_CBTYPE_SBRC(                                                   \
            FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CallbackType::NONE) \
    else                                                                               \
        GET_KERNEL_FUNC_CBTYPE_SBRC(                                                   \
            FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CallbackType::USER_LOAD_STORE)
#define GET_KERNEL_FUNC_SBRC(                                              \
    FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;      \
    GET_KERNEL_FUNC_CBTYPE_SBRC(                                           \
        FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, EBTYPE, CallbackType::NONE)

#define POWX_SMALL_GENERATOR(FUNCTION_NAME,                                                   \
                             IP_FWD_KERN_NAME,                                                \
                             IP_BACK_KERN_NAME,                                               \
                             OP_FWD_KERN_NAME,                                                \
                             OP_BACK_KERN_NAME,                                               \
                             PRECISION)                                                       \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                      \
    {                                                                                         \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                                  \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                    \
                                                                                              \
        if(data->node->placement == rocfft_placement_inplace)                                 \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_CB(IP_FWD_KERN_NAME,                                          \
                                   IP_BACK_KERN_NAME,                                         \
                                   PRECISION,                                                 \
                                   EmbeddedType::NONE,                                        \
                                   KERNEL_BASE_ARGS_IP,                                       \
                                   PRECISION* __restrict__);                                  \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0]);                               \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(IP_FWD_KERN_NAME,                                             \
                                IP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                EmbeddedType::NONE,                                           \
                                KERNEL_BASE_ARGS_IP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1]);                  \
            }                                                                                 \
        }                                                                                     \
        else /* out of place */                                                               \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_CB(OP_FWD_KERN_NAME,                                          \
                                   OP_BACK_KERN_NAME,                                         \
                                   PRECISION,                                                 \
                                   EmbeddedType::NONE,                                        \
                                   KERNEL_BASE_ARGS_OP,                                       \
                                   PRECISION* __restrict__,                                   \
                                   PRECISION* __restrict__);                                  \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_interleaved(data->node->inArrayType)                        \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                EmbeddedType::NONE,                                           \
                                KERNEL_BASE_ARGS_OP,                                          \
                                PRECISION* __restrict__,                                      \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_interleaved(data->node->outArrayType))                   \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                EmbeddedType::NONE,                                           \
                                KERNEL_BASE_ARGS_OP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                PRECISION* __restrict__);                                     \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                EmbeddedType::NONE,                                           \
                                KERNEL_BASE_ARGS_OP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
        }                                                                                     \
    }

#define POWX_LARGE_SBCC_GENERATOR(FUNCTION_NAME,                                              \
                                  IP_FWD_KERN_NAME,                                           \
                                  IP_BACK_KERN_NAME,                                          \
                                  OP_FWD_KERN_NAME,                                           \
                                  OP_BACK_KERN_NAME,                                          \
                                  PRECISION)                                                  \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                      \
    {                                                                                         \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                                  \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                    \
                                                                                              \
        const size_t batch = data->node->batch;                                               \
                                                                                              \
        if(data->node->placement == rocfft_placement_inplace)                                 \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC_CB(IP_FWD_KERN_NAME,                                     \
                                        IP_BACK_KERN_NAME,                                    \
                                        PRECISION,                                            \
                                        EmbeddedType::NONE,                                   \
                                        KERNEL_BASE_ARGS_IP_SBCC,                             \
                                        PRECISION* __restrict__);                             \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0]);                               \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(IP_FWD_KERN_NAME,                                        \
                                     IP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     EmbeddedType::NONE,                                      \
                                     KERNEL_BASE_ARGS_IP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1]);                  \
            }                                                                                 \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC_CB(OP_FWD_KERN_NAME,                                     \
                                        OP_BACK_KERN_NAME,                                    \
                                        PRECISION,                                            \
                                        EmbeddedType::NONE,                                   \
                                        KERNEL_BASE_ARGS_OP_SBCC,                             \
                                        PRECISION* __restrict__,                              \
                                        PRECISION* __restrict__);                             \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_interleaved(data->node->inArrayType)                        \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     EmbeddedType::NONE,                                      \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     PRECISION* __restrict__,                                 \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_interleaved(data->node->outArrayType))                   \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     EmbeddedType::NONE,                                      \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     PRECISION* __restrict__);                                \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     EmbeddedType::NONE,                                      \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   data->gridParam.lds_bytes,                                 \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   data->node->lds_padding,                                   \
                                   data->callbacks.load_cb_fn,                                \
                                   data->callbacks.load_cb_data,                              \
                                   data->callbacks.load_cb_lds_bytes,                         \
                                   data->callbacks.store_cb_fn,                               \
                                   data->callbacks.store_cb_data,                             \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
        }                                                                                     \
    }

#define POWX_LARGE_SBRC_GENERATOR(                                                        \
    FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION, COL_DIM, TRANSPOSE_TYPE)     \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                  \
    {                                                                                     \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                              \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                \
                                                                                          \
        const size_t batch = data->node->batch;                                           \
                                                                                          \
        if(array_type_is_interleaved(data->node->inArrayType)                             \
           && array_type_is_interleaved(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC_CB(FWD_KERN_NAME,                                        \
                                    BACK_KERN_NAME,                                       \
                                    PRECISION,                                            \
                                    COL_DIM,                                              \
                                    TRANSPOSE_TYPE,                                       \
                                    EmbeddedType::NONE,                                   \
                                    KERNEL_BASE_ARGS_OP,                                  \
                                    PRECISION* __restrict__,                              \
                                    PRECISION* __restrict__);                             \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               data->gridParam.lds_bytes,                                 \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               data->node->lds_padding,                                   \
                               data->callbacks.load_cb_fn,                                \
                               data->callbacks.load_cb_data,                              \
                               data->callbacks.load_cb_lds_bytes,                         \
                               data->callbacks.store_cb_fn,                               \
                               data->callbacks.store_cb_data,                             \
                               (PRECISION*)data->bufIn[0],                                \
                               (PRECISION*)data->bufOut[0]);                              \
        }                                                                                 \
        else if(array_type_is_interleaved(data->node->inArrayType)                        \
                && array_type_is_planar(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 EmbeddedType::NONE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 PRECISION* __restrict__,                                 \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__);                   \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               data->gridParam.lds_bytes,                                 \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               data->node->lds_padding,                                   \
                               data->callbacks.load_cb_fn,                                \
                               data->callbacks.load_cb_data,                              \
                               data->callbacks.load_cb_lds_bytes,                         \
                               data->callbacks.store_cb_fn,                               \
                               data->callbacks.store_cb_data,                             \
                               (PRECISION*)data->bufIn[0],                                \
                               (real_type_t<PRECISION>*)data->bufOut[0],                  \
                               (real_type_t<PRECISION>*)data->bufOut[1]);                 \
        }                                                                                 \
        else if(array_type_is_planar(data->node->inArrayType)                             \
                && array_type_is_interleaved(data->node->outArrayType))                   \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 EmbeddedType::NONE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 PRECISION* __restrict__);                                \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               data->gridParam.lds_bytes,                                 \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               data->node->lds_padding,                                   \
                               data->callbacks.load_cb_fn,                                \
                               data->callbacks.load_cb_data,                              \
                               data->callbacks.load_cb_lds_bytes,                         \
                               data->callbacks.store_cb_fn,                               \
                               data->callbacks.store_cb_data,                             \
                               (real_type_t<PRECISION>*)data->bufIn[0],                   \
                               (real_type_t<PRECISION>*)data->bufIn[1],                   \
                               (PRECISION*)data->bufOut[0]);                              \
        }                                                                                 \
        else if(array_type_is_planar(data->node->inArrayType)                             \
                && array_type_is_planar(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 EmbeddedType::NONE,                                      \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__);                   \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               data->gridParam.lds_bytes,                                 \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               data->node->lds_padding,                                   \
                               data->callbacks.load_cb_fn,                                \
                               data->callbacks.load_cb_data,                              \
                               data->callbacks.load_cb_lds_bytes,                         \
                               data->callbacks.store_cb_fn,                               \
                               data->callbacks.store_cb_data,                             \
                               (real_type_t<PRECISION>*)data->bufIn[0],                   \
                               (real_type_t<PRECISION>*)data->bufIn[1],                   \
                               (real_type_t<PRECISION>*)data->bufOut[0],                  \
                               (real_type_t<PRECISION>*)data->bufOut[1]);                 \
        }                                                                                 \
    }

#endif // KERNEL_LAUNCH_SINGLE
