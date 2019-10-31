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
#include "kargs.h"
#include "kernel_launch_generator.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include "tree_node.h"
#include <iostream>

struct DeviceCallIn
{
    TreeNode* node;
    void*     bufIn[2]; // TODO, bufIn[1] is never used
    void*     bufOut[2]; // TODO, bufOut[1] is never used

    hipStream_t rocfft_stream;
    GridParam   gridParam;
};

struct DeviceCallOut
{
    int err;
};

extern "C" {

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
}

/*
   data->node->devKernArg : points to the internal length device pointer
   data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH : points to the intenal in
   stride device pointer
   data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH : points to the internal out
   stride device pointer, only used in outof place kernels
*/

#define POWX_SMALL_GENERATOR(FUNCTION_NAME,                                                       \
                             IP_FWD_KERN_NAME,                                                    \
                             IP_BACK_KERN_NAME,                                                   \
                             OP_FWD_KERN_NAME,                                                    \
                             OP_BACK_KERN_NAME,                                                   \
                             PRECISION)                                                           \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                          \
    {                                                                                             \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                                      \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                        \
        if(data->node->placement == rocfft_placement_inplace)                                     \
        {                                                                                         \
            if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                     \
            {                                                                                     \
                if(data->node->direction == -1)                                                   \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(IP_FWD_KERN_NAME<PRECISION, SB_UNIT>),     \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0]);                               \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(IP_BACK_KERN_NAME<PRECISION, SB_UNIT>),    \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0]);                               \
                }                                                                                 \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                if(data->node->direction == -1)                                                   \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(IP_FWD_KERN_NAME<PRECISION, SB_NONUNIT>),  \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0]);                               \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(IP_BACK_KERN_NAME<PRECISION, SB_NONUNIT>), \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0]);                               \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)                     \
            {                                                                                     \
                if(data->node->direction == -1)                                                   \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(OP_FWD_KERN_NAME<PRECISION, SB_UNIT>),     \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0],                                \
                                       (PRECISION*)data->bufOut[0]);                              \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(OP_BACK_KERN_NAME<PRECISION, SB_UNIT>),    \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0],                                \
                                       (PRECISION*)data->bufOut[0]);                              \
                }                                                                                 \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                if(data->node->direction == -1)                                                   \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(OP_FWD_KERN_NAME<PRECISION, SB_NONUNIT>),  \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0],                                \
                                       (PRECISION*)data->bufOut[0]);                              \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(OP_BACK_KERN_NAME<PRECISION, SB_NONUNIT>), \
                                       dim3(data->gridParam.b_x),                                 \
                                       dim3(data->gridParam.tpb_x),                               \
                                       0,                                                         \
                                       rocfft_stream,                                             \
                                       (PRECISION*)data->node->twiddles,                          \
                                       data->node->length.size(),                                 \
                                       data->node->devKernArg,                                    \
                                       data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,        \
                                       data->node->batch,                                         \
                                       (PRECISION*)data->bufIn[0],                                \
                                       (PRECISION*)data->bufOut[0]);                              \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

#define POWX_LARGE_SBCC_GENERATOR(FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION) \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                   \
    {                                                                                      \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                               \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                 \
        if(data->node->direction == -1)                                                    \
        {                                                                                  \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(FWD_KERN_NAME<PRECISION, SB_UNIT>),         \
                               dim3(data->gridParam.b_x),                                  \
                               dim3(data->gridParam.tpb_x),                                \
                               0,                                                          \
                               rocfft_stream,                                              \
                               (PRECISION*)data->node->twiddles,                           \
                               (PRECISION*)data->node->twiddles_large,                     \
                               data->node->length.size(),                                  \
                               data->node->devKernArg,                                     \
                               data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->batch,                                          \
                               (PRECISION*)data->bufIn[0],                                 \
                               (PRECISION*)data->bufOut[0]);                               \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(BACK_KERN_NAME<PRECISION, SB_UNIT>),        \
                               dim3(data->gridParam.b_x),                                  \
                               dim3(data->gridParam.tpb_x),                                \
                               0,                                                          \
                               rocfft_stream,                                              \
                               (PRECISION*)data->node->twiddles,                           \
                               (PRECISION*)data->node->twiddles_large,                     \
                               data->node->length.size(),                                  \
                               data->node->devKernArg,                                     \
                               data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->batch,                                          \
                               (PRECISION*)data->bufIn[0],                                 \
                               (PRECISION*)data->bufOut[0]);                               \
        }                                                                                  \
    }

#define POWX_LARGE_SBRC_GENERATOR(FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION) \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                   \
    {                                                                                      \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                               \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                 \
        if(data->node->direction == -1)                                                    \
        {                                                                                  \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(FWD_KERN_NAME<PRECISION, SB_UNIT>),         \
                               dim3(data->gridParam.b_x),                                  \
                               dim3(data->gridParam.tpb_x),                                \
                               0,                                                          \
                               rocfft_stream,                                              \
                               (PRECISION*)data->node->twiddles,                           \
                               data->node->length.size(),                                  \
                               data->node->devKernArg,                                     \
                               data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->batch,                                          \
                               (PRECISION*)data->bufIn[0],                                 \
                               (PRECISION*)data->bufOut[0]);                               \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(BACK_KERN_NAME<PRECISION, SB_UNIT>),        \
                               dim3(data->gridParam.b_x),                                  \
                               dim3(data->gridParam.tpb_x),                                \
                               0,                                                          \
                               rocfft_stream,                                              \
                               (PRECISION*)data->node->twiddles,                           \
                               data->node->length.size(),                                  \
                               data->node->devKernArg,                                     \
                               data->node->devKernArg + 1 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->devKernArg + 2 * KERN_ARGS_ARRAY_WIDTH,         \
                               data->node->batch,                                          \
                               (PRECISION*)data->bufIn[0],                                 \
                               (PRECISION*)data->bufOut[0]);                               \
        }                                                                                  \
    }

#endif // KERNEL_LAUNCH_SINGLE
