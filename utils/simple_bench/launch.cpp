/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <assert.h>
#include <iostream>
#include <vector>

#include "kernel_launch.h"

#include <iostream>

#include "./kernels/pow2_ip_entry.h"
#include "./kernels/pow2_large_entry.h"
#include "./kernels/pow2_op_entry.h"

// precision, placement, iL, oL, scheme, dim, length **, iStrides **, oStrides
// **
void device_call_template(void*, void*);

/*
#define POW2_SMALL_IP_C(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
        DeviceCallIn *data = (DeviceCallIn *)data_p;\
        DeviceCallOut *back = (DeviceCallOut *)back_p;\
        if(data->direction == -1) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME( KERN_NAME<float2,-1> ),
dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2
*)data->twiddles, (float2 *)data->bufIn[0]);\
        else \
        hipLaunchKernelGGL(HIP_KERNEL_NAME( KERN_NAME<float2,1> ),
dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2
*)data->twiddles, (float2 *)data->bufIn[0]);\
}

POW2_SMALL_IP_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4096),fft_4096_ip_d1_pk)
POW2_SMALL_IP_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2048),fft_2048_ip_d1_pk)
*/

#define TRANSPOSE_CALL(NUM_Y, DRN, TWL, TTD)                                   \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(transpose_var1<float2, DRN, TWL, TTD>), \
                       dim3(data->gridParam.b_x, data->gridParam.b_y),         \
                       dim3(data->gridParam.tpb_x, data->gridParam.tpb_x),     \
                       0,                                                      \
                       0,                                                      \
                       (float2*)data->twiddles_large,                          \
                       (float2*)data->bufIn[0],                                \
                       (float2*)data->bufOut[0],                               \
                       NUM_Y,                                                  \
                       data->inStride[1],                                      \
                       data->outStride[1],                                     \
                       data->iDist,                                            \
                       data->oDist)

void FN_PRFX(transpose_var1_sp)(void* data_p, void* back_p)
{
    DeviceCallIn*  data = (DeviceCallIn*)data_p;
    DeviceCallOut* back = (DeviceCallOut*)back_p;

    if(data->transTileDir == TTD_IP_HOR)
    {
        if(data->large1D == 0)
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[1] / 64), -1, 0, TTD_IP_HOR);
            else
                TRANSPOSE_CALL((data->length[1] / 64), 1, 0, TTD_IP_HOR);
        }
        else if(data->large1D <= 16777216)
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[1] / 64), -1, 3, TTD_IP_HOR);
            else
                TRANSPOSE_CALL((data->length[1] / 64), 1, 3, TTD_IP_HOR);
        }
        else
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[1] / 64), -1, 4, TTD_IP_HOR);
            else
                TRANSPOSE_CALL((data->length[1] / 64), 1, 4, TTD_IP_HOR);
        }
    }
    else
    {
        if(data->large1D == 0)
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[0] / 64), -1, 0, TTD_IP_VER);
            else
                TRANSPOSE_CALL((data->length[0] / 64), 1, 0, TTD_IP_VER);
        }
        else if(data->large1D <= 16777216)
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[0] / 64), -1, 3, TTD_IP_VER);
            else
                TRANSPOSE_CALL((data->length[0] / 64), 1, 3, TTD_IP_VER);
        }
        else
        {
            if(data->direction == -1)
                TRANSPOSE_CALL((data->length[0] / 64), -1, 4, TTD_IP_VER);
            else
                TRANSPOSE_CALL((data->length[0] / 64), 1, 4, TTD_IP_VER);
        }
    }
}

#define POW2_SMALL_IP_2_C(FUNCTION_NAME, KERN_NAME)                    \
    void FUNCTION_NAME(void* data_p, void* back_p)                     \
    {                                                                  \
        DeviceCallIn*  data = (DeviceCallIn*)data_p;                   \
        DeviceCallOut* back = (DeviceCallOut*)back_p;                  \
        if(data->direction == -1)                                      \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(KERN_NAME<float2, -1>), \
                               dim3(data->gridParam.b_x),              \
                               dim3(data->gridParam.tpb_x),            \
                               0,                                      \
                               0,                                      \
                               (float2*)data->twiddles,                \
                               (float2*)data->bufIn[0],                \
                               data->length[1],                        \
                               data->inStride[1],                      \
                               data->iDist);                           \
        else                                                           \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(KERN_NAME<float2, 1>),  \
                               dim3(data->gridParam.b_x),              \
                               dim3(data->gridParam.tpb_x),            \
                               0,                                      \
                               0,                                      \
                               (float2*)data->twiddles,                \
                               (float2*)data->bufIn[0],                \
                               data->length[1],                        \
                               data->inStride[1],                      \
                               data->iDist);                           \
    }

#define POW2_SMALL_OP_2_C(FUNCTION_NAME, KERN_NAME)                    \
    void FUNCTION_NAME(void* data_p, void* back_p)                     \
    {                                                                  \
        DeviceCallIn*  data = (DeviceCallIn*)data_p;                   \
        DeviceCallOut* back = (DeviceCallOut*)back_p;                  \
        if(data->direction == -1)                                      \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(KERN_NAME<float2, -1>), \
                               dim3(data->gridParam.b_x),              \
                               dim3(data->gridParam.tpb_x),            \
                               0,                                      \
                               0,                                      \
                               (float2*)data->twiddles,                \
                               (float2*)data->bufIn[0],                \
                               (float2*)data->bufOut[0],               \
                               data->length[1],                        \
                               data->inStride[1],                      \
                               data->outStride[1],                     \
                               data->iDist,                            \
                               data->oDist);                           \
        else                                                           \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(KERN_NAME<float2, 1>),  \
                               dim3(data->gridParam.b_x),              \
                               dim3(data->gridParam.tpb_x),            \
                               0,                                      \
                               0,                                      \
                               (float2*)data->twiddles,                \
                               (float2*)data->bufIn[0],                \
                               (float2*)data->bufOut[0],               \
                               data->length[1],                        \
                               data->inStride[1],                      \
                               data->outStride[1],                     \
                               data->iDist,                            \
                               data->oDist);                           \
    }

POW2_SMALL_IP_2_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_4096), fft_4096_ip_d2_s1)
POW2_SMALL_IP_2_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_2048), fft_2048_ip_d2_s1)

POW2_SMALL_OP_2_C(FN_PRFX(dfn_sp_op_ci_ci_stoc_2_4096), fft_4096_op_d2_s1)
POW2_SMALL_OP_2_C(FN_PRFX(dfn_sp_op_ci_ci_stoc_2_2048), fft_2048_op_d2_s1)
