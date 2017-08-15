
#include <iostream>
#include "kernel_launch.h"
#include "./kernels/common.h"
#include "rocfft_hip.h"
#include "./kernels/transpose.h"

void rocfft_internal_transpose_var2(void *data_p, void *back_p)
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

/*
    if(data->node->transTileDir == TTD_IP_HOR)
    {
        if(data->node->large1D == 0)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 0, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 0, TTD_IP_HOR);
        }
        else if(data->node->large1D <= 16777216)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 3, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 3, TTD_IP_HOR);
        }
        else
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[1]/64), -1, 4, TTD_IP_HOR);
            else
            TRANSPOSE_CALL((data->node->length[1]/64),  1, 4, TTD_IP_HOR);
        }
    }
    else
    {
        if(data->node->large1D == 0)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 0, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 0, TTD_IP_VER);
        }
        else if(data->node->large1D <= 16777216)
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 3, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 3, TTD_IP_VER);
        }
        else
        {
            if(data->node->direction == -1)
            TRANSPOSE_CALL((data->node->length[0]/64), -1, 4, TTD_IP_VER);
            else
            TRANSPOSE_CALL((data->node->length[0]/64),  1, 4, TTD_IP_VER);
        }
    }
*/

#define     trans_block_size_16     16
#define     trans_block_size_8       8
#define     trans_block_size_1       1
#define     trans_micro_tile_size_1  1
#define     trans_micro_tile_size_2  2
#define     trans_micro_tile_size_4  4

    int input_row_size = data->node->length[1];
    int input_col_size = data->node->length[0];
    int batch_size = data->node->batch;
    for(size_t j=2; j<data->node->length.size(); j++) batch_size *= data->node->length[j];
    
    if(data->node->precision == rocfft_precision_single)
    {
        if(input_row_size  % (trans_block_size_16*trans_micro_tile_size_4) == 0 && input_col_size % (trans_block_size_16*trans_micro_tile_size_4) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,trans_micro_tile_size_4,trans_micro_tile_size_4,trans_block_size_16,trans_block_size_16>),
                    dim3(input_col_size/trans_micro_tile_size_4/trans_block_size_16 * batch_size, input_row_size/trans_micro_tile_size_4/trans_block_size_16),
                    dim3(trans_block_size_16, trans_block_size_16, 1), 0, 0, (float2*)data->bufIn[0], (float2*)data->bufOut[0], 
                    (float2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);
        }
        else if(input_row_size % (trans_block_size_16*trans_micro_tile_size_2) == 0 && input_col_size % (trans_block_size_16*trans_micro_tile_size_2) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,trans_micro_tile_size_2,trans_micro_tile_size_2,trans_block_size_16,trans_block_size_16>),
                    dim3(input_col_size/trans_micro_tile_size_2/trans_block_size_16 * batch_size, input_row_size/trans_micro_tile_size_2/trans_block_size_16),
                    dim3(trans_block_size_16, trans_block_size_16, 1), 0, 0, (float2*)data->bufIn[0], (float2*)data->bufOut[0],
                    (float2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);
        }
        else if(input_row_size % (trans_block_size_1*trans_micro_tile_size_1) == 0 && input_col_size % (trans_block_size_1*trans_micro_tile_size_1) == 0)
        {
            // the kernel should be able to work on any size with no guarantee of performance 
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,trans_micro_tile_size_1,trans_micro_tile_size_1,trans_block_size_1,trans_block_size_1>),
                    dim3(input_col_size/trans_micro_tile_size_1/trans_block_size_1 * batch_size, input_row_size/trans_micro_tile_size_1/trans_block_size_1),
                    dim3(trans_block_size_1, trans_block_size_1, 1), 0, 0, (float2*)data->bufIn[0], (float2*)data->bufOut[0],
                    (float2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);
        }
        else
             printf("Missing tranpose control flow");
    }
    else if(data->node->precision == rocfft_precision_double)
    {
        if(input_row_size % (trans_block_size_16*trans_micro_tile_size_2) == 0 && input_col_size % (trans_block_size_16*trans_micro_tile_size_2) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double2,trans_micro_tile_size_2,trans_micro_tile_size_2,trans_block_size_16,trans_block_size_16>),
                    dim3(input_col_size/trans_micro_tile_size_2/trans_block_size_16 * batch_size, input_row_size/trans_micro_tile_size_2/trans_block_size_16),
                    dim3(trans_block_size_16, trans_block_size_16, 1), 0, 0, (double2*)data->bufIn[0], (double2*)data->bufOut[0],
                    (double2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);
        }
        else if(input_row_size % (trans_block_size_1*trans_micro_tile_size_1) == 0 && input_col_size % (trans_block_size_1*trans_micro_tile_size_1) == 0)
        {
            // the kernel should be able to work on any size with no guarantee of performance
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double2,trans_micro_tile_size_1,trans_micro_tile_size_1,trans_block_size_1,trans_block_size_1>),
                    dim3(input_col_size/trans_micro_tile_size_1/trans_block_size_1 * batch_size, input_row_size/trans_micro_tile_size_1/trans_block_size_1),
                    dim3(trans_block_size_1, trans_block_size_1, 1), 0, 0, (double2*)data->bufIn[0], (double2*)data->bufOut[0],
                    (double2 *)data->node->twiddles_large, data->node->batch, data->node->length.size(),
                    data->node->devKernArg, data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH, data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH);
        }
        else
             printf("Missing tranpose control flow");

    }

}


