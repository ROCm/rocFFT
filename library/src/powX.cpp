/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <vector>
#include <assert.h>
#include <iostream>
#include <unordered_map>
#include <atomic>

#include "rocfft.h"
#include "plan.h"
#include "repo.h"
#include "transform.h"
#include "radix_table.h"
#include "kernel_launch.h"
#include "function_pool.h"
#include "ref_cpu.h"
#include "real2complex.h"

#ifdef TMP_DEBUG
#include "rocfft_hip.h"
#endif

std::atomic<bool> fn_checked(false);

/* this function is called during creation of plan : enqueue the HIP kernels by function pointers*/
void PlanPowX(ExecPlan &execPlan)
{
    for(size_t i=0; i<execPlan.execSeq.size(); i++)
    {
        if(    (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) ||
            (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) ||
            (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) )
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->precision);
        }

        if(execPlan.execSeq[i]->large1D != 0)
        {
            execPlan.execSeq[i]->twiddles_large = twiddles_create(execPlan.execSeq[i]->large1D, execPlan.execSeq[i]->precision);
        }
    }

    //copy host buffer to device buffer
    for(size_t i=0; i<execPlan.execSeq.size(); i++)
    {
        execPlan.execSeq[i]->devKernArg = kargs_create(execPlan.execSeq[i]->length,
                                                        execPlan.execSeq[i]->inStride, execPlan.execSeq[i]->outStride,
                                                        execPlan.execSeq[i]->iDist, execPlan.execSeq[i]->oDist);
    }

    if(!fn_checked)
    {
        fn_checked = true;
        function_pool::verify_no_null_functions();
    }

    if(execPlan.execSeq[0]->precision == rocfft_precision_single)
    {

            for(size_t i=0; i<execPlan.execSeq.size(); i++)
            {
                DevFnCall ptr = nullptr;
                GridParam gp;
                size_t bwd,wgs,lds;                        

                if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
                {
                    //assert(execPlan.execSeq[i]->length[0] <= 4096);
                    size_t workGroupSize;
                    size_t numTransforms;
                    GetWGSAndNT(execPlan.execSeq[i]->length[0], workGroupSize, numTransforms);//get working group size and number of transforms

                    ptr = function_pool::get_function_single(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM));

                    size_t batch = execPlan.execSeq[i]->batch;
                    for(size_t j=1; j<execPlan.execSeq[i]->length.size(); j++) batch *= execPlan.execSeq[i]->length[j];
                    gp.b_x = (batch%numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
                    gp.tpb_x = workGroupSize;
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                { 
                    ptr = function_pool::get_function_single(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC)) ;

                    GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
                    gp.b_x =  (execPlan.execSeq[i]->length[1])/bwd * execPlan.execSeq[i]->batch;

                    if (execPlan.execSeq[i]->length.size() == 3)
                    {
                        gp.b_x *= execPlan.execSeq[i]->length[2];
                    }
                    gp.tpb_x = wgs;
                }        
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                {
                    ptr = function_pool::get_function_single(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC)) ;

                    GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
                    gp.b_x =  (execPlan.execSeq[i]->length[1])/bwd * execPlan.execSeq[i]->batch;
                    if (execPlan.execSeq[i]->length.size() == 3)
                    {
                        gp.b_x *= execPlan.execSeq[i]->length[2];
                    }
                    gp.tpb_x = wgs;
                }
                else if((execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE) ||
                        (execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE_XY_Z) ||
                        (execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE_Z_XY))
                {
                    ptr = &FN_PRFX(transpose_var2);
                    gp.tpb_x = 64;
                    gp.tpb_y = 16;
                    
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
                {
                    ptr = &real2complex;
                    gp.b_x = (execPlan.execSeq[i]->length[0]-1)/512 + 1;
                    gp.b_y = execPlan.execSeq[i]->batch;
                    gp.tpb_x = 512; gp.tpb_y = 1;
                }   
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_COPY_CMPLX_TO_R)
                {
                    ptr = &complex2real;
                    gp.b_x = (execPlan.execSeq[i]->length[0]-1)/512 + 1;
                    gp.b_y = execPlan.execSeq[i]->batch;
                    gp.tpb_x = 512; gp.tpb_y = 1;
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
                {
                    ptr = &hermitian2complex;
                    gp.b_x = (execPlan.execSeq[i]->length[0]-1)/512 + 1;
                    gp.b_y = execPlan.execSeq[i]->batch;
                    gp.tpb_x = 512; gp.tpb_y = 1;
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
                {
                    ptr = &complex2hermitian;
                    gp.b_x = (execPlan.execSeq[i]->length[0]-1)/512 + 1;
                    gp.b_y = execPlan.execSeq[i]->batch;
                    gp.tpb_x = 512; gp.tpb_y = 1;
                }
                else
                {
                    std::cout << "should not be in this else block" << std::endl;
                    std::cout << "scheme: " << execPlan.execSeq[i]->scheme << std::endl;
                }

                execPlan.devFnCall.push_back(ptr);
                execPlan.gridParam.push_back(gp);
            
            }//end for
    }// end if(execPlan.execSeq[0]->precision == rocfft_precision_single)
    else if(execPlan.execSeq[0]->precision == rocfft_precision_double)
    {
        
            for(size_t i=0; i<execPlan.execSeq.size(); i++)
            {
                DevFnCall ptr = nullptr;
                GridParam gp;
                size_t bwd,wgs,lds;    

                if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
                {
                    size_t workGroupSize;
                    size_t numTransforms;
                    GetWGSAndNT(execPlan.execSeq[i]->length[0], workGroupSize, numTransforms);//get working group size and number of transforms

                    ptr = function_pool::get_function_double(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM));

                    size_t batch = execPlan.execSeq[i]->batch;
                    for(size_t j=1; j<execPlan.execSeq[i]->length.size(); j++) batch *= execPlan.execSeq[i]->length[j];
                    gp.b_x = (batch%numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
                    gp.tpb_x = workGroupSize;
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
                {
                    ptr = function_pool::get_function_double(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC)) ;

                    GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
                    gp.b_x =  (execPlan.execSeq[i]->length[1])/bwd * execPlan.execSeq[i]->batch;

                    if (execPlan.execSeq[i]->length.size() == 3)
                    {
                        gp.b_x *= execPlan.execSeq[i]->length[2];
                    }
                    gp.tpb_x = wgs;
                }
                else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                {
                    ptr = function_pool::get_function_double(std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC)) ;

                    GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
                    gp.b_x =  (execPlan.execSeq[i]->length[1])/bwd * execPlan.execSeq[i]->batch;

                    if (execPlan.execSeq[i]->length.size() == 3)
                    {
                        gp.b_x *= execPlan.execSeq[i]->length[2];
                    }
                    gp.tpb_x = wgs;
                }
                else if((execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE) ||
                        (execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE_XY_Z) ||
                        (execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE_Z_XY))
                {
                    ptr = &FN_PRFX(transpose_var2);
                    gp.tpb_x = 32;
                    gp.tpb_y = 32;
                }
                else
                {
                    std::cout << "should not be in this else block" << std::endl;
                    std::cout << "scheme: " << execPlan.execSeq[i]->scheme << std::endl;
                }

                execPlan.devFnCall.push_back(ptr);
                execPlan.gridParam.push_back(gp);
            }
        
    }// end if(execPlan.execSeq[0]->precision == rocfft_precision_double)

}

void TransformPowX(const ExecPlan &execPlan, void *in_buffer[], void *out_buffer[], rocfft_execution_info info)
{
    assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    //for(size_t i=0; i<1; i++) //multiple kernels involving transpose
    for(size_t i=0; i<execPlan.execSeq.size(); i++) //multiple kernels involving transpose
    {
        DeviceCallIn data;
        DeviceCallOut back;

        data.node = execPlan.execSeq[i];

        switch(data.node->obIn)
        {
        case OB_USER_IN:                data.bufIn[0] = in_buffer[0]; break;
        case OB_USER_OUT:               data.bufIn[0] = out_buffer[0]; break;
        case OB_TEMP:                   data.bufIn[0] = info->workBuffer; break;
        case OB_TEMP_CMPLX_FOR_REAL:    data.bufIn[0] = (void *)((char *)info->workBuffer + execPlan.tmpWorkBufSize); break;
        default: assert(false);
        }

        switch(data.node->obOut)
        {
        case OB_USER_IN:                data.bufOut[0] = in_buffer[0]; break;
        case OB_USER_OUT:               data.bufOut[0] = out_buffer[0]; break;
        case OB_TEMP:                   data.bufOut[0] = info->workBuffer; break;
        case OB_TEMP_CMPLX_FOR_REAL:    data.bufOut[0] = (void *)((char *)info->workBuffer + execPlan.tmpWorkBufSize); break;
        default: assert(false);
        }

        data.gridParam = execPlan.gridParam[i];

#ifdef TMP_DEBUG
        size_t in_size = data.node->iDist * data.node->batch;
        size_t in_size_bytes = in_size * 2 * sizeof(float);
        void *dbg_in = malloc(in_size_bytes);
        hipMemcpy(dbg_in, data.bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

        size_t out_size = data.node->oDist * data.node->batch;
        size_t out_size_bytes = out_size * 2 * sizeof(float);
        void *dbg_out = malloc(out_size_bytes);
        memset(dbg_out, 0x40, out_size_bytes);
        if(data.node->placement != rocfft_placement_inplace)
        {
            hipMemcpy(data.bufOut[0], dbg_out, out_size_bytes, hipMemcpyHostToDevice);
        }
        printf("attempting kernel: %zu\n", i); fflush(stdout);
#endif

        DevFnCall fn = execPlan.devFnCall[i];
        if(fn)
        {
#ifdef REF_DEBUG 
            // verify results for simple and five-stage scheme not for RC, CC scheme
            printf("\n---------------------------------------------\n");
            printf("\n\nkernel: %zu\n", i); fflush(stdout);
            RefLibOp refLibOp(&data);
#endif
            fn(&data, &back);//execution kernel here
#ifdef REF_DEBUG
            refLibOp.VerifyResult(&data);
#endif
        }
        else
        {
            printf("null ptr function call error\n");
        }

#ifdef TMP_DEBUG
        hipDeviceSynchronize();
        printf("executed kernel: %zu\n", i); fflush(stdout);
        hipMemcpy(dbg_out, data.bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);
        printf("copied from device\n");
       
        /*if(i == 0 || i == 2 || i == 4)
        { 
        float *f_in = (float *)dbg_in;
        float *f_out = (float *)dbg_out;

        for(size_t kr=0; kr<data.node->length[1]; kr++)
        {
            for(size_t kc=0; kc<data.node->length[0]; kc++)
            {
                if(f_in[2*(kr*data.node->length[0] + kc)] != f_out[2*(kc*data.node->length[1] + kr)])
                    printf("fail\n");
                
            }
        }
        }*/

        free(dbg_out);
        free(dbg_in);
#endif

    }
}


