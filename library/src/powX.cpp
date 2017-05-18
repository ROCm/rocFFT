/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <vector>
#include <assert.h>
#include <iostream>
#include <unordered_map>

#include "rocfft.h"
#include "plan.h"
#include "repo.h"
#include "transform.h"
#include "twiddles.h"
#include "kernel_launch.h"
#include "function_pool.h"
#ifdef TMP_DEBUG
#include <hip/hip_runtime.h>
#endif

/* this function is called during creation of plan of pow 2: enqueue the HIP kernels by function pointers*/
void PlanPowX(ExecPlan &execPlan)
{
    for(size_t i=0; i<execPlan.execSeq.size(); i++)
    {
        if(    (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) ||
            (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) ||
            (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) )
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(execPlan.execSeq[i]->length[0], execPlan.execSeq[0]->precision);
        }

        if(execPlan.execSeq[i]->large1D != 0)
        {
            execPlan.execSeq[i]->twiddles_large = twiddles_create(execPlan.execSeq[i]->large1D, execPlan.execSeq[0]->precision);
        }
    }

    //function_pool func_pool;

    if(execPlan.execSeq[0]->precision == rocfft_precision_single)
    {
        if(execPlan.rootPlan->dimension == 1)
        {
            if(execPlan.execSeq.size() == 1)
            {
                if(
                    (execPlan.execSeq[0]->inArrayType == rocfft_array_type_complex_interleaved) &&
                    (execPlan.execSeq[0]->outArrayType == rocfft_array_type_complex_interleaved) )
                {
                    assert(execPlan.execSeq[0]->length[0] <= 4096);

                    size_t workGroupSize;
                    size_t numTransforms;
                    DevFnCall ptr = nullptr;// typedef void (*DevFnCall)(void *, void *);
                    GetWGSAndNT(execPlan.execSeq[0]->length[0], workGroupSize, numTransforms);//get working group size and number of transforms

                    //ptr = func_pool.get_function_single(execPlan.execSeq[0]->length[0]);
                    switch(execPlan.execSeq[0]->length[0])
                    {

                            //pow2
                            //assume the hash map has been set up, mymap
                            //ptr = mymap[ execPlan.execSeq[0]->length[0] ];

                            case 4096: ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_4096); break;
                            case 2048: ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_2048); break;
                            case 1024: ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_1024); break;
                            case 512:  ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_512); break;
                            case 256:  ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_256); break;
                            case 128:  ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_128); break;
                            case 64:   ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_64); break;
                            case 32:   ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_32); break;
                            case 16:   ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_16); break;
                            case 8:    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_8); break;
                            case 4:    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_4); break;
                            case 2:    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_2); break;
                            case 1:    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_1); break;


                            //pow3
                            case 2187:      ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_2187); break;
                            case 729:       ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_729); break;
                            case 243:       ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_243); break;
                            case 81:        ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_81); break;
                            case 27:        ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_27); break;
                            case 9:         ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_9); break;
                            case 3:         ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_3); break;

                            //pow5
                            case 3125:      ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_3125); break;
                            case 625:       ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_625); break;
                            case 125:       ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_125); break;
                            case 25:        ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_25); break;
                            case 5:         ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_5); break;

                    }

                    execPlan.devFnCall.push_back(ptr);
                    GridParam gp;
                    size_t batch = execPlan.execSeq[0]->batch;
                    gp.b_x = (batch%numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
                    gp.tpb_x = workGroupSize;
                    execPlan.gridParam.push_back(gp);
                }//
            }// if(execPlan.execSeq.size() == 1)
            else
            {
                for(size_t i=0; i<execPlan.execSeq.size(); i++)
                {
                    DevFnCall ptr = nullptr;
                    GridParam gp;

                    if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq[i]->length.size() == 2) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_128);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_256);
                            gp.b_x = 16 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_128_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }

                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 2048) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_2048);
                            gp.b_x = 128 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 4096) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_4096);
                            gp.b_x = 256 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq[i]->length.size() == 3) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_3_64_128);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_3_64_256);
                            gp.b_x = 16 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_3_128_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbcc_3_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (execPlan.execSeq[i]->length.size() == 2) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_128_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_128);
                            gp.b_x = 16 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (execPlan.execSeq[i]->length.size() == 3) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_3_128_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_3_256_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_3_256_128);
                            gp.b_x = 16 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_sp_op_ci_ci_sbrc_3_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) && (execPlan.execSeq.size() == 3) )
                    {
                        gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;

                        switch(execPlan.execSeq[i]->length[0])
                        {
                            case 4096:     gp.tpb_x = 256;
                                    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_4096); break;
                            case 2048:     gp.tpb_x = 256;
                                    ptr = &FN_PRFX(dfn_sp_ci_ci_stoc_1_2048); break;
                            default: assert(false);
                        }
                    }
                    else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
                    {
                        gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;

                        if(execPlan.execSeq[i]->placement == rocfft_placement_inplace)
                        {
                            switch(execPlan.execSeq[i]->length[0])
                            {
                                case 4096:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_4096); break;
                                case 2048:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_2048); break;
                                case 1024:     gp.tpb_x = 128;
                                        ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_1024); break;
                                case 512:     gp.tpb_x = 64;
                                        ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_512); break;
                                default: assert(false);
                            }
                        }
                        else
                        {
                            switch(execPlan.execSeq[i]->length[0])
                            {
                                case 4096:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_4096); break;
                                case 2048:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_2048); break;
                                case 1024:     gp.tpb_x = 128;
                                        ptr = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_1024); break;
                                case 512:     gp.tpb_x = 64;
                                        ptr = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_512); break;
                                default: assert(false);
                            }
                        }
                    }
                    else if(execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE)
                    {
                        ptr = &FN_PRFX(transpose_var1_sp);
                        gp.tpb_x = 16;
                        gp.tpb_y = 16;
                        if(execPlan.execSeq[i]->transTileDir == TTD_IP_HOR)
                        {
                            gp.b_x = execPlan.execSeq[i]->length[0] / 64;
                            gp.b_y = (execPlan.execSeq[i]->length[1] / 64) * execPlan.execSeq[i]->batch;
                        }
                        else
                        {
                            gp.b_x = execPlan.execSeq[i]->length[1] / 64;
                            gp.b_y = (execPlan.execSeq[i]->length[0] / 64) * execPlan.execSeq[i]->batch;
                        }
                    }

                    execPlan.devFnCall.push_back(ptr);
                    execPlan.gridParam.push_back(gp);
                }
            }
        }
    }// end if(execPlan.execSeq[0]->precision == rocfft_precision_single)
    else if(execPlan.execSeq[0]->precision == rocfft_precision_double)
    {
        if(execPlan.rootPlan->dimension == 1)
        {
            if(execPlan.execSeq.size() == 1)
            {
                if(
                    (execPlan.execSeq[0]->inArrayType == rocfft_array_type_complex_interleaved) &&
                    (execPlan.execSeq[0]->outArrayType == rocfft_array_type_complex_interleaved) )
                {
                    assert(execPlan.execSeq[0]->length[0] <= 4096);

                    size_t workGroupSize;// work group size
                    size_t numTransforms;
                    DevFnCall ptr = nullptr;
                    GetWGSAndNT(execPlan.execSeq[0]->length[0], workGroupSize, numTransforms);//get working group size and number of transforms

                    switch(execPlan.execSeq[0]->length[0])
                    {
                            //pow2
                            case 4096: ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_4096); break;
                            case 2048: ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_2048); break;
                            case 1024: ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_1024); break;
                            case 512:  ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_512); break;
                            case 256:  ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_256); break;
                            case 128:  ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_128); break;
                            case 64:   ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_64); break;
                            case 32:   ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_32); break;
                            case 16:   ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_16); break;
                            case 8:    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_8); break;
                            case 4:    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_4); break;
                            case 2:    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_2); break;
                            case 1:    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_1); break;

                            //pow3
                            case 2187:      ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_2187); break;
                            case 729:       ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_729); break;
                            case 243:       ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_243); break;
                            case 81:        ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_81); break;
                            case 27:        ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_27); break;
                            case 9:         ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_9); break;
                            case 3:         ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_3); break;

                            //pow5
                            case 3125:      ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_3125); break;
                            case 625:       ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_625); break;
                            case 125:       ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_125); break;
                            case 25:        ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_25); break;
                            case 5:         ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_5); break;

                    }

                    execPlan.devFnCall.push_back(ptr);

                    GridParam gp;
                    size_t batch = execPlan.execSeq[0]->batch;
                    gp.b_x = (batch%numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
                    gp.tpb_x = workGroupSize;
                    execPlan.gridParam.push_back(gp);
                }
            }
            else
            {
                for(size_t i=0; i<execPlan.execSeq.size(); i++)
                {
                    DevFnCall ptr = nullptr;
                    GridParam gp;

                    if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq[i]->length.size() == 2) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_64_128);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_64_256);
                            gp.b_x = 16 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_128_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }

                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 2048) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_64_2048);
                            gp.b_x = 128 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 4096) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_2_64_4096);
                            gp.b_x = 256 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq[i]->length.size() == 3) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_3_64_128);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_3_64_256);
                            gp.b_x = 16 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_3_128_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbcc_3_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (execPlan.execSeq[i]->length.size() == 2) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_2_128_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_2_256_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_2_256_128);
                            gp.b_x = 16 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_2_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (execPlan.execSeq[i]->length.size() == 3) )
                    {
                        if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_3_128_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 128;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 64) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_3_256_64);
                            gp.b_x = 8 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 128) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_3_256_128);
                            gp.b_x = 16 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                        if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
                        {
                            ptr = &FN_PRFX(dfn_dp_op_ci_ci_sbrc_3_256_256);
                            gp.b_x = 32 * execPlan.execSeq[i]->length[2] * execPlan.execSeq[i]->batch;
                            gp.tpb_x = 256;
                        }
                    }
                    else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) && (execPlan.execSeq.size() == 3) )
                    {
                        gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;

                        switch(execPlan.execSeq[i]->length[0])
                        {
                            case 4096:     gp.tpb_x = 256;
                                    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_4096); break;
                            case 2048:     gp.tpb_x = 256;
                                    ptr = &FN_PRFX(dfn_dp_ci_ci_stoc_1_2048); break;
                            default: assert(false);
                        }
                    }
                    else if(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
                    {
                        gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;

                        if(execPlan.execSeq[i]->placement == rocfft_placement_inplace)
                        {
                            switch(execPlan.execSeq[i]->length[0])
                            {
                                case 4096:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_dp_ip_ci_ci_stoc_2_4096); break;
                                case 2048:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_dp_ip_ci_ci_stoc_2_2048); break;
                                case 1024:     gp.tpb_x = 128;
                                        ptr = &FN_PRFX(dfn_dp_ip_ci_ci_stoc_2_1024); break;
                                case 512:     gp.tpb_x = 64;
                                        ptr = &FN_PRFX(dfn_dp_ip_ci_ci_stoc_2_512); break;
                                default: assert(false);
                            }
                        }
                        else
                        {
                            switch(execPlan.execSeq[i]->length[0])
                            {
                                case 4096:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_dp_op_ci_ci_stoc_2_4096); break;
                                case 2048:     gp.tpb_x = 256;
                                        ptr = &FN_PRFX(dfn_dp_op_ci_ci_stoc_2_2048); break;
                                case 1024:     gp.tpb_x = 128;
                                        ptr = &FN_PRFX(dfn_dp_op_ci_ci_stoc_2_1024); break;
                                case 512:     gp.tpb_x = 64;
                                        ptr = &FN_PRFX(dfn_dp_op_ci_ci_stoc_2_512); break;
                                default: assert(false);
                            }
                        }
                    }
                    else if(execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE)
                    {
                        ptr = &FN_PRFX(transpose_var1_dp);
                        gp.tpb_x = 16;
                        gp.tpb_y = 16;
                        if(execPlan.execSeq[i]->transTileDir == TTD_IP_HOR)
                        {
                            gp.b_x = execPlan.execSeq[i]->length[0] / 64;
                            gp.b_y = (execPlan.execSeq[i]->length[1] / 64) * execPlan.execSeq[i]->batch;
                        }
                        else
                        {
                            gp.b_x = execPlan.execSeq[i]->length[1] / 64;
                            gp.b_y = (execPlan.execSeq[i]->length[0] / 64) * execPlan.execSeq[i]->batch;
                        }
                    }

                    execPlan.devFnCall.push_back(ptr);
                    execPlan.gridParam.push_back(gp);
                }
            }
        }
    }// end if(execPlan.execSeq[0]->precision == rocfft_precision_double)

}

void TransformPowX(const ExecPlan &execPlan, void *in_buffer[], void *out_buffer[], rocfft_execution_info info)
{
    assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    if(execPlan.rootPlan->dimension == 1)
    {
        if(execPlan.execSeq.size() == 1) // one kernel 
        {
            DeviceCallIn data;
            DeviceCallOut back;

            data.node = execPlan.execSeq[0];
            data.bufIn[0] = in_buffer[0];
            data.bufOut[0] = out_buffer[0];
            data.gridParam = execPlan.gridParam[0];

            DevFnCall fn = execPlan.devFnCall[0];
            fn(&data, &back);//execution kernel here
        }
        else
        {
            for(size_t i=0; i<execPlan.execSeq.size(); i++) //multiple kernels involving transpose
            {
                DeviceCallIn data;
                DeviceCallOut back;

                data.node = execPlan.execSeq[i];

                switch(data.node->obIn)
                {
                case OB_USER_IN:    data.bufIn[0] = in_buffer[0]; break;
                case OB_USER_OUT:    data.bufIn[0] = out_buffer[0]; break;
                case OB_TEMP:        data.bufIn[0] = info->workBuffer; break;
                default: assert(false);
                }

                switch(data.node->obOut)
                {
                case OB_USER_IN:    data.bufOut[0] = in_buffer[0]; break;
                case OB_USER_OUT:    data.bufOut[0] = out_buffer[0]; break;
                case OB_TEMP:        data.bufOut[0] = info->workBuffer; break;
                default: assert(false);
                }

                data.gridParam = execPlan.gridParam[i];

#ifdef TMP_DEBUG
		size_t out_size = data.node->oDist * data.node->batch;
		size_t out_size_bytes = out_size * 2 * sizeof(float);
		void *dbg_out = malloc(out_size_bytes);
		memset(dbg_out, 0x40, out_size_bytes);
		if(data.node->placement != rocfft_placement_inplace)
		{
			hipMemcpy(data.bufOut[0], dbg_out, out_size_bytes, hipMemcpyHostToDevice);
		}
		printf("in debug block of kernel: %zu\n", i);
#endif

                DevFnCall fn = execPlan.devFnCall[i];
                fn(&data, &back);

#ifdef TMP_DEBUG
		hipDeviceSynchronize();
		hipMemcpy(dbg_out, data.bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);
		printf("copied from device\n");
		free(dbg_out);
#endif
            }
        }
    }
}


