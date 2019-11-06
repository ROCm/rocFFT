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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "rocfft.h"

#include "plan.h"
#include "repo.h"
#include "transform.h"

#include "radix_table.h"

#include "kernel_launch.h"

#include "function_pool.h"
#include "ref_cpu.h"

#include "real2complex.h"

//#define TMP_DEBUG
#ifdef TMP_DEBUG
#include "rocfft_hip.h"
#endif

std::atomic<bool> fn_checked(false);

// This function is called during creation of plan : enqueue the HIP kernels by function
// pointers
void PlanPowX(ExecPlan& execPlan)
{
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        if((execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
           || (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
           || (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC))
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(
                execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->precision, false, false);
        }
        else if((execPlan.execSeq[i]->scheme == CS_KERNEL_R_TO_CMPLX)
                || (execPlan.execSeq[i]->scheme == CS_KERNEL_CMPLX_TO_R))
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(
                2 * execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->precision, false, true);
        }

        if(execPlan.execSeq[i]->large1D != 0)
        {
            execPlan.execSeq[i]->twiddles_large = twiddles_create(
                execPlan.execSeq[i]->large1D, execPlan.execSeq[i]->precision, true, false);
        }
    }
    // copy host buffer to device buffer
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        execPlan.execSeq[i]->devKernArg = kargs_create(execPlan.execSeq[i]->length,
                                                       execPlan.execSeq[i]->inStride,
                                                       execPlan.execSeq[i]->outStride,
                                                       execPlan.execSeq[i]->iDist,
                                                       execPlan.execSeq[i]->oDist);
    }

    if(!fn_checked)
    {
        fn_checked = true;
        function_pool::verify_no_null_functions();
    }

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DevFnCall ptr = nullptr;
        GridParam gp;
        size_t    bwd, wgs, lds;

        switch(execPlan.execSeq[i]->scheme)
        {
        case CS_KERNEL_STOCKHAM:
        {
            // get working group size and number of transforms
            size_t workGroupSize;
            size_t numTransforms;
            GetWGSAndNT(execPlan.execSeq[i]->length[0], workGroupSize, numTransforms);
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(
                          std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM))
                      : function_pool::get_function_double(
                          std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM));
            size_t batch = execPlan.execSeq[i]->batch;
            for(size_t j = 1; j < execPlan.execSeq[i]->length.size(); j++)
                batch *= execPlan.execSeq[i]->length[j];
            gp.b_x
                = (batch % numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
            gp.tpb_x = workGroupSize;
        }
        break;
        case CS_KERNEL_STOCKHAM_BLOCK_CC:
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC))
                      : function_pool::get_function_double(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC));
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = (execPlan.execSeq[i]->length[1]) / bwd * execPlan.execSeq[i]->batch;
            if(execPlan.execSeq[i]->length.size() == 3)
            {
                gp.b_x *= execPlan.execSeq[i]->length[2];
            }
            gp.tpb_x = wgs;
            break;
        case CS_KERNEL_STOCKHAM_BLOCK_RC:
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC))
                      : function_pool::get_function_double(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC));
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = (execPlan.execSeq[i]->length[1]) / bwd * execPlan.execSeq[i]->batch;
            if(execPlan.execSeq[i]->length.size() == 3)
            {
                gp.b_x *= execPlan.execSeq[i]->length[2];
            }
            gp.tpb_x = wgs;
            break;
        case CS_KERNEL_TRANSPOSE:
        case CS_KERNEL_TRANSPOSE_XY_Z:
        case CS_KERNEL_TRANSPOSE_Z_XY:
            ptr      = &FN_PRFX(transpose_var2);
            gp.tpb_x = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 64;
            gp.tpb_y = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 16;
            break;
        case CS_KERNEL_COPY_R_TO_CMPLX:
            ptr      = &real2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_R:
            ptr      = &complex2real;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_HERM_TO_CMPLX:
            ptr      = &hermitian2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
            ptr      = &complex2hermitian;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_R_TO_CMPLX:
            ptr = &r2c_1d_post;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CMPLX_TO_R:
            ptr = &c2r_1d_pre;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CHIRP:
            ptr      = &FN_PRFX(chirp);
            gp.tpb_x = 64;
            break;
        case CS_KERNEL_PAD_MUL:
        case CS_KERNEL_FFT_MUL:
        case CS_KERNEL_RES_MUL:
            ptr      = &FN_PRFX(mul);
            gp.tpb_x = 64;
            break;
        default:
            std::cout << "should not be in this case" << std::endl;
            std::cout << "scheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
        }

        execPlan.devFnCall.push_back(ptr);
        execPlan.gridParam.push_back(gp);
    }
}

void TransformPowX(const ExecPlan&       execPlan,
                   void*                 in_buffer[],
                   void*                 out_buffer[],
                   rocfft_execution_info info)
{
    assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DeviceCallIn  data;
        DeviceCallOut back;

        data.node          = execPlan.execSeq[i];
        data.rocfft_stream = (info == nullptr) ? 0 : info->rocfft_stream;
        size_t inBytes     = (data.node->precision == rocfft_precision_single) ? sizeof(float) * 2
                                                                           : sizeof(double) * 2;

        switch(data.node->obIn)
        {
        case OB_USER_IN:
            data.bufIn[0] = in_buffer[0];
            break;
        case OB_USER_OUT:
            data.bufIn[0] = out_buffer[0];
            break;
        case OB_TEMP:
            data.bufIn[0] = info->workBuffer;
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufIn[0] = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * inBytes);
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufIn[0] = (void*)((char*)info->workBuffer
                                    + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                       + data.node->iOffset)
                                          * inBytes);
            break;
        case OB_UNINIT:
            std::cerr << "Error: operating buffer not initialized for kernel!\n";
            assert(data.node->obIn != OB_UNINIT);
        default:
            std::cerr << "Error: operating buffer not specified for kernel!\n";
            assert(false);
        }

        switch(data.node->obOut)
        {
        case OB_USER_IN:
            data.bufOut[0] = in_buffer[0];
            break;
        case OB_USER_OUT:
            data.bufOut[0] = out_buffer[0];
            break;
        case OB_TEMP:
            data.bufOut[0] = info->workBuffer;
            break;
        case OB_TEMP_CMPLX_FOR_REAL:
            data.bufOut[0] = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * inBytes);
            break;
        case OB_TEMP_BLUESTEIN:
            data.bufOut[0] = (void*)((char*)info->workBuffer
                                     + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                        + data.node->oOffset)
                                           * inBytes);
            break;
        default:
            assert(false);
        }

        data.gridParam = execPlan.gridParam[i];

#ifdef TMP_DEBUG
        size_t in_size = data.node->iDist * data.node->batch;
        // FIXME: real data? double precision
        size_t in_size_bytes = in_size * 2 * sizeof(float);
        void*  dbg_in        = malloc(in_size_bytes);
        hipDeviceSynchronize();
        hipMemcpy(dbg_in, data.bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

        size_t out_size = data.node->oDist * data.node->batch;
        // FIXME: real data? double precision
        size_t out_size_bytes = out_size * 2 * sizeof(float);
        void*  dbg_out        = malloc(out_size_bytes);
        memset(dbg_out, 0x40, out_size_bytes);
        if(data.node->placement != rocfft_placement_inplace)
        {
            hipDeviceSynchronize();
            hipMemcpy(data.bufOut[0], dbg_out, out_size_bytes, hipMemcpyHostToDevice);
        }
        std::cout << "attempting kernel: " << i << std::endl;
#endif

        DevFnCall fn = execPlan.devFnCall[i];
        if(fn)
        {
#ifdef REF_DEBUG
            std::cout << "\n---------------------------------------------\n";
            std::cout << "\n\nkernel: " << i << std::endl;
            std::cout << "\tscheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
            std::cout << "\titype: " << PrintArrayType(execPlan.execSeq[i]->inArrayType)
                      << std::endl;
            std::cout << "\totype: " << PrintArrayType(execPlan.execSeq[i]->outArrayType)
                      << std::endl;
            std::cout << "\tlength: ";
            for(const auto& i : execPlan.execSeq[i]->length)
            {
                std::cout << i << " ";
            }
            std::cout << std::endl;
            std::cout << "\tbatch:   " << execPlan.execSeq[i]->batch << std::endl;
            std::cout << "\tidist:   " << execPlan.execSeq[i]->iDist << std::endl;
            std::cout << "\todist:   " << execPlan.execSeq[i]->oDist << std::endl;
            std::cout << "\tistride:";
            for(const auto& i : execPlan.execSeq[i]->inStride)
            {
                std::cout << " " << i;
            }
            std::cout << std::endl;
            std::cout << "\tostride:";
            for(const auto& i : execPlan.execSeq[i]->outStride)
            {
                std::cout << " " << i;
            }
            std::cout << std::endl;

            RefLibOp refLibOp(&data);
#endif

            // execution kernel:
            fn(&data, &back);

#ifdef REF_DEBUG
            refLibOp.VerifyResult(&data);
#endif
        }
        else
        {
            std::cout << "null ptr function call error\n";
        }

#ifdef TMP_DEBUG
        hipDeviceSynchronize();
        std::cout << "executed kernel: " << i << std::endl;
        hipMemcpy(dbg_out, data.bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);
        std::cout << "copied from device\n";

        float2* f_in  = (float2*)dbg_in;
        float2* f_out = (float2*)dbg_out;
        // temporary print out the kernel output
        std::cout << "input:" << std::endl;
        for(size_t i = 0; i < data.node->iDist * data.node->batch; i++)
        {
            std::cout << f_in[i].x << " " << f_in[i].y << "\n";
        }
        std::cout << "output:" << std::endl;
        for(size_t i = 0; i < data.node->oDist * data.node->batch; i++)
        {
            std::cout << f_out[i].x << " " << f_out[i].y << "\n";
        }
        std::cout << "\n---------------------------------------------\n";
        free(dbg_out);
        free(dbg_in);
#endif
    }
}
