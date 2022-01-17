// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node.h"
#include "function_pool.h"
#include "kernel_launch.h"

NodeMetaData::NodeMetaData(TreeNode* refNode)
{
    if(refNode != nullptr)
    {
        precision  = refNode->precision;
        batch      = refNode->batch;
        direction  = refNode->direction;
        rootIsC2C  = refNode->IsRootPlanC2CTransform();
        deviceProp = refNode->deviceProp;
    }
}

#if !GENERIC_BUF_ASSIGMENT
void LeafNode::AssignBuffers_internal(TraverseState&   state,
                                      OperatingBuffer& flipIn,
                                      OperatingBuffer& flipOut,
                                      OperatingBuffer& obOutBuf)
{
    if(isRootNode())
    {
        obOut = obOutBuf;
    }
    else
    {
        assert(obIn != OB_UNINIT && obOut != OB_UNINIT);
        if(obIn != obOut)
        {
            std::swap(flipIn, flipOut);
        }
    }
}
#endif

bool LeafNode::CreateLargeTwdTable()
{
    if(large1D != 0)
    {
        twiddles_large = twiddles_create(large1D, precision, true, largeTwdBase, false, false, {});
    }

    return true;
}

size_t LeafNode::GetTwiddleTableLength()
{
    // length used by twiddle table is length[0] by default
    // could be override by some special schemes
    return length[0];
}

void LeafNode::GetKernelFactors()
{
    ComputeScheme _scheme = scheme;
    if(_scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z || _scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
       || _scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
        _scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;

    FMKey key     = (dimension == 1) ? fpkey(length[0], precision, _scheme)
                                     : fpkey(length[0], length[1], precision, _scheme);
    kernelFactors = function_pool::get_kernel(key).factors;
}

bool LeafNode::KernelCheck()
{
    if(!externalKernel)
        return true;

    // check we have the kernel
    FMKey key = (dimension == 1) ? fpkey(length[0], precision, scheme)
                                 : fpkey(length[0], length[1], precision, scheme);
    if(!function_pool::has_function(key))
    {
        PrintMissingKernelInfo(key);
        return false;
    }

    GetKernelFactors();
    return true;
}

void LeafNode::SanityCheck()
{
    if(!KernelCheck())
        throw std::runtime_error("Kernel not found");

    TreeNode::SanityCheck();
}

bool LeafNode::CreateDevKernelArgs()
{
    devKernArg = kargs_create(length, inStride, outStride, iDist, oDist);
    return (devKernArg != nullptr);
}

bool LeafNode::CreateTwiddleTableResource()
{
    if(need_twd_table)
    {
        if(!twd_no_radices)
            GetKernelFactors();
        size_t twd_len = GetTwiddleTableLength();
        twiddles       = twiddles_create(twd_len,
                                   precision,
                                   false,
                                   LTWD_BASE_DEFAULT,
                                   twd_no_radices,
                                   twd_attach_2N,
                                   kernelFactors);
    }

    return CreateLargeTwdTable();
}

void LeafNode::SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp)
{
    // derived classes setup the gp (bwd, wgs, lds, padding), funPtr
    SetupGPAndFnPtr_internal(fnPtr, gp);

    // common: sum up the value;
    gp.lds_bytes = (lds + lds_padding * bwd) * sizeof_precision(precision);
    if(scheme == CS_KERNEL_STOCKHAM && ebtype == EmbeddedType::NONE)
    {
        auto key = fpkey(length[0], precision, scheme);
        if(function_pool::has_function(key))
        {
            auto kernel = function_pool::get_kernel(key);

            // NB:
            // Special case on specific arch:
            // For some cases using hald_lds, finer tuning(enlarge) dynamic
            // lds allocation size affects occupancy without changing the
            // kernel code. It is a middle solution between perf and code
            // consistency. Eventually, we need better solution arch
            // specific.
            bool double_half_lds_alloc = false;
            if(is_device_gcn_arch(deviceProp, "gfx90a") && (length[0] == 343 || length[0] == 49))
            {
                double_half_lds_alloc = true;
            }

            if(kernel.half_lds && (!double_half_lds_alloc))
                gp.lds_bytes /= 2;
        }
    }
    return;
}

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/
void TransposeNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    fnPtr    = &FN_PRFX(transpose_var2);
    gp.wgs_x = (precision == rocfft_precision_single) ? 32 : 64;
    gp.wgs_y = (precision == rocfft_precision_single) ? 32 : 16;

    return;
}
