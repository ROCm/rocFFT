// Copyright (C) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "logging.h"
#include "repo.h"
#include "twiddles.h"

TreeNode::~TreeNode()
{
    if(twiddles)
    {
        if(scheme == CS_KERNEL_2D_SINGLE)
            Repo::ReleaseTwiddle2D(twiddles);
        else
            Repo::ReleaseTwiddle1D(twiddles);
        twiddles = nullptr;
    }
    if(twiddles_large)
    {
        Repo::ReleaseTwiddle1D(twiddles_large);
        twiddles_large = nullptr;
    }
}

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
        std::tie(twiddles_large, twiddles_large_size)
            = Repo::GetTwiddles1D(large1D, 0, precision, largeTwdBase, false, {});
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
        if(LOG_TRACE_ENABLED())
            (*LogSingleton::GetInstance().GetTraceOS()) << PrintMissingKernelInfo(key);

        return false;
    }

    dir2regMode = (function_pool::get_kernel(key).direct_to_from_reg)
                      ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                      : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;

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
        size_t twd_len                    = GetTwiddleTableLength();
        std::tie(twiddles, twiddles_size) = Repo::GetTwiddles1D(
            twd_len, GetTwiddleTableLengthLimit(), precision, 0, twd_attach_halfN, kernelFactors);
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
    if((scheme == CS_KERNEL_STOCKHAM_BLOCK_CC || scheme == CS_KERNEL_STOCKHAM_BLOCK_CR)
       && (dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT) && (ebtype == EmbeddedType::NONE))
    {
        auto key = fpkey(length[0], precision, scheme);
        if(function_pool::has_function(key))
        {
            auto kernel = function_pool::get_kernel(key);
            if(kernel.half_lds)
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

void TreeNode::SetTransposeOutputLength()
{
    switch(scheme)
    {
    case CS_KERNEL_TRANSPOSE:
    {
        outputLength = length;
        std::swap(outputLength[0], outputLength[1]);
        break;
    }
    case CS_KERNEL_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    {
        outputLength = length;
        std::swap(outputLength[1], outputLength[2]);
        std::swap(outputLength[0], outputLength[1]);
        break;
    }
    case CS_KERNEL_TRANSPOSE_Z_XY:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    {
        outputLength = length;
        std::swap(outputLength[0], outputLength[1]);
        std::swap(outputLength[1], outputLength[2]);
        break;
    }
    default:
        throw std::runtime_error("can't set transpose output length on non-transpose node");
    }
}

void TreeNode::CollapseContiguousDims()
{
    // collapse children
    for(auto& child : childNodes)
        child->CollapseContiguousDims();

    const auto collapsibleDims = CollapsibleDims();
    if(collapsibleDims.empty())
        return;

    // utility function to collect the dims to collapse
    auto collectCollapse = [&collapsibleDims](const size_t               dist,
                                              size_t&                    newBatch,
                                              const std::vector<size_t>& length,
                                              const std::vector<size_t>& stride) {
        std::vector<size_t> dimsToCollapse;
        // start with batch dim and go backwards through collapsible dims
        // so we can collapse them without invalidating remaining indexes
        auto curStride = dist;
        for(auto i = collapsibleDims.rbegin(); i != collapsibleDims.rend(); ++i)
        {
            if(curStride % stride[*i] != 0)
                break;
            if(curStride / stride[*i] != length[*i])
                break;
            dimsToCollapse.push_back(*i);
            newBatch *= length[*i];
            curStride = stride[*i];
        }
        return dimsToCollapse;
    };

    // utility function to actually do the collapsing -
    // dimsToCollapse must be in reverse order so we erase dims from
    // highest to lowest
    auto doCollapse = [](size_t&                    dist,
                         const std::vector<size_t>& dimsToCollapse,
                         std::vector<size_t>&       lengthToCollapse,
                         std::vector<size_t>&       strideToCollapse) {
        for(auto i : dimsToCollapse)
        {
            dist /= lengthToCollapse[i];
            lengthToCollapse.erase(lengthToCollapse.begin() + i);
            strideToCollapse.erase(strideToCollapse.begin() + i);
        }
    };

    size_t              newInputBatch = batch;
    std::vector<size_t> inputDimsToCollapse
        = collectCollapse(iDist, newInputBatch, length, inStride);
    auto                outputLengthTemp = GetOutputLength();
    size_t              newOutputBatch   = batch;
    std::vector<size_t> outputDimsToCollapse
        = collectCollapse(oDist, newOutputBatch, outputLengthTemp, outStride);
    if(inputDimsToCollapse != outputDimsToCollapse || newInputBatch != newOutputBatch)
        return;

    if(!inputDimsToCollapse.empty())
    {
        std::stringstream msg;
        msg << "collapsed contiguous high length(s)";
        for(auto i = inputDimsToCollapse.rbegin(); i != inputDimsToCollapse.rend(); ++i)
            msg << " " << length[*i];
        msg << " into batch";
        comments.push_back(msg.str());
    }

    doCollapse(iDist, inputDimsToCollapse, length, inStride);
    doCollapse(oDist, outputDimsToCollapse, outputLengthTemp, outStride);
    batch = newInputBatch;

    if(!outputLength.empty())
        outputLength = outputLengthTemp;
}
