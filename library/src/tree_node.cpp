// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../shared/precision_type.h"
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
    if(chirp)
    {
        Repo::ReleaseChirp(chirp);
        chirp = nullptr;
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

bool LeafNode::CreateLargeTwdTable()
{
    if(large1D != 0)
    {
        std::tie(twiddles_large, twiddles_large_size)
            = Repo::GetTwiddles1D(large1D, 0, precision, deviceProp, largeTwdBase, false, {});
    }

    return true;
}

size_t LeafNode::GetTwiddleTableLength()
{
    // length used by twiddle table is length[0] by default
    // could be override by some special schemes
    return length[0];
}

FMKey LeafNode::GetKernelKey() const
{
    if(!externalKernel)
        return FMKey::EmptyFMKey();

    return TreeNode::GetKernelKey();
}

void LeafNode::GetKernelFactors()
{
    FMKey key     = GetKernelKey();
    kernelFactors = function_pool::get_kernel(key).factors;
}

bool LeafNode::KernelCheck(std::vector<FMKey>& kernel_keys)
{
    if(!externalKernel)
    {
        // such as solutions kernels for 2D_RTRT or 1D_CRT, the "T" kernel is not an external one
        // so in the solution map we will keep it as a empty key. By storing and checking the emptykey,
        // we can increase the reilability of solution map.
        if(!kernel_keys.empty())
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "solution kernel is an built-in kernel" << std::endl;

            // kernel_key from solution map should be an EmptyFMKey for a built-in kernel
            if(kernel_keys.front() != FMKey::EmptyFMKey())
                return false;
            kernel_keys.erase(kernel_keys.begin());
        }
        return true;
    }

    specified_key = nullptr;
    if(!kernel_keys.empty())
    {
        FMKey assignedKey = kernel_keys.front();
        kernel_keys.erase(kernel_keys.begin());

        // for kernels that haven't been supported tuning. (i.e. 2D_SINGLE)
        // don't assign specified_key
        if(assignedKey == FMKey::EmptyFMKey())
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "kernel is not tuned, use default kernel." << std::endl;
        }
        // check if the assigned key is consistent with the node information
        else if((length[0] != assignedKey.lengths[0])
                || (dimension == 2 && length[1] != assignedKey.lengths[1])
                || (precision != assignedKey.precision) || (scheme != assignedKey.scheme)
                || (ebtype != assignedKey.kernel_config.ebType))
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "solution kernel keys are invalid: key properties != node's properties"
                    << std::endl;
            return false;
        }
        else
        {
            // get sbrc_trans_type from assignedKey (for sbrc)
            sbrcTranstype = assignedKey.sbrcTrans;

            function_pool::add_new_kernel(assignedKey);
            specified_key = std::make_unique<FMKey>(assignedKey);
        }
    }

    // get the final key and check if we have the kernel.
    // Note that the check is trivial if we are using "specified_key"
    // since we definitly have the kernel, but not trivial if it's the auto-gen key
    FMKey key = GetKernelKey();
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

void LeafNode::SanityCheck(SchemeTree* solution_scheme, std::vector<FMKey>& kernels_keys)
{
    if(!KernelCheck(kernels_keys))
        throw std::runtime_error("Kernel not found or mismatches node (solution map issue)");

    TreeNode::SanityCheck(solution_scheme, kernels_keys);
}

void LeafNode::Print(rocfft_ostream& os, int indent) const
{
    TreeNode::Print(os, indent);

    std::string indentStr;
    while(indent--)
        indentStr += "    ";

    os << indentStr.c_str() << "Leaf-Node: external-kernel configuration: ";
    indentStr += "    ";
    os << "\n" << indentStr.c_str() << "workgroup_size: " << wgs;
    os << "\n" << indentStr.c_str() << "trans_per_block: " << bwd;
    os << "\n" << indentStr.c_str() << "radices: [ ";
    for(size_t i = 0; i < kernelFactors.size(); i++)
    {
        os << kernelFactors[i] << " ";
    }
    os << "]\n";
}

bool LeafNode::CreateDevKernelArgs()
{
    devKernArg = kargs_create(length, inStride, outStride, iDist, oDist);
    return (devKernArg != nullptr);
}

bool LeafNode::CreateDeviceResources()
{
    if(need_chirp)
    {
        std::tie(chirp, chirp_size) = Repo::GetChirp(lengthBlueN, precision, deviceProp);
    }

    if(need_twd_table)
    {
        if(!twd_no_radices)
            GetKernelFactors();
        size_t twd_len                    = GetTwiddleTableLength();
        std::tie(twiddles, twiddles_size) = Repo::GetTwiddles1D(twd_len,
                                                                GetTwiddleTableLengthLimit(),
                                                                precision,
                                                                deviceProp,
                                                                0,
                                                                twd_attach_halfN,
                                                                kernelFactors);
    }

    return CreateLargeTwdTable();
}

void LeafNode::SetupGridParamAndFuncPtr(DevFnCall& fnPtr, GridParam& gp)
{
    // derived classes setup the gp (bwd, wgs, lds, padding), funPtr
    SetupGPAndFnPtr_internal(fnPtr, gp);

    auto key = GetKernelKey();

    // common: sum up the value;
    gp.lds_bytes = lds * complex_type_size(precision);
    if(scheme == CS_KERNEL_STOCKHAM && ebtype == EmbeddedType::NONE)
    {
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
    // SBCC support half-lds conditionally
    if((scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
       && (dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT) && (ebtype == EmbeddedType::NONE))
    {
        if(function_pool::has_function(key))
        {
            auto kernel = function_pool::get_kernel(key);
            if(kernel.half_lds)
                gp.lds_bytes /= 2;
        }
    }
    // NB:
    //   SBCR / SBRC are not able to use half-lds due to both of them can't satisfy dir-to/from-registers at them same time.

    // Confirm that the requested LDS bytes will fit into what the
    // device can provide.  If it can't, we've made a mistake in our
    // computation somewhere.
    if(gp.lds_bytes > deviceProp.sharedMemPerBlock)
        throw std::runtime_error(std::to_string(gp.lds_bytes)
                                 + " bytes of LDS requested, but device only provides "
                                 + std::to_string(deviceProp.sharedMemPerBlock));
}

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/

// grid params are set up by RTC
void TransposeNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) {}

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

bool TreeNode::IsBluesteinChirpSetup()
{
    // setup nodes must be under a bluestein parent. multi-kernel fused
    // bluestein is an exception to this rule as the first two chirp + padding
    // nodes are under an L1D_CC node.
    if(typeBlue != BT_MULTI_KERNEL_FUSED && (parent == nullptr || parent->scheme != CS_BLUESTEIN))
        return false;
    // bluestein could either be 3-kernel plan (so-called single kernel Bluestein),
    // meaning the first two are setup kernels, or multi-kernel bluestein (fused or non-fused)
    // where only the first is setup
    switch(parent->typeBlue)
    {
    case BluesteinType::BT_NONE:
        return false;
    case BluesteinType::BT_SINGLE_KERNEL:
        return this == parent->childNodes[0].get() || this == parent->childNodes[1].get();
    case BluesteinType::BT_MULTI_KERNEL:
        return this == parent->childNodes[0].get();
    case BluesteinType::BT_MULTI_KERNEL_FUSED:
        return (fuseBlue == BFT_FWD_CHIRP) ? true : false;
    }

    throw std::runtime_error("unexpected bluestein plan shape");
}
