// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node_3D.h"
#include "function_pool.h"
#include "node_factory.h"
#include "radix_table.h"

SBRC_TRANSPOSE_TYPE sbrc_3D_transpose_type(unsigned int               blockWidth,
                                           size_t                     alignment_dimension,
                                           const std::vector<size_t>& length)
{
    if(is_diagonal_sbrc_3D_length(length.front()) && is_cube_size(length))
        return DIAGONAL;
    if(alignment_dimension % blockWidth == 0)
        return TILE_ALIGNED;
    return TILE_UNALIGNED;
}

/*****************************************************
 * 3D_RTRT  *
 *****************************************************/
void RTRT3DNode::BuildTree_internal()
{
    // 2d fft
    NodeMetaData xyPlanData(this);
    xyPlanData.length    = length;
    xyPlanData.dimension = 2;
    auto xyPlan          = NodeFactory::CreateExplicitNode(xyPlanData, this);
    xyPlan->RecursiveBuildTree();

    // first transpose
    auto trans1Plan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
    trans1Plan->length    = length;
    trans1Plan->dimension = 2;

    // z fft
    NodeMetaData zPlanData(this);
    zPlanData.dimension = 1;
    zPlanData.length.push_back(length[2]);
    zPlanData.length.push_back(length[0]);
    zPlanData.length.push_back(length[1]);
    auto zPlan = NodeFactory::CreateExplicitNode(zPlanData, this);
    zPlan->RecursiveBuildTree();

    // second transpose
    auto trans2Plan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
    trans2Plan->length    = zPlan->length;
    trans2Plan->dimension = 2;

    // RTRT
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(zPlan)); // notice that move() will set zPlan to nullptr
    childNodes.emplace_back(std::move(trans2Plan));
}

void RTRT3DNode::AssignParams_internal()
{
    assert(childNodes.size() == 4);

    const size_t biggerDim  = std::max(length[0] * length[1], length[2]);
    const size_t smallerDim = std::min(length[0] * length[1], length[2]);
    const size_t padding
        = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512) ? 64 : 0;

    // B -> B
    auto& xyPlan = childNodes[0];
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    // B -> T
    auto& trans1Plan = childNodes[1];
    assert(trans1Plan->obOut == OB_TEMP);
    trans1Plan->inStride = xyPlan->outStride;
    trans1Plan->iDist    = xyPlan->oDist;

    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[2] + padding);
    trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[1]);
    trans1Plan->oDist = trans1Plan->length[1] * trans1Plan->outStride[2];

    for(size_t index = 3; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    auto& zPlan = childNodes[2];
    assert(zPlan->obOut == OB_TEMP);
    zPlan->inStride = trans1Plan->outStride;
    zPlan->iDist    = trans1Plan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->AssignParams();

    // T -> B
    auto& trans2Plan = childNodes[3];
    assert((trans2Plan->obOut == OB_USER_OUT) || (trans2Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (trans2Plan->obOut == OB_TEMP_BLUESTEIN));
    trans2Plan->inStride = zPlan->outStride;
    trans2Plan->iDist    = zPlan->oDist;

    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

/*****************************************************
 * 3D_TRTRTR  *
 *****************************************************/
void TRTRTR3DNode::BuildTree_internal()
{
    std::vector<size_t> cur_length = length;

    for(int i = 0; i < 6; i += 2)
    {
        // transpose Z_XY
        auto trans_plan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans_plan->length    = cur_length;
        trans_plan->dimension = 2;

        std::swap(cur_length[0], cur_length[1]);
        std::swap(cur_length[1], cur_length[2]);

        // row ffts
        NodeMetaData row_plan_data(this);
        row_plan_data.length    = cur_length;
        row_plan_data.dimension = 1;
        auto row_plan           = NodeFactory::CreateExplicitNode(row_plan_data, this);
        row_plan->RecursiveBuildTree();

        // TR
        childNodes.emplace_back(std::move(trans_plan));
        childNodes.emplace_back(std::move(row_plan));
    }
}

void TRTRTR3DNode::AssignParams_internal()
{
    assert(scheme == CS_3D_TRTRTR);
    assert(childNodes.size() == 6);

    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];
        if(i == 0)
        {
            trans_plan->inStride = inStride;
            trans_plan->iDist    = iDist;
        }
        else
        {
            trans_plan->inStride = childNodes[i - 1]->outStride;
            trans_plan->iDist    = childNodes[i - 1]->oDist;
        }

        trans_plan->outStride.push_back(1);
        trans_plan->outStride.push_back(trans_plan->outStride[0] * trans_plan->length[1]);
        trans_plan->outStride.push_back(trans_plan->outStride[1] * trans_plan->length[2]);
        trans_plan->oDist = trans_plan->outStride[2] * trans_plan->length[0];

        auto& row_plan     = childNodes[i + 1];
        row_plan->inStride = trans_plan->outStride;
        row_plan->iDist    = trans_plan->oDist;

        if(i == 4)
        {
            row_plan->outStride = outStride;
            row_plan->oDist     = oDist;
        }
        else
        {
            row_plan->outStride.push_back(1);
            row_plan->outStride.push_back(row_plan->outStride[0] * row_plan->length[0]);
            row_plan->outStride.push_back(row_plan->outStride[1] * row_plan->length[1]);
            row_plan->oDist = row_plan->outStride[2] * row_plan->length[2];
        }
        row_plan->AssignParams();
    }
}

void TRTRTR3DNode::AssignBuffers_internal(TraverseState&   state,
                                          OperatingBuffer& flipIn,
                                          OperatingBuffer& flipOut,
                                          OperatingBuffer& obOutBuf)
{
    assert(childNodes.size() == 6);

    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];

        // T
        trans_plan->SetInputBuffer(state);
        trans_plan->obOut        = OB_TEMP;
        trans_plan->inArrayType  = (i == 0) ? inArrayType : childNodes[i - 1]->outArrayType;
        trans_plan->outArrayType = rocfft_array_type_complex_interleaved;

        auto& row_plan = childNodes[i + 1];
        row_plan->SetInputBuffer(state);
        row_plan->obOut        = obOutBuf;
        row_plan->inArrayType  = rocfft_array_type_complex_interleaved;
        row_plan->outArrayType = outArrayType;
        row_plan->AssignBuffers(state, flipIn, flipOut, obOutBuf);
    }

    obOut = childNodes[childNodes.size() - 1]->obOut;
}

/*****************************************************
 * CS_3D_BLOCK_RC  *
 *****************************************************/
void BLOCKRC3DNode::BuildTree_internal()
{
    std::vector<size_t> cur_length = length;

    size_t total_sbrc = 0;
    for(int i = 0; i < 3; ++i)
    {
        // if we have an sbrc kernel for this length, use it,
        // otherwise, fall back to row FFT+transpose
        bool have_sbrc = function_pool::has_SBRC_kernel(cur_length.front(), precision);
        // ensure the kernel would be tile-aligned
        if(have_sbrc)
        {
            size_t bwd, wgs, lds;
            GetBlockComputeTable(cur_length[0], bwd, wgs, lds);
            if(cur_length[2] % bwd != 0)
                have_sbrc = false;
        }
        if(have_sbrc)
            ++total_sbrc;

        // if we're doing in-place, we are unable to do more than 2
        // sbrc kernels.  we'd ideally want 3 sbrc, but each needs to
        // be out-of-place:
        // - kernel 1: IN -> TEMP
        // - kernel 2: TEMP -> IN
        // - kernel 3: IN -> ???
        //
        // So we have no way to put the results back into IN.  So
        // limit ourselves to 2 sbrc kernels in that case.
        if(total_sbrc >= 3 && placement == rocfft_placement_inplace)
            have_sbrc = false;

        if(have_sbrc)
        {
            auto sbrc_node
                = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z, this);
            sbrc_node->length = cur_length;
            childNodes.emplace_back(std::move(sbrc_node));
        }
        else
        {
            // row ffts
            NodeMetaData row_plan_data(this);
            row_plan_data.length    = cur_length;
            row_plan_data.dimension = 1;
            auto row_plan           = NodeFactory::CreateExplicitNode(row_plan_data, this);
            row_plan->RecursiveBuildTree();

            // transpose XY_Z
            auto trans_plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
            trans_plan->length = cur_length;
            trans_plan->dimension = 2;

            // RT
            childNodes.emplace_back(std::move(row_plan));
            childNodes.emplace_back(std::move(trans_plan));
        }

        std::swap(cur_length[2], cur_length[1]);
        std::swap(cur_length[1], cur_length[0]);
    }
}

void BLOCKRC3DNode::AssignParams_internal()
{
    assert(scheme == CS_3D_BLOCK_RC);
    // could go as low as 3 kernels if all dimensions are SBRC-able,
    // but less than 6.  If we ended up with 6 we should have just
    // done 3D_TRTRTR instead.
    assert(childNodes.size() >= 3 && childNodes.size() < 6);

    childNodes.front()->inStride = inStride;
    childNodes.front()->iDist    = iDist;

    std::vector<size_t> prev_outStride;
    size_t              prev_oDist = 0;
    for(auto& node : childNodes)
    {
        // set initial inStride + iDist, or connect it to previous node
        if(prev_outStride.empty())
        {
            node->inStride = inStride;
            node->iDist    = iDist;
        }
        else
        {
            node->inStride = prev_outStride;
            node->iDist    = prev_oDist;
        }

        // each node is either:
        // - a fused sbrc+transpose node (for dimensions we have SBRC kernels for)
        // - a transpose node or a row FFT node (otherwise)
        switch(node->scheme)
        {
        case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[2]);
            node->outStride.push_back(node->outStride[1] * node->length[0]);
            node->oDist = node->outStride[2] * node->length[1];
            break;
        }
        case CS_KERNEL_STOCKHAM:
        {
            node->outStride = node->inStride;
            node->oDist     = node->iDist;
            node->AssignParams();
            break;
        }
        case CS_KERNEL_TRANSPOSE_XY_Z:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->outStride[0] * node->length[2]);
            node->outStride.push_back(node->outStride[1] * node->length[0]);
            node->oDist = node->iDist;
            break;
        }
        default:
            // build_CS_3D_BLOCK_RC should not have created any other node types
            throw std::runtime_error("Scheme Assertion Failed, unexpected node scheme.");
            assert(false);
        }
        prev_outStride = node->outStride;
        prev_oDist     = node->oDist;
    }
    childNodes.back()->outStride = outStride;
    childNodes.back()->oDist     = oDist;
}

void BLOCKRC3DNode::AssignBuffers_internal(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    for(size_t i = 0; i < childNodes.size(); ++i)
    {
        auto& node = childNodes[i];
        node->SetInputBuffer(state);
        node->inArrayType = (i == 0) ? inArrayType : childNodes[i - 1]->outArrayType;
        node->obOut       = flipOut == OB_USER_OUT && placement == rocfft_placement_notinplace
                                ? OB_USER_IN
                                : flipOut;

        // temp is interleaved, in/out might not be
        switch(node->obOut)
        {
        case OB_USER_IN:
            node->outArrayType = inArrayType;
            break;
        case OB_USER_OUT:
            node->outArrayType = outArrayType;
            break;
        default:
            node->outArrayType = rocfft_array_type_complex_interleaved;
        }

        node->AssignBuffers(state, flipIn, flipOut, obOutBuf);
    }

    obOut                           = obOutBuf;
    childNodes.back()->obOut        = obOut;
    childNodes.back()->outArrayType = outArrayType;
}

/*****************************************************
 * CS_3D_RC  *
 *****************************************************/
void RC3DNode::BuildTree_internal()
{
    // 2d fft
    NodeMetaData xyPlanData(this);
    xyPlanData.length.push_back(length[0]);
    xyPlanData.length.push_back(length[1]);
    xyPlanData.dimension = 2;
    xyPlanData.length.push_back(length[2]);
    for(size_t index = 3; index < length.size(); index++)
    {
        xyPlanData.length.push_back(length[index]);
    }
    auto xyPlan = NodeFactory::CreateExplicitNode(xyPlanData, this);
    xyPlan->RecursiveBuildTree();

    // z col fft
    auto zPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
    // make this always inplace, and let the previous one follow the root's placement
    zPlan->placement = rocfft_placement_inplace;
    zPlan->length.push_back(length[2]);
    zPlan->dimension = 1;
    zPlan->length.push_back(length[0]);
    zPlan->length.push_back(length[1]);
    for(size_t index = 3; index < length.size(); index++)
    {
        zPlan->length.push_back(length[index]);
    }

    // RC
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(zPlan));
}

void RC3DNode::AssignParams_internal()
{
    auto& xyPlan = childNodes[0];
    auto& zPlan  = childNodes[1];

    // B -> B
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    // B -> B
    assert((zPlan->obOut == OB_USER_OUT) || (zPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (zPlan->obOut == OB_TEMP_BLUESTEIN));
    zPlan->inStride.push_back(inStride[2]);
    zPlan->inStride.push_back(inStride[0]);
    zPlan->inStride.push_back(inStride[1]);
    for(size_t index = 3; index < length.size(); index++)
        zPlan->inStride.push_back(inStride[index]);

    zPlan->iDist = xyPlan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;
}

// Leaf Node
/*****************************************************
 * Base Class of fused SBRC and Transpose
 *****************************************************/
void SBRCTranspose3DNode::KernelCheck()
{
    // check we have the kernel
    // TODO: TILE_UNALIGNED if we have it
    FMKey key = fpkey(length[0], precision, scheme, TILE_ALIGNED);
    if(!function_pool::has_function(key))
    {
        PrintMissingKernelInfo(key);
        throw std::runtime_error("Kernel not found");
        return;
    }

    if(is_diagonal_sbrc_3D_length(length[0]) && is_cube_size(length))
    {
        key = fpkey(length[0], precision, scheme, DIAGONAL);
        if(!function_pool::has_function(key))
        {
            PrintMissingKernelInfo(key);
            throw std::runtime_error("Kernel not found");
            return;
        }
    }
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
 *****************************************************/
void SBRCTransXY_ZNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    GetBlockComputeTable(length[0], bwd, wgs, lds);
    auto transposeType = sbrc_3D_transpose_type(bwd, length[2], length);
    fnPtr = function_pool::get_function(fpkey(length[0], precision, scheme, transposeType));

    // each block handles 'bwd' rows
    gp.b_x   = DivRoundingUp(length[2], bwd) * length[1] * batch;
    gp.tpb_x = wgs;

    return;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
 *****************************************************/
void SBRCTransZ_XYNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    GetBlockComputeTable(length[0], bwd, wgs, lds);
    auto transposeType = sbrc_3D_transpose_type(bwd, length[1] * length[2], length);
    fnPtr = function_pool::get_function(fpkey(length[0], precision, scheme, transposeType));

    // do 'bwd' rows per block
    gp.b_x = std::accumulate(length.begin() + 1, length.end(), batch, std::multiplies<size_t>());
    gp.b_x /= bwd;
    gp.tpb_x = wgs;

    return;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
 *****************************************************/
void RealCmplxTransZ_XYNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    GetBlockComputeTable(length[0], bwd, wgs, lds);
    auto transposeType = sbrc_3D_transpose_type(bwd, length[1] * length[2], length);
    fnPtr = function_pool::get_function(fpkey(length[0], precision, scheme, transposeType));

    // do 'bwd' rows per block
    gp.b_x = std::accumulate(length.begin() + 1, length.end(), batch, std::multiplies<size_t>());
    gp.b_x /= bwd;
    gp.tpb_x = wgs;

    lds_padding = 1; // 1 element padding per row for even-length real2complx usage

    return;
}