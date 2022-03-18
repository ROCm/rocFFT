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

#include "tree_node_2D.h"
#include "arithmetic.h"
#include "function_pool.h"
#include "fuse_shim.h"
#include "node_factory.h"
#include "repo.h"

/*****************************************************
 * 2D_RTRT  *
 *****************************************************/
void RTRT2DNode::BuildTree_internal()
{
    // first row fft
    NodeMetaData row1PlanData(this);
    row1PlanData.length.push_back(length[0]);
    row1PlanData.dimension = 1;
    row1PlanData.length.push_back(length[1]);
    for(size_t index = 2; index < length.size(); index++)
    {
        row1PlanData.length.push_back(length[index]);
    }
    auto row1Plan = NodeFactory::CreateExplicitNode(row1PlanData, this);
    row1Plan->RecursiveBuildTree();

    // first transpose
    auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans1Plan->length.push_back(length[0]);
    trans1Plan->length.push_back(length[1]);
    trans1Plan->dimension = 2;
    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }
    trans1Plan->SetTransposeOutputLength();

    // second row fft
    NodeMetaData row2PlanData(this);
    row2PlanData.length.push_back(length[1]);
    row2PlanData.dimension = 1;
    row2PlanData.length.push_back(length[0]);
    for(size_t index = 2; index < length.size(); index++)
    {
        row2PlanData.length.push_back(length[index]);
    }
    auto row2Plan = NodeFactory::CreateExplicitNode(row2PlanData, this);
    row2Plan->RecursiveBuildTree();

    // second transpose
    auto trans2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans2Plan->length.push_back(length[1]);
    trans2Plan->length.push_back(length[0]);
    trans2Plan->dimension = 2;
    for(size_t index = 2; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }
    trans2Plan->SetTransposeOutputLength();

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto RT1
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row1Plan.get(), trans1Plan.get()});
    if(RT1->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT1));

    auto RT2
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row2Plan.get(), trans2Plan.get()});
    if(RT2->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // RTRT
    // --------------------------------
    childNodes.emplace_back(std::move(row1Plan));
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(row2Plan));
    childNodes.emplace_back(std::move(trans2Plan));
}

void RTRT2DNode::AssignParams_internal()
{
    auto& row1Plan      = childNodes[0];
    row1Plan->inStride  = inStride;
    row1Plan->iDist     = iDist;
    row1Plan->outStride = outStride;
    row1Plan->oDist     = oDist;
    row1Plan->AssignParams();

    auto& trans1Plan     = childNodes[1];
    trans1Plan->inStride = row1Plan->outStride;
    trans1Plan->iDist    = row1Plan->oDist;
    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[1]);
    trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    auto& row2Plan      = childNodes[2];
    row2Plan->inStride  = trans1Plan->outStride;
    row2Plan->iDist     = trans1Plan->oDist;
    row2Plan->outStride = row2Plan->inStride;
    row2Plan->oDist     = row2Plan->iDist;
    row2Plan->AssignParams();

    auto& trans2Plan      = childNodes[3];
    trans2Plan->inStride  = row2Plan->outStride;
    trans2Plan->iDist     = row2Plan->oDist;
    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
// 3D RTRT calls this as well
void RTRT2DNode::AssignBuffers_internal(TraverseState&   state,
                                        OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        obOut = OB_USER_OUT;
    }

    // Copy the flip buffers, which are swapped by recursive calls.
    auto flipIn0  = flipIn;
    auto flipOut0 = flipOut;

    // Transform:
    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut = flipIn;
    childNodes[0]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    // Transpose:
    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = flipOut0;

    // Stockham:
    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut = flipOut0;
    childNodes[2]->AssignBuffers(state, flipOut0, flipIn0, obOutBuf);

    // Transpose:
    childNodes[3]->SetInputBuffer(state);
    childNodes[3]->obOut = obOut;

    // Transposes must be out-of-place:
    assert(childNodes[1]->obIn != childNodes[1]->obOut);
    assert(childNodes[3]->obIn != childNodes[3]->obOut);
}
#endif

/*****************************************************
 * 2D_RC  *
 *****************************************************/
void RC2DNode::BuildTree_internal()
{
    // row fft
    NodeMetaData rowPlanData(this);
    rowPlanData.length.push_back(length[0]);
    rowPlanData.dimension = 1;
    rowPlanData.length.push_back(length[1]);
    for(size_t index = 2; index < length.size(); index++)
    {
        rowPlanData.length.push_back(length[index]);
    }
    auto rowPlan = NodeFactory::CreateExplicitNode(rowPlanData, this);
    rowPlan->RecursiveBuildTree();

    // column fft
    auto colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
    colPlan->length.push_back(length[1]);
    colPlan->dimension = 1;
    colPlan->length.push_back(length[0]);
    colPlan->large1D = 0; // No twiddle factor in sbcc kernel
    for(size_t index = 2; index < length.size(); index++)
    {
        colPlan->length.push_back(length[index]);
    }

    // RC
    childNodes.emplace_back(std::move(rowPlan));
    childNodes.emplace_back(std::move(colPlan));
}

void RC2DNode::AssignParams_internal()
{
    auto& rowPlan = childNodes[0];
    auto& colPlan = childNodes[1];

    rowPlan->inStride = inStride;
    rowPlan->iDist    = iDist;

    // row plan is in-place, so keep same strides in case parent's
    // in/out strides are incompatible for the same buffer
    rowPlan->outStride = inStride;
    rowPlan->oDist     = iDist;
    rowPlan->AssignParams();

    colPlan->inStride = rowPlan->outStride;
    std::swap(colPlan->inStride[0], colPlan->inStride[1]);

    colPlan->iDist = rowPlan->oDist;

    colPlan->outStride = outStride;
    std::swap(colPlan->outStride[0], colPlan->outStride[1]);
    colPlan->oDist = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
// 3D RC calls this as well...
void RC2DNode::AssignBuffers_internal(TraverseState&   state,
                                      OperatingBuffer& flipIn,
                                      OperatingBuffer& flipOut,
                                      OperatingBuffer& obOutBuf)
{
    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut = flipOut;
    childNodes[0]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = obOutBuf;
    childNodes[1]->AssignBuffers(state, flipOut, flipIn, obOutBuf);

    obIn  = childNodes[0]->obIn;
    obOut = childNodes[1]->obOut;
}
#endif

// Leaf Node..
/*****************************************************
 * CS_KERNEL_2D_SINGLE  *
 *****************************************************/
bool Single2DNode::CreateTwiddleTableResource()
{
    // create one set of twiddles for each dimension
    std::tie(twiddles, twiddles_size) = Repo::GetTwiddles2D(length[0], length[1], precision);

    return CreateLargeTwdTable();
}

void Single2DNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel = function_pool::get_kernel(fpkey(length[0], length[1], precision));
    fnPtr       = kernel.device_function;
    bwd         = kernel.transforms_per_block;

    gp.b_x   = (batch + bwd - 1) / bwd;
    gp.wgs_x = kernel.workgroup_size;

    // if fastest length is power of 2, pad it to avoid LDS bank conflicts
    auto padded_len0 = IsPo2(length[0]) ? length[0] + 1 : length[0];
    lds              = padded_len0 * length[1] * bwd;

    // if we're doing 3D transform, we need to repeat the 2D
    // transform in the 3rd dimension
    if(length.size() > 2)
        gp.b_x *= length[2];

    return;
}
