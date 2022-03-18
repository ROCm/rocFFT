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
#include "arithmetic.h"
#include "function_pool.h"
#include "node_factory.h"
#include <numeric>

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
    auto trans1Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
    trans1Plan->length = length;
    trans1Plan->SetTransposeOutputLength();
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
    auto trans2Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
    trans2Plan->length = zPlan->length;
    trans2Plan->SetTransposeOutputLength();
    trans2Plan->dimension = 2;

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto RT1
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {xyPlan.get(), trans1Plan.get()});
    if(RT1->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT1));

    auto RT2 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {zPlan.get(), trans2Plan.get()});
    if(RT2->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // Push to child nodes : 3D_RTRT
    // --------------------------------
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(zPlan)); // notice that move() will set zPlan to nullptr
    childNodes.emplace_back(std::move(trans2Plan));

    // NB:
    //    Don't fuse zPlan, trans2Plan to FT_STOCKHAM_WITH_TRANS_Z_XY
    //    because the preceding trans1 is XY_Z:
    //    xyPlan handles the first 2 dimensions without a transpose.
    //    So the XY_Z tranpose arranges the Z dimension to the fastest dim, and
    //    Z_XY puts it back to the expected arrangement for output.
}

void RTRT3DNode::AssignParams_internal()
{
    assert(childNodes.size() == 4);

    // B -> B
    auto& xyPlan     = childNodes[0];
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    // B -> T
    auto& trans1Plan     = childNodes[1];
    trans1Plan->inStride = xyPlan->outStride;
    trans1Plan->iDist    = xyPlan->oDist;

    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[2]);
    trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[1]);
    trans1Plan->oDist = trans1Plan->length[1] * trans1Plan->outStride[2];

    for(size_t index = 3; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    auto& zPlan     = childNodes[2];
    zPlan->inStride = trans1Plan->outStride;
    zPlan->iDist    = trans1Plan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->AssignParams();

    // T -> B
    auto& trans2Plan     = childNodes[3];
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
        auto trans_plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans_plan->length = cur_length;
        trans_plan->SetTransposeOutputLength();
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

    // --------------------------------
    // Fuse Shims
    // T-[RT]-RTR and TRT-[RT]-R
    // --------------------------------
    auto RT1 = NodeFactory::CreateFuseShim(
        FT_STOCKHAM_WITH_TRANS_Z_XY,
        {childNodes[0].get(), childNodes[1].get(), childNodes[2].get()});
    bool RT1Fusable = RT1->IsSchemeFusable();
    if(RT1Fusable)
        fuseShims.emplace_back(std::move(RT1));

    auto RT2 = NodeFactory::CreateFuseShim(
        FT_STOCKHAM_WITH_TRANS_Z_XY,
        {childNodes[2].get(), childNodes[3].get(), childNodes[4].get()});
    bool RT2Fusable = RT2->IsSchemeFusable();
    if(RT2Fusable)
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // If only partial fusions work, try if we could fuse alternatively
    // [TR][TR][TR]
    // --------------------------------
    if(!RT1Fusable)
    {
        auto TR1 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[0].get(), childNodes[1].get()});
        if(TR1->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR1));
    }
    if(!RT1Fusable && !RT2Fusable)
    {
        auto TR2 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[2].get(), childNodes[3].get()});
        if(TR2->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR2));
    }
    if(!RT2Fusable)
    {
        auto TR3 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[4].get(), childNodes[5].get()});
        if(TR3->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR3));
    }
}

void TRTRTR3DNode::AssignParams_internal()
{
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

#if !GENERIC_BUF_ASSIGMENT
void TRTRTR3DNode::AssignBuffers_internal(TraverseState&   state,
                                          OperatingBuffer& flipIn,
                                          OperatingBuffer& flipOut,
                                          OperatingBuffer& obOutBuf)
{
    assert(childNodes.size() == 6);

    auto& T0 = childNodes[0];
    auto& R0 = childNodes[1];
    auto& T1 = childNodes[2];
    auto& R1 = childNodes[3];
    auto& T2 = childNodes[4];
    auto& R2 = childNodes[5];

    T0->SetInputBuffer(state);
    T0->obOut        = flipOut;
    T0->inArrayType  = inArrayType;
    T0->outArrayType = T0->obOut == OB_TEMP ? rocfft_array_type_complex_interleaved : outArrayType;

    R0->SetInputBuffer(state);
    R0->inArrayType  = T0->outArrayType;
    R0->obOut        = flipOut;
    R0->outArrayType = R0->obOut == OB_TEMP ? rocfft_array_type_complex_interleaved : outArrayType;
    R0->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    T1->SetInputBuffer(state);
    T1->inArrayType  = R0->outArrayType;
    T1->obOut        = flipIn;
    T1->outArrayType = T1->obOut == OB_TEMP ? rocfft_array_type_complex_interleaved : outArrayType;

    R1->SetInputBuffer(state);
    R1->inArrayType  = T1->outArrayType;
    R1->obOut        = flipIn;
    R1->outArrayType = R1->obOut == OB_TEMP ? rocfft_array_type_complex_interleaved : outArrayType;
    R1->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    T2->SetInputBuffer(state);
    T2->inArrayType  = R1->outArrayType;
    T2->obOut        = flipOut;
    T2->outArrayType = T2->obOut == OB_TEMP ? rocfft_array_type_complex_interleaved : outArrayType;

    R2->SetInputBuffer(state);
    R2->inArrayType  = T2->outArrayType;
    R2->obOut        = obOut;
    R2->outArrayType = outArrayType;
    R2->AssignBuffers(state, flipIn, flipOut, obOutBuf);
}
#endif

/*****************************************************
 * CS_3D_BLOCK_RC  *
 *****************************************************/
void BLOCKRC3DNode::BuildTree_internal()
{
    std::vector<size_t> cur_length = length;

    // NB:
    //   The idea is to change the SBRC from doing XZ-plane to doing XY-plane
    //   Some problems are still faster with 3D_RC so they never go here
    bool is906    = is_device_gcn_arch(deviceProp, "gfx906");
    bool is908    = is_device_gcn_arch(deviceProp, "gfx908");
    bool is90a    = is_device_gcn_arch(deviceProp, "gfx90a");
    bool isDouble = (precision == rocfft_precision_double);

    // specific flags for lengths
    bool has50           = (length[0] == 50 || length[1] == 50 || length[2] == 50);
    bool has64           = (length[0] == 64 || length[1] == 64 || length[2] == 64);
    bool has100          = (length[0] == 100 || length[1] == 100 || length[2] == 100);
    bool has200          = (length[0] == 200 || length[1] == 200 || length[2] == 200);
    bool hasPow2         = (IsPo2(length[0]) || IsPo2(length[1]) || IsPo2(length[2]));
    bool isDiagonalTrans = is_diagonal_sbrc_3D_length(length[0]) && is_cube_size(length);

    //   none of diagonal is better by Z_XY (every arch)
    //   both 50, 100 are worse by Z_XY (every arch)
    bool use_ZXY_sbrc = (is906 || is908 || is90a) && (!isDiagonalTrans) && (!has50) && (!has100);

    if(use_ZXY_sbrc)
    {
        if(is906)
        {
            // sbrc 64 performs worse by Z_XY on 906
            use_ZXY_sbrc = (has64 == false);
        }
        else // 908, 90a
        {
            // pow-of-2 performs worse by z_xy on 908, 90a
            if(hasPow2)
                use_ZXY_sbrc = false;

            // sbrc_200 (dp) performs worse by z_xy on 90a
            if(is90a && has200 && isDouble)
                use_ZXY_sbrc = false;
        }
    }

    size_t total_sbrc = 0;
    for(int i = 0; i < 3; ++i)
    {
        // If we have an sbrc kernel for this length, use it,
        // otherwise, fall back to row FFT+transpose
        bool have_sbrc = function_pool::has_SBRC_kernel(cur_length.front(), precision);
        // ensure the kernel would be tile-aligned
        if(have_sbrc)
        {
            auto kernel = function_pool::get_kernel(
                fpkey(cur_length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC));

            size_t otherDim = use_ZXY_sbrc ? cur_length[1] : cur_length[2];
            if(otherDim % kernel.transforms_per_block != 0)
                have_sbrc = false;
        }
        if(have_sbrc)
        {
            ++total_sbrc;
        }

        // We are unable to do more than 2 sbrc kernels.  We'd ideally want 3 sbrc, but each needs
        // to be out-of-place:
        //
        // - kernel 1: IN/OUT -> TEMP
        // - kernel 2: TEMP -> OUT
        // - kernel 3: OUT -> ???
        //
        // So we have no way to put the results back into IN.  So
        // limit ourselves to 2 sbrc kernels in that case.
        if(total_sbrc >= 3 && placement == rocfft_placement_inplace)
        {
            have_sbrc = false;
        }

        if(have_sbrc)
        {
            auto sbrcScheme   = (use_ZXY_sbrc) ? CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
                                               : CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            auto sbrc_node    = NodeFactory::CreateNodeFromScheme(sbrcScheme, this);
            sbrc_node->length = cur_length;
            sbrc_node->SetTransposeOutputLength();
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
            auto transScheme = (use_ZXY_sbrc) ? CS_KERNEL_TRANSPOSE_Z_XY : CS_KERNEL_TRANSPOSE_XY_Z;
            auto trans_plan  = NodeFactory::CreateNodeFromScheme(transScheme, this);
            trans_plan->length = cur_length;
            trans_plan->SetTransposeOutputLength();
            trans_plan->dimension = 2;

            // RT
            childNodes.emplace_back(std::move(row_plan));
            childNodes.emplace_back(std::move(trans_plan));
        }

        if(use_ZXY_sbrc)
        {
            std::swap(cur_length[1], cur_length[0]);
            std::swap(cur_length[2], cur_length[1]);
        }
        else
        {
            std::swap(cur_length[2], cur_length[1]);
            std::swap(cur_length[1], cur_length[0]);
        }
    }
}

void BLOCKRC3DNode::AssignParams_internal()
{
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
        case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[1]);
            node->outStride.push_back(node->outStride[1] * node->length[2]);
            node->oDist = node->outStride[2] * node->length[0];
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
        case CS_KERNEL_TRANSPOSE_Z_XY:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->outStride[0] * node->length[1]);
            node->outStride.push_back(node->outStride[1] * node->length[2]);
            node->oDist = node->iDist;
            break;
        }
        default:
            // build_CS_3D_BLOCK_RC should not have created any other node types
            throw std::runtime_error("Scheme Assertion Failed, unexpected node scheme.");
        }
        prev_outStride = node->outStride;
        prev_oDist     = node->oDist;
    }
    childNodes.back()->outStride = outStride;
    childNodes.back()->oDist     = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
void BLOCKRC3DNode::AssignBuffers_internal(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    auto& RT0 = childNodes[0];
    auto& R1  = childNodes[1];
    auto& RT2 = childNodes[2];
    auto& RT3 = childNodes[3];

    RT0->SetInputBuffer(state);
    RT0->obOut = flipOut;
    RT0->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    R1->SetInputBuffer(state);
    R1->obOut = obOut == flipIn ? flipIn : flipOut;
    R1->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    RT2->SetInputBuffer(state);
    RT2->obOut = RT2->obIn == flipIn ? flipOut : flipIn;
    RT2->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    RT3->SetInputBuffer(state);
    RT3->obOut = obOut;
    RT3->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    for(size_t i = 0; i < childNodes.size(); ++i)
    {
        auto& node = childNodes[i];
        // temp is interleaved, out might not be
        switch(node->obOut)
        {
        case OB_USER_OUT:
            node->outArrayType = outArrayType;
            break;
        case OB_TEMP:
        case OB_TEMP_CMPLX_FOR_REAL:
        case OB_TEMP_BLUESTEIN:

            node->outArrayType = rocfft_array_type_complex_interleaved;
            break;
        default:
            throw std::runtime_error("Invalid buffer in BLOCKRC3DNode");
        }
    }
}
#endif

/*****************************************************
 * CS_3D_BLOCK_CR  *
 *****************************************************/
void BLOCKCR3DNode::BuildTree_internal()
{
    // TODO: It works only for 3 SBCR children nodes for now.
    //       The final logic will be similar to what SBRC has.

    std::vector<size_t> cur_length = length;
    for(int i = 0; i < 3; ++i)
    {
        auto node = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this);
        node->length.push_back(cur_length[2]);
        node->length.push_back(cur_length[0] * cur_length[1]);
        childNodes.emplace_back(std::move(node));
        std::swap(cur_length[1], cur_length[2]);
        std::swap(cur_length[1], cur_length[0]);
    }
}

void BLOCKCR3DNode::AssignParams_internal()
{
    // TODO: It works only for 3 SBCR children nodes for now.
    //       The final logic will be similar to what SBRC has.

    assert(scheme == CS_3D_BLOCK_CR);

    childNodes[0]->inStride.push_back(inStride[2]);
    childNodes[0]->inStride.push_back(inStride[0]);
    childNodes[0]->iDist = iDist;

    childNodes[0]->outStride.push_back(1);
    childNodes[0]->outStride.push_back(childNodes[0]->length[0]);
    childNodes[0]->oDist = childNodes[0]->outStride[1] * childNodes[0]->length[1];

    childNodes[1]->inStride.push_back(childNodes[1]->length[1]);
    childNodes[1]->inStride.push_back(1);
    childNodes[1]->iDist = childNodes[0]->oDist;

    childNodes[1]->outStride.push_back(1);
    childNodes[1]->outStride.push_back(childNodes[1]->length[0]);
    childNodes[1]->oDist = childNodes[1]->outStride[1] * childNodes[1]->length[1];

    childNodes[2]->inStride.push_back(childNodes[2]->length[1]);
    childNodes[2]->inStride.push_back(1);
    childNodes[2]->iDist = childNodes[1]->oDist;

    childNodes[2]->outStride.push_back(outStride[0]);
    childNodes[2]->outStride.push_back(outStride[1]);
    childNodes[2]->oDist = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
void BLOCKCR3DNode::AssignBuffers_internal(TraverseState&   state,
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
#endif

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
    NodeMetaData zPlanData(this);
    zPlanData.length.push_back(length[2]);
    zPlanData.dimension = 1;
    zPlanData.length.push_back(length[0]);
    zPlanData.length.push_back(length[1]);
    for(size_t index = 3; index < length.size(); index++)
    {
        zPlanData.length.push_back(length[index]);
    }

    // use explicit SBCC kernel if available
    std::unique_ptr<TreeNode> zPlan;
    if(function_pool::has_SBCC_kernel(length[2], precision))
    {
        zPlan            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
        zPlan->length    = zPlanData.length;
        zPlan->dimension = 1;
    }
    else
    {
        zPlan = NodeFactory::CreateExplicitNode(zPlanData, this);
        zPlan->RecursiveBuildTree();
    }

    // RC
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(zPlan));
}

void RC3DNode::AssignParams_internal()
{
    auto& xyPlan = childNodes[0];
    auto& zPlan  = childNodes[1];

    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    zPlan->inStride.push_back(inStride[2]);
    zPlan->inStride.push_back(inStride[0]);
    zPlan->inStride.push_back(inStride[1]);
    for(size_t index = 3; index < length.size(); index++)
        zPlan->inStride.push_back(inStride[index]);

    zPlan->iDist = xyPlan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;
    zPlan->AssignParams();
}

// Leaf Node
/*****************************************************
 * Base Class of fused SBRC and Transpose
 *****************************************************/
bool SBRCTranspose3DNode::KernelCheck()
{
    // check we have the kernel
    // TODO: TILE_UNALIGNED if we have it
    FMKey key = fpkey(length[0], precision, scheme, TILE_ALIGNED);
    if(!function_pool::has_function(key))
    {
        PrintMissingKernelInfo(key);
        return false;
    }

    if(is_diagonal_sbrc_3D_length(length[0]) && is_cube_size(length))
    {
        key = fpkey(length[0], precision, scheme, DIAGONAL);
        if(!function_pool::has_function(key))
        {
            PrintMissingKernelInfo(key);
            return false;
        }
    }

    return true;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
 *****************************************************/
void SBRCTransXY_ZNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel
        = function_pool::get_kernel(fpkey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
    bwd           = kernel.transforms_per_block;
    wgs           = kernel.workgroup_size;
    lds           = length[0] * bwd;
    sbrcTranstype = sbrc_transpose_type(bwd);
    fnPtr         = function_pool::get_function(fpkey(length[0], precision, scheme, sbrcTranstype));
    gp.b_x        = DivRoundingUp(length[2], bwd) * length[1] * batch;
    gp.wgs_x      = kernel.workgroup_size;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
 *****************************************************/
void SBRCTransZ_XYNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel
        = function_pool::get_kernel(fpkey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
    bwd           = kernel.transforms_per_block;
    wgs           = kernel.workgroup_size;
    lds           = length[0] * bwd;
    sbrcTranstype = sbrc_transpose_type(bwd);
    fnPtr         = function_pool::get_function(fpkey(length[0], precision, scheme, sbrcTranstype));
    gp.b_x        = DivRoundingUp(length[1], bwd) * length[2] * batch;
    gp.wgs_x      = kernel.workgroup_size;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
 *****************************************************/
void RealCmplxTransZ_XYNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    auto kernel
        = function_pool::get_kernel(fpkey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
    bwd           = kernel.transforms_per_block;
    wgs           = kernel.workgroup_size;
    lds           = length[0] * bwd;
    lds_padding   = 1;
    sbrcTranstype = sbrc_transpose_type(bwd);
    fnPtr         = function_pool::get_function(fpkey(length[0], precision, scheme, sbrcTranstype));
    gp.b_x        = DivRoundingUp(length[1], bwd) * length[2] * batch;
    gp.wgs_x      = kernel.workgroup_size;
}

bool RealCmplxTransZ_XYNode::CreateDevKernelArgs()
{
    // We have a case where this 3D kernel is shoehorned into a 2D plan.
    // If so, add a third dimension when creating kernel args.
    if(length.size() == 2)
    {
        length.push_back(1);
        inStride.push_back(inStride.back());
        outStride.push_back(outStride.back());
    }
    return SBRCTranspose3DNode::CreateDevKernelArgs();
}
