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

#include "tree_node_real.h"
#include "function_pool.h"
#include "node_factory.h"
#include "radix_table.h"
#include "real2complex.h"

// check if we have an SBCC kernel along the specified dimension
static bool SBCC_dim_available(const std::vector<size_t>& length,
                               size_t                     sbcc_dim,
                               rocfft_precision           precision)
{
    // Check the C part.
    // The first R is built recursively with 2D_FFT, leave the check part to themselves
    size_t numTrans = 0;
    // do we have a purpose-built sbcc kernel
    bool have_sbcc   = false;
    bool is_old_sbcc = false;
    try
    {
        numTrans = function_pool::get_kernel(
                       fpkey(length[sbcc_dim], precision, CS_KERNEL_STOCKHAM_BLOCK_CC))
                       .batches_per_block;
        have_sbcc = true;
        if(!numTrans)
        {
            // this is old sbcc kernel...
            is_old_sbcc = true;
            size_t wgs, lds;
            GetBlockComputeTable(length[sbcc_dim], numTrans, wgs, lds);
        }
    }
    catch(std::out_of_range&)
    {
        try
        {
            numTrans
                = function_pool::get_kernel(fpkey(length[sbcc_dim], precision)).batches_per_block;

            // if we have a size for this transform but numTrans, this must
            // be an old-generator kernel
            if(!numTrans)
            {
                size_t wgs = 0;
                DetermineSizes(length[sbcc_dim], wgs, numTrans);
            }
        }
        catch(std::out_of_range&)
        {
            return false;
        }
    }

    // dim should be >= the blockwidth, or it might perform worse..
    // for old sbcc, must be divisable by BWD(numTrans)
    if(is_old_sbcc)
    {
        size_t remainLen0 = (length[0] / 2 + 1);
        if(remainLen0 % numTrans != 0)
            return false;
    }
    else if(length[0] < numTrans)
        return false;

    // for regular stockham kernels, ensure we are doing enough rows
    // to coalesce properly. 4 seems to be enough for
    // double-precision, whereas some sizes that do 7 rows seem to be
    // slower for single.
    if(!have_sbcc)
    {
        size_t minRows = precision == rocfft_precision_single ? 8 : 4;
        if(numTrans < minRows)
            return false;
    }

    return true;
}

/*****************************************************
 * CS_REAL_TRANSFORM_USING_CMPLX
 *****************************************************/
void RealTransCmplxNode::BuildTree_internal()
{
    // Embed the data into a full-length complex array, perform a
    // complex transform, and then extract the relevant output.
    bool r2c = inArrayType == rocfft_array_type_real;

    auto copyHeadPlan = NodeFactory::CreateNodeFromScheme(
        (r2c ? CS_KERNEL_COPY_R_TO_CMPLX : CS_KERNEL_COPY_HERM_TO_CMPLX), this);
    // head copy plan
    copyHeadPlan->dimension = dimension;
    copyHeadPlan->length    = length;
    childNodes.emplace_back(std::move(copyHeadPlan));

    // complex fft
    NodeMetaData fftPlanData(this);
    fftPlanData.dimension = dimension;
    fftPlanData.length    = length;
    auto fftPlan          = NodeFactory::CreateExplicitNode(fftPlanData, this);
    fftPlan->RecursiveBuildTree();

    // NB:
    //   The tail copy kernel allows only CI type, so the previous kernel should output CI type
    // TODO: make it more elegant..
    //   for example, simply set allowedOutArrayTypes to fftPlan without GetLastLeaf() (propagate)
    //   or add a allowedInArrayType..
    fftPlan->GetLastLeaf()->allowedOutArrayTypes = {rocfft_array_type_complex_interleaved};
    childNodes.emplace_back(std::move(fftPlan));

    // tail copy plan
    auto copyTailPlan = NodeFactory::CreateNodeFromScheme(
        (r2c ? CS_KERNEL_COPY_CMPLX_TO_HERM : CS_KERNEL_COPY_CMPLX_TO_R), this);
    copyTailPlan->dimension = dimension;
    copyTailPlan->length    = length;
    childNodes.emplace_back(std::move(copyTailPlan));
}

void RealTransCmplxNode::AssignParams_internal()
{
    assert(childNodes.size() == 3);
    auto& copyHeadPlan = childNodes[0];
    auto& fftPlan      = childNodes[1];
    auto& copyTailPlan = childNodes[2];

    copyHeadPlan->inStride = inStride;
    copyHeadPlan->iDist    = iDist;

    copyHeadPlan->outStride.push_back(1);
    copyHeadPlan->oDist = copyHeadPlan->length[0];
    for(size_t index = 1; index < length.size(); index++)
    {
        copyHeadPlan->outStride.push_back(copyHeadPlan->oDist);
        copyHeadPlan->oDist *= length[index];
    }

    fftPlan->inStride  = copyHeadPlan->outStride;
    fftPlan->iDist     = copyHeadPlan->oDist;
    fftPlan->outStride = fftPlan->inStride;
    fftPlan->oDist     = fftPlan->iDist;

    fftPlan->AssignParams();

    copyTailPlan->inStride = fftPlan->outStride;
    copyTailPlan->iDist    = fftPlan->oDist;

    copyTailPlan->outStride = outStride;
    copyTailPlan->oDist     = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
void RealTransCmplxNode::AssignBuffers_internal(TraverseState&   state,
                                                OperatingBuffer& flipIn,
                                                OperatingBuffer& flipOut,
                                                OperatingBuffer& obOutBuf)
{
    assert(isRootNode());
    // init flipIn, Out, obOutBuf, this must be a first non-trivial node
    flipIn   = OB_TEMP_CMPLX_FOR_REAL;
    flipOut  = OB_TEMP;
    obOutBuf = OB_TEMP_CMPLX_FOR_REAL;

    assert(childNodes.size() == 3);
    assert((direction == -1 && childNodes[0]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
           || (direction == 1 && childNodes[0]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX));

    obOut = OB_USER_OUT;

    childNodes[0]->SetInputBuffer(state);
    childNodes[0]->obOut        = OB_TEMP_CMPLX_FOR_REAL;
    childNodes[0]->inArrayType  = inArrayType;
    childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut       = flipIn;
    childNodes[1]->inArrayType = rocfft_array_type_complex_interleaved;
    //To check: we might to check childNodes[1]->outArrayType depending on flipIn
    childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
    size_t cs = childNodes[1]->childNodes.size();
    if(cs)
    {
        if(childNodes[1]->scheme == CS_BLUESTEIN)
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_BLUESTEIN);
            assert(childNodes[1]->childNodes[1]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        else
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        assert(childNodes[1]->childNodes[cs - 1]->obOut == OB_TEMP_CMPLX_FOR_REAL);
    }

    assert((direction == -1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
           || (direction == 1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_R));
    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut        = obOut;
    childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
    childNodes[2]->outArrayType = outArrayType;
}
#endif

/*****************************************************
 * CS_REAL_TRANSFORM_EVEN
 *****************************************************/
void RealTransEvenNode::BuildTree_internal()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    // NB:
    // immediate FFT children of CS_REAL_TRANSFORM_EVEN must be
    // in-place because they're working directly on the real buffer,
    // but pretending it's complex

    NodeMetaData cfftPlanData(this);
    cfftPlanData.dimension = dimension;
    cfftPlanData.length    = length;
    cfftPlanData.length[0] = cfftPlanData.length[0] / 2;
    auto cfftPlan          = NodeFactory::CreateExplicitNode(cfftPlanData, this);
    // cfftPlan works in-place on the input buffer for R2C, on the
    // output buffer for C2R
    cfftPlan->allowOutofplace = false; // force it to be inplace
    // NB: the buffer is real, but we treat it as complex
    cfftPlan->RecursiveBuildTree();

    // fuse pre/post-processing into fft if it's single-kernel
    // and **generated with new generator**
    if(try_fuse_pre_post_processing)
    {
        // NOTE: get_kernel won't throw because we would only have a
        // single kernel if it's in the function map.  We're also relying
        // on the fact that old-generator would not populate factors.
        bool singleKernelFFT
            = cfftPlan->isLeafNode()
              && !function_pool::get_kernel(fpkey(cfftPlan->length[0], precision)).factors.empty();
        if(!singleKernelFFT)
            try_fuse_pre_post_processing = false;
    }

    switch(direction)
    {
    case -1:
    {
        // real-to-complex transform: in-place complex transform then post-process

        // insert a node that's prepared to apply the user's
        // callback, since the callback would expect reals and this
        // plan would otherwise pretend it's complex
        auto applyCallback = NodeFactory::CreateNodeFromScheme(CS_KERNEL_APPLY_CALLBACK, this);
        applyCallback->dimension = dimension;
        applyCallback->length    = length;

        if(try_fuse_pre_post_processing)
        {
            cfftPlan->ebtype          = EmbeddedType::Real2C_POST;
            cfftPlan->allowOutofplace = true; // re-enable out-of-place
        }

        childNodes.emplace_back(std::move(applyCallback));
        childNodes.emplace_back(std::move(cfftPlan));

        // add separate post-processing if we couldn't fuse
        if(!try_fuse_pre_post_processing)
        {
            // NB:
            //   input of CS_KERNEL_R_TO_CMPLX allows single-ptr-buffer type only (can't be planar),
            //   so we set the allowed-out-type of the previous kernel to follow the rule.
            //   Precisely, it should be {real, interleaved}, but CI is enough since we only use
            //   CI/CP internally during assign-buffer.
            childNodes.back()->GetLastLeaf()->allowedOutArrayTypes
                = {rocfft_array_type_complex_interleaved};

            auto postPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_R_TO_CMPLX, this);
            postPlan->dimension = 1;
            postPlan->length    = length;
            postPlan->length[0] /= 2;
            childNodes.emplace_back(std::move(postPlan));
        }
        break;
    }
    case 1:
    {
        // complex-to-real transform: pre-process followed by in-place complex transform

        if(!try_fuse_pre_post_processing)
        {
            // add separate pre-processing if we couldn't fuse
            auto prePlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CMPLX_TO_R, this);
            prePlan->dimension = 1;
            prePlan->length    = length;
            prePlan->length[0] /= 2;
            childNodes.emplace_back(std::move(prePlan));
        }
        else
        {
            cfftPlan->ebtype          = EmbeddedType::C2Real_PRE;
            cfftPlan->allowOutofplace = true; // re-enable out-of-place
        }

        // insert a node that's prepared to apply the user's
        // callback, since the callback would expect reals and this
        // plan would otherwise pretend it's complex
        auto applyCallback = NodeFactory::CreateNodeFromScheme(CS_KERNEL_APPLY_CALLBACK, this);
        applyCallback->dimension = dimension;
        applyCallback->length    = length;

        childNodes.emplace_back(std::move(cfftPlan));
        childNodes.emplace_back(std::move(applyCallback));
        break;
    }
    default:
    {
        std::cerr << "invalid direction: plan creation failed!\n";
    }
    }
}

void RealTransEvenNode::AssignParams_internal()
{
    // definitely will have FFT + apply callback.  pre/post processing
    // might be fused into the FFT or separate.
    assert(childNodes.size() == 2 || childNodes.size() == 3);

    if(direction == -1)
    {
        // forward transform, r2c

        // iDist is in reals, subplan->iDist is in complexes
        auto& applyCallback      = childNodes[0];
        applyCallback->inStride  = inStride;
        applyCallback->iDist     = iDist;
        applyCallback->outStride = inStride;
        applyCallback->oDist     = iDist;

        auto& fftPlan     = childNodes[1];
        fftPlan->inStride = inStride;
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            fftPlan->inStride[i] /= 2;
        }
        fftPlan->iDist     = iDist / 2;
        fftPlan->outStride = inStride;
        for(int i = 1; i < fftPlan->outStride.size(); ++i)
        {
            fftPlan->outStride[i] /= 2;
        }
        fftPlan->oDist = iDist / 2;
        fftPlan->AssignParams();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        if(childNodes.size() == 3)
        {
            auto& postPlan = childNodes[2];
            assert(postPlan->scheme == CS_KERNEL_R_TO_CMPLX
                   || postPlan->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE);
            postPlan->inStride = inStride;
            for(int i = 1; i < postPlan->inStride.size(); ++i)
            {
                postPlan->inStride[i] /= 2;
            }
            postPlan->iDist     = iDist / 2;
            postPlan->outStride = outStride;
            postPlan->oDist     = oDist;

            assert(postPlan->length.size() == postPlan->inStride.size());
            assert(postPlan->length.size() == postPlan->outStride.size());
        }
        else
        {
            // we fused post-proc into the FFT kernel, so give the correct out strides
            fftPlan->outStride = outStride;
            fftPlan->oDist     = oDist;
        }
    }
    else
    {
        // backward transform, c2r
        bool fusedPreProcessing = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;

        // oDist is in reals, subplan->oDist is in complexes

        if(!fusedPreProcessing)
        {
            auto& prePlan = childNodes[0];
            assert(prePlan->scheme == CS_KERNEL_CMPLX_TO_R);

            prePlan->iDist = iDist;
            prePlan->oDist = oDist / 2;

            // Strides are actually distances for multimensional transforms.
            // Only the first value is used, but we require dimension values.
            prePlan->inStride  = inStride;
            prePlan->outStride = outStride;
            // Strides are in complex types
            for(int i = 1; i < prePlan->outStride.size(); ++i)
            {
                prePlan->outStride[i] /= 2;
            }
            assert(prePlan->length.size() == prePlan->inStride.size());
            assert(prePlan->length.size() == prePlan->outStride.size());
        }

        auto& fftPlan = fusedPreProcessing ? childNodes[0] : childNodes[1];
        // Transform the strides from real to complex.

        fftPlan->inStride  = fusedPreProcessing ? inStride : outStride;
        fftPlan->iDist     = fusedPreProcessing ? iDist : oDist / 2;
        fftPlan->outStride = outStride;
        fftPlan->oDist     = oDist / 2;
        // The strides must be translated from real to complex.
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            if(!fusedPreProcessing)
                fftPlan->inStride[i] /= 2;
            fftPlan->outStride[i] /= 2;
        }

        fftPlan->AssignParams();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        // we apply callbacks on the root plan's output
        TreeNode* rootPlan = this;
        while(rootPlan->parent != nullptr)
            rootPlan = rootPlan->parent;

        auto& applyCallback      = childNodes.back();
        applyCallback->inStride  = rootPlan->outStride;
        applyCallback->iDist     = rootPlan->oDist;
        applyCallback->outStride = rootPlan->outStride;
        applyCallback->oDist     = rootPlan->oDist;
    }
}

#if !GENERIC_BUF_ASSIGMENT
void RealTransEvenNode::AssignBuffers_internal(TraverseState&   state,
                                               OperatingBuffer& flipIn,
                                               OperatingBuffer& flipOut,
                                               OperatingBuffer& obOutBuf)
{
    // if this is root node, init the flipIn,flipOut and obOutBuf
    if(isRootNode())
    {
        // for R2C transform, output side is complex so we can
        // flip into the output buffer
        // if(direction == -1), we already set [OUT / TEMP / OUT] in AssignBuffers

        // for C2R transform, input side is complex so we can
        // flip into the input buffer
        if(direction == 1)
        {
            flipIn   = placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
            flipOut  = OB_TEMP;
            obOutBuf = OB_USER_OUT;
        }
    }

    auto flipIn0  = flipIn;
    auto flipOut0 = flipOut;

    if(direction == -1)
    {
        // real-to-complex

        flipIn  = obIn;
        flipOut = OB_TEMP;

        // apply callback
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obIn;
        childNodes[0]->inArrayType  = rocfft_array_type_real;
        childNodes[0]->outArrayType = rocfft_array_type_real;

        // we would have only 2 child nodes if we're able to fuse the
        // post-processing into the single FFT kernel
        bool fusedPostProcessing = childNodes[1]->ebtype == EmbeddedType::Real2C_POST;

        // complex FFT kernel
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut       = fusedPostProcessing ? obOut : obIn;
        childNodes[1]->inArrayType = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType
            = fusedPostProcessing ? outArrayType : rocfft_array_type_complex_interleaved;
        childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        if(!fusedPostProcessing)
        {
            // real-to-complex post kernel
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = obOut;
            childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[2]->outArrayType = outArrayType;
        }

        flipIn  = flipIn0;
        flipOut = flipOut0;
    }
    else
    {
        // we would only have 2 child nodes if we fused the
        // pre-processing into the single FFT kernel
        bool fusedPreProcessing = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;

        auto& fftNode = fusedPreProcessing ? childNodes[0] : childNodes[1];

        if(!fusedPreProcessing)
        {
            // complex-to-real

            // complex-to-real pre kernel
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut = obOut;

            childNodes[0]->inArrayType  = inArrayType;
            childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

            // NB: The case here indicates parent's input buffer is not
            //     complex_planar or hermitian_planar, so the child must
            //     be a hermitian_interleaved.
            if(inArrayType == rocfft_array_type_complex_interleaved)
            {
                childNodes[0]->inArrayType = rocfft_array_type_hermitian_interleaved;
            }
        }

        // complex FFT kernel
        fftNode->SetInputBuffer(state);
        fftNode->obOut = obOut;
        fftNode->inArrayType
            = fusedPreProcessing ? inArrayType : rocfft_array_type_complex_interleaved;
        fftNode->outArrayType = rocfft_array_type_complex_interleaved;
        fftNode->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        // apply callback
        childNodes.back()->obIn         = obOut;
        childNodes.back()->obOut        = obOut;
        childNodes.back()->inArrayType  = rocfft_array_type_real;
        childNodes.back()->outArrayType = rocfft_array_type_real;
    }
}
#endif

/*****************************************************
 * CS_REAL_2D_EVEN
 *****************************************************/
void Real2DEvenNode::BuildTree_internal()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    const size_t biggerDim  = std::max(length[0], length[1]);
    const size_t smallerDim = std::min(length[0], length[1]);
    const size_t padding
        = (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512)) ? 64 : 0;

    if(inArrayType == rocfft_array_type_real) //forward
    {
        // RTRT
        // first row fft
        auto row1Plan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        row1Plan->length.push_back(length[0]);
        row1Plan->dimension = 1;
        row1Plan->length.push_back(length[1]);
        for(size_t index = 2; index < length.size(); index++)
        {
            row1Plan->length.push_back(length[index]);
        }
        row1Plan->outputHasPadding = false;
        row1Plan->RecursiveBuildTree();

        // first transpose
        auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans1Plan->length.push_back(length[0] / 2 + 1);
        trans1Plan->length.push_back(length[1]);
        trans1Plan->dimension = 2;
        for(size_t index = 2; index < length.size(); index++)
        {
            trans1Plan->length.push_back(length[index]);
        }
        trans1Plan->outputHasPadding = (padding > 0);

        // second row fft
        NodeMetaData row2PlanData(this);
        row2PlanData.length.push_back(length[1]);
        row2PlanData.dimension = 1;
        row2PlanData.length.push_back(length[0] / 2 + 1);
        for(size_t index = 2; index < length.size(); index++)
        {
            row2PlanData.length.push_back(length[index]);
        }
        auto row2Plan              = NodeFactory::CreateExplicitNode(row2PlanData, this);
        row2Plan->outputHasPadding = trans1Plan->outputHasPadding;
        row2Plan->RecursiveBuildTree();

        // second transpose
        auto trans2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans2Plan->length.push_back(length[1]);
        trans2Plan->length.push_back(length[0] / 2 + 1);
        trans2Plan->dimension = 2;
        for(size_t index = 2; index < length.size(); index++)
        {
            trans2Plan->length.push_back(length[index]);
        }
        trans2Plan->outputHasPadding = this->outputHasPadding;

        // --------------------------------
        // Fuse Shims:
        // 1-1. Try (stockham + r2c)(from real even) + transpose
        // 1-2. else, try r2c (from real even) + transpose
        // 2. row2 and trans2: RTFuse
        // --------------------------------
        auto STK_R2CTrans = NodeFactory::CreateFuseShim(FT_STOCKHAM_R2C_TRANSPOSE,
                                                        {row1Plan.get(), trans1Plan.get()});
        if(STK_R2CTrans->IsSchemeFusable())
        {
            fuseShims.emplace_back(std::move(STK_R2CTrans));
        }
        else
        {
            auto R2CTrans = NodeFactory::CreateFuseShim(
                FT_R2C_TRANSPOSE, {row1Plan.get(), trans1Plan.get(), row2Plan.get()});
            if(R2CTrans->IsSchemeFusable())
                fuseShims.emplace_back(std::move(R2CTrans));
        }

        auto RT = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS,
                                              {row2Plan.get(), trans2Plan.get()});
        if(RT->IsSchemeFusable())
            fuseShims.emplace_back(std::move(RT));

        // --------------------------------
        // RTRT
        // --------------------------------
        // Fuse r2c trans
        childNodes.emplace_back(std::move(row1Plan));
        childNodes.emplace_back(std::move(trans1Plan));
        // Fuse RT
        childNodes.emplace_back(std::move(row2Plan));
        childNodes.emplace_back(std::move(trans2Plan));
    }
    else
    {
        // TRTR
        // first transpose
        auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans1Plan->length.push_back(length[0] / 2 + 1);
        trans1Plan->length.push_back(length[1]);
        trans1Plan->dimension = 2;
        for(size_t index = 2; index < length.size(); index++)
        {
            trans1Plan->length.push_back(length[index]);
        }
        trans1Plan->outputHasPadding = (padding > 0);

        // c2c row transform
        NodeMetaData c2cPlanData(this);
        c2cPlanData.dimension = 1;
        c2cPlanData.length.push_back(length[1]);
        c2cPlanData.length.push_back(length[0] / 2 + 1);
        for(size_t index = 2; index < length.size(); index++)
        {
            c2cPlanData.length.push_back(length[index]);
        }
        auto c2cPlan              = NodeFactory::CreateExplicitNode(c2cPlanData, this);
        c2cPlan->outputHasPadding = trans1Plan->outputHasPadding;
        c2cPlan->RecursiveBuildTree();

        // second transpose
        auto trans2plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans2plan->length.push_back(length[1]);
        trans2plan->length.push_back(length[0] / 2 + 1);
        trans2plan->dimension = 2;
        for(size_t index = 2; index < length.size(); index++)
        {
            trans2plan->length.push_back(length[index]);
        }
        // NOTE
        trans2plan->outputHasPadding = false;

        // c2r row transform
        auto c2rPlan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        c2rPlan->length.push_back(length[0]);
        c2rPlan->length.push_back(length[1]);
        c2rPlan->dimension = 1;
        for(size_t index = 2; index < length.size(); index++)
        {
            c2rPlan->length.push_back(length[index]);
        }
        c2rPlan->outputHasPadding = this->outputHasPadding;
        c2rPlan->RecursiveBuildTree();

        // --------------------------------
        // Fuse Shims:
        // 1. trans1 and c2c
        // 2. transpose + c2r (first child of real even)
        // --------------------------------
        auto TR = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                              {trans1Plan.get(), c2cPlan.get()});
        if(TR->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR));

        auto TransC2R
            = NodeFactory::CreateFuseShim(FT_TRANSPOSE_C2R, {trans2plan.get(), c2rPlan.get()});
        if(TransC2R->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TransC2R));

        // --------------------------------
        // TRTR
        // --------------------------------
        childNodes.emplace_back(std::move(trans1Plan));
        childNodes.emplace_back(std::move(c2cPlan));
        //
        childNodes.emplace_back(std::move(trans2plan));
        childNodes.emplace_back(std::move(c2rPlan));
    }
}

void Real2DEvenNode::AssignParams_internal()
{
    const size_t biggerDim  = std::max(length[0], length[1]);
    const size_t smallerDim = std::min(length[0], length[1]);
    const size_t padding
        = (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512)) ? 64 : 0;

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& row1Plan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            row1Plan->inStride = inStride;
            row1Plan->iDist    = iDist;

            row1Plan->outStride = outStride;
            row1Plan->oDist     = oDist;

            row1Plan->AssignParams();
        }

        auto& trans1Plan = childNodes[1];
        {
            // B -> T
            trans1Plan->inStride = row1Plan->outStride;
            trans1Plan->iDist    = row1Plan->oDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }

        auto& row2Plan = childNodes[2];
        {
            // T -> T
            row2Plan->inStride = trans1Plan->outStride;
            row2Plan->iDist    = trans1Plan->oDist;

            row2Plan->outStride = row2Plan->inStride;
            row2Plan->oDist     = row2Plan->iDist;

            row2Plan->AssignParams();
        }

        auto& trans2Plan = childNodes[3];
        {
            // T -> B
            trans2Plan->inStride = row2Plan->outStride;
            trans2Plan->iDist    = row2Plan->oDist;

            trans2Plan->outStride = outStride;
            trans2Plan->oDist     = oDist;
        }
    }
    else
    {
        auto& trans1Plan = childNodes[0];
        {
            trans1Plan->inStride = inStride;
            trans1Plan->iDist    = iDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }
        auto& c2cPlan = childNodes[1];
        {
            c2cPlan->inStride = trans1Plan->outStride;
            c2cPlan->iDist    = trans1Plan->oDist;

            c2cPlan->outStride = c2cPlan->inStride;
            c2cPlan->oDist     = c2cPlan->iDist;

            c2cPlan->AssignParams();
        }
        auto& trans2Plan = childNodes[2];
        {
            trans2Plan->inStride = trans1Plan->outStride;
            trans2Plan->iDist    = trans1Plan->oDist;

            trans2Plan->outStride = trans1Plan->inStride;
            trans2Plan->oDist     = trans2Plan->length[0] * trans2Plan->outStride[1];
        }
        auto& c2rPlan = childNodes[3];
        {
            c2rPlan->inStride = trans2Plan->outStride;
            c2rPlan->iDist    = trans2Plan->oDist;

            c2rPlan->outStride = outStride;
            c2rPlan->oDist     = oDist;

            c2rPlan->AssignParams();
        }
    }
}

#if !GENERIC_BUF_ASSIGMENT
void Real2DEvenNode::AssignBuffers_internal(TraverseState&   state,
                                            OperatingBuffer& flipIn,
                                            OperatingBuffer& flipOut,
                                            OperatingBuffer& obOutBuf)
{
    assert(isRootNode());
    // if(isRootNode()), init the flipIn,flipOut and obOutBuf
    //    same as what we do in RealTransEvenNode. Besides, this must be a root node
    // (for direction == -1, we already set [OUT / TEMP / OUT])
    if(direction == 1)
    {
        flipIn   = placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
        flipOut  = OB_TEMP;
        obOutBuf = OB_USER_OUT;
    }

    if(direction == -1)
    {
        // RTRT

        // real-to-complex
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obOut;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = outArrayType;
        childNodes[0]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        // T
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut        = flipOut;
        childNodes[1]->inArrayType  = childNodes[0]->outArrayType;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;

        // complex-to-complex
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut = flipOut;
        childNodes[2]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

        // T
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut        = obOut;
        childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[3]->outArrayType = outArrayType;
    }
    else
    { // TRTR

        // The first transform only gets to play with the input buffer.
        auto flipIn0 = flipIn;
        flipIn       = obIn;

        // T
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = flipOut;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

        // complex-to-complex
        childNodes[1]->SetInputBuffer(state);
        childNodes[1]->obOut        = flipOut;
        childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;
        childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        // T
        childNodes[2]->SetInputBuffer(state);
        childNodes[2]->obOut        = obIn;
        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = inArrayType;

        // complex-to-real
        childNodes[3]->SetInputBuffer(state);
        childNodes[3]->obOut        = obOut;
        childNodes[3]->inArrayType  = inArrayType;
        childNodes[3]->outArrayType = rocfft_array_type_real;

        flipIn = flipIn0;

        childNodes[3]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
    }
}
#endif

/*****************************************************
 * CS_REAL_3D_EVEN
 *****************************************************/
void Real3DEvenNode::BuildTree_internal()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes.
    bool sbcc_inplace
        = SBCC_dim_available(length, 1, precision) && SBCC_dim_available(length, 2, precision);

#if 0
    // ensure the fastest dimensions are big enough to get enough
    // column tiles to perform well
    if(length[0] <= 52 || length[1] <= 52)
        sbcc_inplace = false;
    // also exclude particular problematic sizes for higher dims
    if(length[1] == 168 || length[2] == 168)
        sbcc_inplace = false;
    // if all 3 lengths are SBRC-able, then R2C will already be 3
    // kernel.  SBRC should be slightly better since row accesses
    // should be a bit nicer in general than column accesses.
    if(function_pool::has_SBRC_kernel(length[0] / 2, precision)
       && function_pool::has_SBRC_kernel(length[1], precision)
       && function_pool::has_SBRC_kernel(length[2], precision))
    {
        sbcc_inplace = false;
    }
#endif

    auto add_sbcc_children = [this](const std::vector<size_t>& remainingLength) {
        ComputeScheme scheme;
        // SBCC along Z dimension
        bool haveSBCC_Z = function_pool::has_SBCC_kernel(remainingLength[2], precision);
        scheme          = haveSBCC_Z ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        auto sbccZ      = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccZ->length   = remainingLength;
        std::swap(sbccZ->length[1], sbccZ->length[2]);
        std::swap(sbccZ->length[0], sbccZ->length[1]);
        childNodes.emplace_back(std::move(sbccZ));

        // SBCC along Y dimension
        bool haveSBCC_Y = function_pool::has_SBCC_kernel(remainingLength[1], precision);
        scheme          = haveSBCC_Y ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        auto sbccY      = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccY->length   = remainingLength;
        std::swap(sbccY->length[0], sbccY->length[1]);
        childNodes.emplace_back(std::move(sbccY));
    };

    std::vector<size_t> remainingLength = {length[0] / 2 + 1, length[1], length[2]};

    if(inArrayType == rocfft_array_type_real) // forward
    {
        // first row fft + postproc is mandatory for fastest dimension
        auto rcplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        static_cast<RealTransEvenNode*>(rcplan.get())->try_fuse_pre_post_processing = sbcc_inplace;

        rcplan->length    = length;
        rcplan->dimension = 1;
        rcplan->RecursiveBuildTree();

        // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes
        if(sbcc_inplace)
        {
            childNodes.emplace_back(std::move(rcplan));
            add_sbcc_children(remainingLength);
        }
        // otherwise, handle remaining dimensions with TRTRT
        else
        {
            // first transpose
            auto trans1       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
            trans1->length    = remainingLength;
            trans1->dimension = 2;

            // first column
            NodeMetaData c1planData(this);
            c1planData.length       = {trans1->length[1], trans1->length[2], trans1->length[0]};
            c1planData.dimension    = 1;
            auto c1plan             = NodeFactory::CreateExplicitNode(c1planData, this);
            c1plan->allowOutofplace = false; // let it be inplace
            c1plan->RecursiveBuildTree();

            // second transpose
            auto trans2       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
            trans2->length    = c1plan->length;
            trans2->dimension = 2;

            // second column
            NodeMetaData c2planData(this);
            c2planData.length       = {trans2->length[1], trans2->length[2], trans2->length[0]};
            c2planData.dimension    = 1;
            auto c2plan             = NodeFactory::CreateExplicitNode(c2planData, this);
            c2plan->allowOutofplace = false; // let it be inplace
            c2plan->RecursiveBuildTree();

            // third transpose
            auto trans3       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
            trans3->length    = c2plan->length;
            trans3->dimension = 2;

            // --------------------------------
            // Fuse Shims: [RealEven + T][RT][RT]
            // 1-1. Try (stockham + r2c)(from real even) + transp
            // 1-2. else, try r2c (from real even) + transp
            // 2. RT1 = trans1 check + c1plan + trans2
            // 3. RT2 = trans2 check + c2plan + trans3
            // --------------------------------
            auto STK_R2CTrans = NodeFactory::CreateFuseShim(FT_STOCKHAM_R2C_TRANSPOSE,
                                                            {rcplan.get(), trans1.get()});
            if(STK_R2CTrans->IsSchemeFusable())
            {
                fuseShims.emplace_back(std::move(STK_R2CTrans));
            }
            else
            {
                auto R2CTrans = NodeFactory::CreateFuseShim(
                    FT_R2C_TRANSPOSE, {rcplan.get(), trans1.get(), c1plan.get()});
                if(R2CTrans->IsSchemeFusable())
                    fuseShims.emplace_back(std::move(R2CTrans));
            }

            auto RT1 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_Z_XY,
                                                   {trans1.get(), c1plan.get(), trans2.get()});
            if(RT1->IsSchemeFusable())
            {
                fuseShims.emplace_back(std::move(RT1));
            }
            else
            {
                auto RTStride1 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS,
                                                             {c1plan.get(), trans2.get()});
                if(RTStride1->IsSchemeFusable())
                    fuseShims.emplace_back(std::move(RTStride1));
            }

            auto RT2 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_Z_XY,
                                                   {trans2.get(), c2plan.get(), trans3.get()});
            if(RT2->IsSchemeFusable())
            {
                fuseShims.emplace_back(std::move(RT2));
            }
            else
            {
                auto RTStride2 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS,
                                                             {c2plan.get(), trans3.get()});
                if(RTStride2->IsSchemeFusable())
                    fuseShims.emplace_back(std::move(RTStride2));
            }

            // --------------------------------
            // 1DEven + TRTRT
            // --------------------------------
            childNodes.emplace_back(std::move(rcplan));
            childNodes.emplace_back(std::move(trans1));
            // Fuse R + TRANSPOSE_Z_XY
            childNodes.emplace_back(std::move(c1plan));
            childNodes.emplace_back(std::move(trans2));
            // Fuse R + TRANSPOSE_Z_XY
            childNodes.emplace_back(std::move(c2plan));
            childNodes.emplace_back(std::move(trans3));
        }
    }
    else
    {
        if(sbcc_inplace)
        {
            add_sbcc_children(remainingLength);
        }
        // otherwise, TRTRTR
        else
        {
            // transpose
            auto trans3       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
            trans3->length    = {length[0] / 2 + 1, length[1], length[2]};
            trans3->dimension = 2;

            // column
            NodeMetaData c2planData(this);
            c2planData.length       = {trans3->length[2], trans3->length[0], trans3->length[1]};
            c2planData.dimension    = 1;
            auto c2plan             = NodeFactory::CreateExplicitNode(c2planData, this);
            c2plan->allowOutofplace = false; // let it be inplace
            c2plan->RecursiveBuildTree();

            // transpose
            auto trans2       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
            trans2->length    = c2plan->length;
            trans2->dimension = 2;

            // column
            NodeMetaData c1planData(this);
            c1planData.length       = {trans2->length[2], trans2->length[0], trans2->length[1]};
            c1planData.dimension    = 1;
            auto c1plan             = NodeFactory::CreateExplicitNode(c1planData, this);
            c1plan->allowOutofplace = false; // let it be inplace
            c1plan->RecursiveBuildTree();

            // transpose
            auto trans1       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
            trans1->length    = c1plan->length;
            trans1->dimension = 2;

            // --------------------------------
            // Fuse Shims:
            // 1. RT = c2plan + trans2 + c1plan(check-stockham)
            // --------------------------------
            auto RT = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_XY_Z,
                                                  {c2plan.get(), trans2.get(), c1plan.get()});
            if(RT->IsSchemeFusable())
                fuseShims.emplace_back(std::move(RT));

            // --------------------------------
            // TRTRT + 1DEven
            // TODO- eventually we should fuse two TR (TRANSPOSE_XY_Z_STOCKHAM)
            // --------------------------------
            childNodes.emplace_back(std::move(trans3));
            // Fuse R + TRANSPOSE_XY_Z
            childNodes.emplace_back(std::move(c2plan));
            childNodes.emplace_back(std::move(trans2));
            childNodes.emplace_back(std::move(c1plan));
            // Fuse this trans and pre-kernel-c2r of 1D-even
            childNodes.emplace_back(std::move(trans1));
        }

        // c2r
        auto crplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        static_cast<RealTransEvenNode*>(crplan.get())->try_fuse_pre_post_processing = sbcc_inplace;

        crplan->length    = length;
        crplan->dimension = 1;
        crplan->RecursiveBuildTree();
        childNodes.emplace_back(std::move(crplan));

        // --------------------------------
        // Fuse Shims:
        // 2. trans1 + c2r (first child of real even)
        // note the CheckSchemeFusable will check if the first one is transpose
        // --------------------------------
        auto TransC2R = NodeFactory::CreateFuseShim(
            FT_TRANSPOSE_C2R, {childNodes[childNodes.size() - 2].get(), childNodes.back().get()});
        if(TransC2R->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TransC2R));
    }
}

void Real3DEvenNode::AssignParams_internal()
{
    // TODO: add padding?

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& rcplan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            rcplan->inStride  = inStride;
            rcplan->iDist     = iDist;
            rcplan->outStride = outStride;
            rcplan->oDist     = oDist;
            rcplan->dimension = 1;
            rcplan->AssignParams();
        }

        // in-place SBCC for higher dims
        if(childNodes.size() == 3)
        {
            auto& sbccZ     = childNodes[1];
            sbccZ->inStride = outStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = oDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = oDist;
            sbccZ->AssignParams();

            auto& sbccY     = childNodes[2];
            sbccY->inStride = outStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = oDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = oDist;
            sbccY->AssignParams();
        }
        // TRTRTR
        else
        {
            auto& trans1 = childNodes[1];
            {
                trans1->inStride = rcplan->outStride;
                trans1->iDist    = rcplan->oDist;
                trans1->outStride.push_back(1);
                trans1->outStride.push_back(trans1->length[1]);
                trans1->outStride.push_back(trans1->length[2] * trans1->outStride[1]);
                trans1->oDist = trans1->iDist;
            }

            auto& c1plan = childNodes[2];
            {
                c1plan->inStride  = trans1->outStride;
                c1plan->iDist     = trans1->oDist;
                c1plan->outStride = c1plan->inStride;
                c1plan->oDist     = c1plan->iDist;
                c1plan->dimension = 1;
                c1plan->AssignParams();
            }

            auto& trans2 = childNodes[3];
            {
                trans2->inStride = c1plan->outStride;
                trans2->iDist    = c1plan->oDist;
                trans2->outStride.push_back(1);
                trans2->outStride.push_back(trans2->length[1]);
                trans2->outStride.push_back(trans2->length[2] * trans2->outStride[1]);
                trans2->oDist = trans2->iDist;
            }

            auto& c2plan = childNodes[4];
            {
                c2plan->inStride  = trans2->outStride;
                c2plan->iDist     = trans2->oDist;
                c2plan->outStride = c2plan->inStride;
                c2plan->oDist     = c2plan->iDist;
                c2plan->dimension = 1;
                c2plan->AssignParams();
            }

            auto& trans3 = childNodes[5];
            {
                trans3->inStride  = c2plan->outStride;
                trans3->iDist     = c2plan->oDist;
                trans3->outStride = outStride;
                trans3->oDist     = oDist;
            }
        }
    }
    else
    {
        // input strides for last c2r node
        std::vector<size_t> c2r_inStride = inStride;
        size_t              c2r_iDist    = iDist;

        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            auto& sbccZ     = childNodes[0];
            sbccZ->inStride = inStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = iDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = iDist;
            sbccZ->AssignParams();

            auto& sbccY     = childNodes[1];
            sbccY->inStride = inStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = iDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = iDist;
            sbccY->AssignParams();
        }
        // RTRTRT
        else
        {
            {
                auto& trans3     = childNodes[0];
                trans3->inStride = inStride;
                trans3->iDist    = iDist;
                trans3->outStride.push_back(1);
                trans3->outStride.push_back(trans3->outStride[0] * trans3->length[2]);
                trans3->outStride.push_back(trans3->outStride[1] * trans3->length[0]);
                trans3->oDist = trans3->iDist;
            }

            {
                auto& ccplan      = childNodes[1];
                ccplan->inStride  = childNodes[0]->outStride;
                ccplan->iDist     = childNodes[0]->oDist;
                ccplan->outStride = ccplan->inStride;
                ccplan->oDist     = ccplan->iDist;
                ccplan->dimension = 1;
                ccplan->AssignParams();
            }

            {
                auto& trans2     = childNodes[2];
                trans2->inStride = childNodes[1]->outStride;
                trans2->iDist    = childNodes[1]->oDist;
                trans2->outStride.push_back(1);
                trans2->outStride.push_back(trans2->outStride[0] * trans2->length[2]);
                trans2->outStride.push_back(trans2->outStride[1] * trans2->length[0]);
                trans2->oDist = trans2->iDist;
            }

            {
                auto& ccplan      = childNodes[3];
                ccplan->inStride  = childNodes[2]->outStride;
                ccplan->iDist     = childNodes[2]->oDist;
                ccplan->outStride = ccplan->inStride;
                ccplan->oDist     = ccplan->iDist;
                ccplan->dimension = 1;
                ccplan->AssignParams();
            }

            {
                auto& trans1     = childNodes[4];
                trans1->inStride = childNodes[3]->outStride;
                trans1->iDist    = childNodes[3]->oDist;
                trans1->outStride.push_back(1);
                trans1->outStride.push_back(trans1->outStride[0] * trans1->length[2]);
                trans1->outStride.push_back(trans1->outStride[1] * trans1->length[0]);
                trans1->oDist = trans1->iDist;
                c2r_inStride  = trans1->outStride;
                c2r_iDist     = trans1->oDist;
            }
        }

        auto& crplan = childNodes.back();
        {
            crplan->inStride  = c2r_inStride;
            crplan->iDist     = c2r_iDist;
            crplan->outStride = outStride;
            crplan->oDist     = oDist;
            crplan->dimension = 1;
            crplan->AssignParams();
        }
    }
}

#if !GENERIC_BUF_ASSIGMENT
void Real3DEvenNode::AssignBuffers_internal(TraverseState&   state,
                                            OperatingBuffer& flipIn,
                                            OperatingBuffer& flipOut,
                                            OperatingBuffer& obOutBuf)
{
    assert(isRootNode());
    // if(isRootNode()), init the flipIn,flipOut and obOutBuf
    //    same as what we do in RealTransEvenNode. Besides, this must be a root node
    // (for direction == -1, we already set [OUT / TEMP / OUT])
    if(direction == 1)
    {
        flipIn   = placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
        flipOut  = OB_TEMP;
        obOutBuf = OB_USER_OUT;
    }

    obOut = OB_USER_OUT;
    if(direction == -1)
    {
        flipIn  = obIn;
        flipOut = OB_TEMP;

        // R: r2c
        childNodes[0]->SetInputBuffer(state);
        childNodes[0]->obOut        = obOutBuf;
        childNodes[0]->inArrayType  = inArrayType;
        childNodes[0]->outArrayType = outArrayType;
        childNodes[0]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        flipIn   = OB_TEMP;
        flipOut  = obOut;
        obOutBuf = obOut;

        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = childNodes[1]->obIn;
            childNodes[1]->inArrayType  = outArrayType;
            childNodes[1]->outArrayType = outArrayType;

            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = childNodes[2]->obIn;
            childNodes[2]->inArrayType  = outArrayType;
            childNodes[2]->outArrayType = outArrayType;
        }
        // RTRTRT
        else if(childNodes.size() == 6)
        {
            // NB: for out-of-place transforms, we can't fit the result of the first r2c transform
            // into the input buffer.

            // T
            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = (childNodes[1]->obIn == flipIn) ? flipOut : flipIn;
            childNodes[1]->inArrayType  = childNodes[0]->outArrayType;
            childNodes[1]->outArrayType = outArrayType;

            // R: c2c
            childNodes[2]->inArrayType  = childNodes[1]->outArrayType;
            childNodes[2]->outArrayType = outArrayType;
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut = obOutBuf;
            childNodes[2]->AssignBuffers(state, flipOut, flipIn, obIn);

            // T
            childNodes[3]->SetInputBuffer(state);
            childNodes[3]->obOut        = (childNodes[3]->obIn == flipIn) ? flipOut : flipIn;
            childNodes[3]->inArrayType  = childNodes[2]->outArrayType;
            childNodes[3]->outArrayType = (childNodes[3]->obOut == OB_TEMP)
                                              ? rocfft_array_type_complex_interleaved
                                              : outArrayType;

            // R: c2c
            childNodes[4]->SetInputBuffer(state);
            childNodes[4]->obOut = (obOutBuf == flipIn) ? flipOut : flipIn;
            childNodes[4]->AssignBuffers(state, flipIn, flipOut, obOutBuf);
            childNodes[4]->inArrayType  = childNodes[3]->outArrayType;
            childNodes[4]->outArrayType = (childNodes[4]->obOut == OB_TEMP)
                                              ? rocfft_array_type_complex_interleaved
                                              : outArrayType;

            // T
            childNodes[5]->inArrayType  = childNodes[4]->outArrayType;
            childNodes[5]->outArrayType = outArrayType;
            childNodes[5]->SetInputBuffer(state);
            childNodes[5]->obOut = obOutBuf;
        }
    }
    else
    {
        // in-place SBCC for higher dimensions
        if(childNodes.size() == 3)
        {
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut        = childNodes[0]->obIn;
            childNodes[0]->inArrayType  = inArrayType;
            childNodes[0]->outArrayType = inArrayType;

            childNodes[1]->SetInputBuffer(state);
            childNodes[1]->obOut        = childNodes[1]->obIn;
            childNodes[1]->inArrayType  = inArrayType;
            childNodes[1]->outArrayType = inArrayType;
        }
        // TRTR
        else
        {
            // NB: only c2r can fit into the output buffer for out-of-place transforms.

            // Transpose
            childNodes[0]->SetInputBuffer(state);
            childNodes[0]->obOut        = OB_TEMP;
            childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

            // c2c
            childNodes[1]->SetInputBuffer(state);
            // Note, need to check ip/op, since IP doesn't have USER_IN
            childNodes[1]->obOut = placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
            childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[1]->outArrayType = inArrayType;
            childNodes[1]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

            // Transpose
            childNodes[2]->SetInputBuffer(state);
            childNodes[2]->obOut        = OB_TEMP;
            childNodes[2]->inArrayType  = childNodes[1]->outArrayType;
            childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

            // c2c
            childNodes[3]->SetInputBuffer(state);
            childNodes[3]->obOut = placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
            childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
            childNodes[3]->outArrayType = inArrayType;
            childNodes[3]->AssignBuffers(state, flipOut, flipIn, obOutBuf);

            // Transpose
            childNodes[4]->SetInputBuffer(state);
            childNodes[4]->obOut        = OB_TEMP;
            childNodes[4]->inArrayType  = childNodes[3]->outArrayType;
            childNodes[4]->outArrayType = rocfft_array_type_complex_interleaved;
        }

        // c2r
        auto& previousNode = childNodes[childNodes.size() - 2];
        childNodes.back()->SetInputBuffer(state);
        childNodes.back()->obOut        = obOutBuf;
        childNodes.back()->inArrayType  = previousNode->outArrayType;
        childNodes.back()->outArrayType = outArrayType;
        childNodes.back()->AssignBuffers(state, flipIn, flipOut, obOutBuf);

        obOut = childNodes.back()->obOut;
    }

#if 0
    rocfft_cout << PrintScheme(scheme) << std::endl;
    for(int i = 0; i < childNodes.size(); ++i)
    {
        rocfft_cout << i << ": " << PrintScheme(childNodes[i]->scheme) << " : "
                  << PrintOperatingBuffer(childNodes[i]->obIn) << " -> "
                  << PrintOperatingBuffer(childNodes[i]->obOut) << std::endl;
    }
#endif
}
#endif

/*****************************************************
 * CS_KERNEL_COPY_R_TO_CMPLX
 * CS_KERNEL_COPY_HERM_TO_CMPLX
 * CS_KERNEL_COPY_CMPLX_TO_HERM
 * CS_KERNEL_COPY_CMPLX_TO_R
 * CS_KERNEL_APPLY_CALLBACK
 * NOTE- Temp Complex Buffer implements interleaved only
 *****************************************************/
RealTransDataCopyNode::SchemeFnCall const RealTransDataCopyNode::FnCallMap
    = {{CS_KERNEL_APPLY_CALLBACK, &apply_real_callback},
       {CS_KERNEL_COPY_R_TO_CMPLX, &real2complex},
       {CS_KERNEL_COPY_CMPLX_TO_R, &complex2real},
       {CS_KERNEL_COPY_HERM_TO_CMPLX, &hermitian2complex},
       {CS_KERNEL_COPY_CMPLX_TO_HERM, &complex2hermitian}};

void RealTransDataCopyNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    fnPtr = FnCallMap.at(scheme);

    if(scheme == CS_KERNEL_APPLY_CALLBACK)
    {
        gp.tpb_x = 64;
    }
    else
    {
        gp.b_x   = (length[0] - 1) / LAUNCH_BOUNDS_R2C_C2R_KERNEL + 1;
        gp.b_y   = batch;
        gp.tpb_x = LAUNCH_BOUNDS_R2C_C2R_KERNEL;
        gp.tpb_y = 1;
    }

    return;
}

/*****************************************************
 * CS_KERNEL_R_TO_CMPLX
 * CS_KERNEL_R_TO_CMPLX_TRANSPOSE
 * CS_KERNEL_CMPLX_TO_R
 * CS_KERNEL_TRANSPOSE_CMPLX_TO_R
 *****************************************************/
PrePostKernelNode::SchemeFnCall const PrePostKernelNode::FnCallMap
    = {{CS_KERNEL_R_TO_CMPLX, &r2c_1d_post},
       {CS_KERNEL_R_TO_CMPLX_TRANSPOSE, &r2c_1d_post_transpose},
       {CS_KERNEL_CMPLX_TO_R, &c2r_1d_pre},
       {CS_KERNEL_TRANSPOSE_CMPLX_TO_R, &transpose_c2r_1d_pre}};

size_t PrePostKernelNode::GetTwiddleTableLength()
{
    if(scheme == CS_KERNEL_R_TO_CMPLX || scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE
       || scheme == CS_KERNEL_CMPLX_TO_R)
        return 2 * length[0];
    else if(scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        return 2 * (length.back() - 1);

    throw std::runtime_error("GetTwiddleTableLength: Unexpected scheme in PrePostKernelNode: "
                             + PrintScheme(scheme));
}

void PrePostKernelNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    fnPtr = FnCallMap.at(scheme);
    // specify grid params only if the kernel from code generator

    return;
}
