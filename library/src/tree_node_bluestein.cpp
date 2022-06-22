// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node_bluestein.h"
#include "kernel_launch.h"
#include "node_factory.h"

inline size_t FindBlue(size_t len, rocfft_precision precision)
{
    size_t lenPow2 = 1;
    while(lenPow2 < len)
        lenPow2 <<= 1;

    size_t minLenBlue  = 2 * len - 1;
    size_t length      = minLenBlue;
    size_t lenPow2Blue = 2 * lenPow2;

    // We don't want to choose a non-pow2 length that is too close to lenPow2Blue,
    // otherwise using a non-pow2 length may end up being slower than using lenPow2Blue.
    // This ratio has been experimentally verified to yield a non-pow2 length that is at
    // least as fast as its corresponding pow2 length.
    double lenCutOffRatio = .9;

    for(length = minLenBlue; length < lenPow2Blue; ++length)
    {
        auto lenSupported = NodeFactory::NonPow2LengthSupported(precision, length);
        auto lenRatio     = static_cast<double>(length) / static_cast<double>(lenPow2Blue);

        if(lenSupported && lenRatio < lenCutOffRatio)
            break;
    }

    return length;
}

/*****************************************************
 * CS_BLUESTEIN
 *****************************************************/
void BluesteinNode::BuildTree_internal()
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.

    lengthBlue = FindBlue(length[0], precision);

    auto chirpPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this);
    chirpPlan->dimension = 1;
    chirpPlan->length.push_back(length[0]);
    chirpPlan->lengthBlue = lengthBlue;
    chirpPlan->direction  = direction;
    chirpPlan->batch      = 1;
    chirpPlan->large1D    = 2 * length[0];

    auto padmulPlan        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_PAD_MUL, this);
    padmulPlan->dimension  = 1;
    padmulPlan->length     = length;
    padmulPlan->lengthBlue = lengthBlue;

    NodeMetaData fftiPlanData(this);
    fftiPlanData.dimension = 1;
    fftiPlanData.length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftiPlanData.length.push_back(length[index]);
    }
    fftiPlanData.iOffset = 2 * lengthBlue;
    fftiPlanData.oOffset = 2 * lengthBlue;
    auto fftiPlan        = NodeFactory::CreateExplicitNode(fftiPlanData, this);
    fftiPlan->RecursiveBuildTree();

    NodeMetaData fftcPlanData(this);
    fftcPlanData.dimension = 1;
    fftcPlanData.length.push_back(lengthBlue);
    fftcPlanData.batch   = 1;
    fftcPlanData.iOffset = lengthBlue;
    fftcPlanData.oOffset = lengthBlue;
    auto fftcPlan        = NodeFactory::CreateExplicitNode(fftcPlanData, this);
    fftcPlan->RecursiveBuildTree();

    auto fftmulPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_FFT_MUL, this);
    fftmulPlan->dimension = 1;
    fftmulPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftmulPlan->length.push_back(length[index]);
    }
    fftmulPlan->lengthBlue = lengthBlue;

    NodeMetaData fftrPlanData(this);
    fftrPlanData.dimension = 1;
    fftrPlanData.length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftrPlanData.length.push_back(length[index]);
    }
    fftrPlanData.direction = -direction;
    fftrPlanData.iOffset   = 2 * lengthBlue;
    fftrPlanData.oOffset   = 2 * lengthBlue;
    auto fftrPlan          = NodeFactory::CreateExplicitNode(fftrPlanData, this);
    fftrPlan->RecursiveBuildTree();

    auto resmulPlan        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_RES_MUL, this);
    resmulPlan->dimension  = 1;
    resmulPlan->length     = length;
    resmulPlan->lengthBlue = lengthBlue;

    // 7 node of bluestein
    childNodes.emplace_back(std::move(chirpPlan));
    childNodes.emplace_back(std::move(padmulPlan));
    childNodes.emplace_back(std::move(fftiPlan));
    childNodes.emplace_back(std::move(fftcPlan));
    childNodes.emplace_back(std::move(fftmulPlan));
    childNodes.emplace_back(std::move(fftrPlan));
    childNodes.emplace_back(std::move(resmulPlan));
}

void BluesteinNode::AssignParams_internal()
{
    auto& chirpPlan  = childNodes[0];
    auto& padmulPlan = childNodes[1];
    auto& fftiPlan   = childNodes[2];
    auto& fftcPlan   = childNodes[3];
    auto& fftmulPlan = childNodes[4];
    auto& fftrPlan   = childNodes[5];
    auto& resmulPlan = childNodes[6];

    chirpPlan->inStride.push_back(1);
    chirpPlan->iDist = chirpPlan->lengthBlue;
    chirpPlan->outStride.push_back(1);
    chirpPlan->oDist = chirpPlan->lengthBlue;

    padmulPlan->inStride = inStride;
    padmulPlan->iDist    = iDist;

    padmulPlan->outStride.push_back(1);
    padmulPlan->oDist = padmulPlan->lengthBlue;
    for(size_t index = 1; index < length.size(); index++)
    {
        padmulPlan->outStride.push_back(padmulPlan->oDist);
        padmulPlan->oDist *= length[index];
    }

    fftiPlan->inStride  = padmulPlan->outStride;
    fftiPlan->iDist     = padmulPlan->oDist;
    fftiPlan->outStride = fftiPlan->inStride;
    fftiPlan->oDist     = fftiPlan->iDist;

    fftiPlan->AssignParams();

    fftcPlan->inStride  = chirpPlan->outStride;
    fftcPlan->iDist     = chirpPlan->oDist;
    fftcPlan->outStride = fftcPlan->inStride;
    fftcPlan->oDist     = fftcPlan->iDist;

    fftcPlan->AssignParams();

    fftmulPlan->inStride  = fftiPlan->outStride;
    fftmulPlan->iDist     = fftiPlan->oDist;
    fftmulPlan->outStride = fftmulPlan->inStride;
    fftmulPlan->oDist     = fftmulPlan->iDist;

    fftrPlan->inStride  = fftmulPlan->outStride;
    fftrPlan->iDist     = fftmulPlan->oDist;
    fftrPlan->outStride = fftrPlan->inStride;
    fftrPlan->oDist     = fftrPlan->iDist;

    fftrPlan->AssignParams();

    resmulPlan->inStride  = fftrPlan->outStride;
    resmulPlan->iDist     = fftrPlan->oDist;
    resmulPlan->outStride = outStride;
    resmulPlan->oDist     = oDist;
}

#if !GENERIC_BUF_ASSIGMENT
void BluesteinNode::AssignBuffers_internal(TraverseState&   state,
                                           OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    assert(childNodes.size() == 7);

    OperatingBuffer savFlipIn  = flipIn;
    OperatingBuffer savFlipOut = flipOut;
    OperatingBuffer savOutBuf  = obOutBuf;

    flipIn   = OB_TEMP_BLUESTEIN;
    flipOut  = OB_TEMP;
    obOutBuf = OB_TEMP_BLUESTEIN;

    // CS_KERNEL_CHIRP effectively takes no inputs and does not
    // connect to the previous kernel in the chain, so don't assign
    // obIn using SetInputBuffer.
    assert(childNodes[0]->scheme == CS_KERNEL_CHIRP);
    childNodes[0]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[0]->obOut = OB_TEMP_BLUESTEIN;

    assert(childNodes[1]->scheme == CS_KERNEL_PAD_MUL);
    childNodes[1]->SetInputBuffer(state);
    childNodes[1]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[2]->SetInputBuffer(state);
    childNodes[2]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[2]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    childNodes[3]->SetInputBuffer(state);
    childNodes[3]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[3]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    assert(childNodes[4]->scheme == CS_KERNEL_FFT_MUL);
    childNodes[4]->SetInputBuffer(state);
    childNodes[4]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[5]->SetInputBuffer(state);
    childNodes[5]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[5]->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    assert(childNodes[6]->scheme == CS_KERNEL_RES_MUL);
    childNodes[6]->SetInputBuffer(state);
    childNodes[6]->obOut = (parent == nullptr) ? OB_USER_OUT : obOut;

    obOut = childNodes[6]->obOut;

    flipIn   = savFlipIn;
    flipOut  = savFlipOut;
    obOutBuf = savOutBuf;
}
#endif

/*****************************************************
 * Component of Bluestein
 * Chirp, XXXMul
 *****************************************************/
void BluesteinComponentNode::SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp)
{
    gp.wgs_x = 64;
    fnPtr    = (scheme == CS_KERNEL_CHIRP) ? &FN_PRFX(chirp) : &FN_PRFX(mul);

    return;
}
