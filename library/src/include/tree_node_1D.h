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

#ifndef TREE_NODE_1D_H
#define TREE_NODE_1D_H

#include "tree_node.h"

/*****************************************************
 * L1D_TRTRT  *
 *****************************************************/
class TRTRT1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit TRTRT1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_TRTRT;
    }
#if !GENERIC_BUF_ASSIGMENT
    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
#endif
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * L1D_CC  *
 *****************************************************/
class CC1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit CC1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_CC;
    }
#if !GENERIC_BUF_ASSIGMENT
    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
#endif
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * L1D_CRT  *
 *****************************************************/
class CRT1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit CRT1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_CRT;
    }
#if !GENERIC_BUF_ASSIGMENT
    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
#endif
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * CS_KERNEL_STOCKHAM  *
 *****************************************************/
class Stockham1DNode : public LeafNode
{
    friend class NodeFactory;

protected:
    Stockham1DNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;

public:
    bool CreateTwiddleTableResource() override;
};

/*****************************************************
 * SBCC  *
 *****************************************************/
class SBCCNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBCCNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;

public:
    // reads + writes are along columns so both may benefit from padding
    bool PaddingBenefitsInput() override
    {
        return true;
    }
    bool PaddingBenefitsOutput() override
    {
        return true;
    }
};

/*****************************************************
 * SBRC  *
 *****************************************************/
class SBRCNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBRCNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
        allowInplace   = false;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;

public:
    SBRC_TRANSPOSE_TYPE sbrc_transpose_type(unsigned int blockWidth) const override;

    // writes are along columns so they may benefit from padding
    bool PaddingBenefitsOutput() override
    {
        return true;
    }
};

/*****************************************************
 * SBCR  *
 *****************************************************/
class SBCRNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBCRNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
        allowInplace   = false;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;

public:
    bool CreateTwiddleTableResource() override;

    // reads are along columns so they may benefit from padding
    bool PaddingBenefitsInput() override
    {
        return true;
    }
};

#endif // TREE_NODE_1D_H
