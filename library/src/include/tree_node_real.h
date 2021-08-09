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

#ifndef TREE_NODE_REAL_H
#define TREE_NODE_REAL_H

#include "tree_node.h"

/*****************************************************
 * CS_REAL_TRANSFORM_USING_CMPLX
 *****************************************************/
class RealTransCmplxNode : public InternalNode
{
    friend class NodeFactory;

protected:
    RealTransCmplxNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_TRANSFORM_USING_CMPLX;
    }

    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * CS_REAL_TRANSFORM_EVEN
 *****************************************************/
class RealTransEvenNode : public InternalNode
{
    friend class NodeFactory;

protected:
    RealTransEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_TRANSFORM_EVEN;
    }

    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
    void AssignParams_internal() override;
    void BuildTree_internal() override;

public:
    // 3D Even can possibly set this
    bool try_fuse_pre_post_processing = false;
};

/*****************************************************
 * CS_REAL_2D_EVEN
 *****************************************************/
class Real2DEvenNode : public InternalNode
{
    friend class NodeFactory;

protected:
    Real2DEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_2D_EVEN;
    }

    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * CS_REAL_3D_EVEN
 *****************************************************/
class Real3DEvenNode : public InternalNode
{
    friend class NodeFactory;

protected:
    Real3DEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_3D_EVEN;
    }

    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * CS_KERNEL_COPY_R_TO_CMPLX
 * CS_KERNEL_COPY_HERM_TO_CMPLX
 * CS_KERNEL_COPY_CMPLX_TO_HERM
 * CS_KERNEL_COPY_CMPLX_TO_R
 * CS_KERNEL_APPLY_CALLBACK
 *****************************************************/
class RealTransDataCopyNode : public LeafNode
{
    friend class NodeFactory;

private:
    typedef std::map<ComputeScheme, DevFnCall> SchemeFnCall;
    static const SchemeFnCall                  FnCallMap;

protected:
    RealTransDataCopyNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        allowInplace = false;
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;
};

/*****************************************************
 * CS_KERNEL_R_TO_CMPLX
 * CS_KERNEL_R_TO_CMPLX_TRANSPOSE
 * CS_KERNEL_CMPLX_TO_R
 * CS_KERNEL_TRANSPOSE_CMPLX_TO_R
 *****************************************************/
class PrePostKernelNode : public LeafNode
{
    friend class NodeFactory;

private:
    typedef std::map<ComputeScheme, DevFnCall> SchemeFnCall;
    static const SchemeFnCall                  FnCallMap;

protected:
    PrePostKernelNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        need_twd_table = true;
        twd_no_radices = true;
        allowInplace   = false;
    }

    size_t GetTwiddleTableLength() override;
    void   SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;
};

#endif // TREE_NODE_REAL_H
