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

#ifndef TREE_NODE_BLUE_H
#define TREE_NODE_BLUE_H

#include "tree_node.h"

/*****************************************************
 * CS_BLUESTEIN
 *****************************************************/
class BluesteinNode : public InternalNode
{
    friend class NodeFactory;

protected:
    BluesteinNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_BLUESTEIN;
    }

    void AssignBuffers_internal(TraverseState&   state,
                                OperatingBuffer& flipIn,
                                OperatingBuffer& flipOut,
                                OperatingBuffer& obOutBuf) override;
    void AssignParams_internal() override;
    void BuildTree_internal() override;
};

/*****************************************************
 * Component of Bluestein
 * Chirp, XXXMul
 *****************************************************/
class BluesteinComponentNode : public LeafNode
{
    friend class NodeFactory;

protected:
    BluesteinComponentNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
    }

    void SetupGPAndFnPtr_internal(DevFnCall& fnPtr, GridParam& gp) override;
};

#endif // TREE_NODE_BLUE_H
