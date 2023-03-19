// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tuning_plan_tuner.h"
#include "solution_map.h"
#include "tuning_helper.h"
#include "tuning_kernel_tuner.h"

#include <iterator>
#include <random>
#include <set>
#include <unordered_set>

// Some problems are not supported yet.
static const std::set<ComputeScheme> supported_prob_schemes = {CS_KERNEL_STOCKHAM,
                                                               CS_L1D_TRTRT,
                                                               CS_L1D_CC,
                                                               CS_L1D_CRT,
                                                               CS_2D_RTRT,
                                                               CS_2D_RC,
                                                               CS_3D_TRTRTR,
                                                               CS_3D_RTRT,
                                                               CS_3D_BLOCK_RC,
                                                               CS_3D_BLOCK_CR,
                                                               CS_3D_RC};

// return size_t: the "option_id" of the return node in its sol-vector
size_t SerializeTree(TreeNode* node, std::string& archName)
{
    std::vector<SolutionPtr> child_nodes;
    std::string              min_token, full_token;

    // if node is internal node, then it has childrens
    for(const auto& c : node->childNodes)
    {
        GetNodeToken(*(c.get()), min_token, full_token);

        size_t child_option = SerializeTree(c.get(), archName);
        child_nodes.push_back({min_token, child_option});
    }
    // if node is a leaf node without child-nodes, then the SOL_LEAF_NODE
    // should have an only childnode that is SOL_KERNEL_ONLY
    if(node->nodeType == NT_LEAF)
    {
        auto kernel_key = node->GetKernelKey();

        if(kernel_key == EmptyFMKey)
            min_token = solution_map::KERNEL_TOKEN_BUILTIN_KERNEL;
        else
            GetKernelToken(kernel_key, min_token);

        // the option-id is irrevalent since we will modify it during tuning
        child_nodes.push_back({min_token, 0});
    }

    GetNodeToken(*node, min_token, full_token);
    ProblemKey problemKey(archName, min_token);

    // Add a solution to primary map (as candidates):
    //   if SOL_INTERNAL_NODE --> childrens = decomposition
    //   if SOL_LEAF_NODE     --> childrens = one kernel-node
    // check_dup=false, primiary_map=true
    size_t my_option_id = TuningBenchmarker::GetSingleton().GetBindingSolutionMap()->add_solution(
        problemKey, node, child_nodes, node->isRootNode(), false);

    // save the problem name;
    if(node->isRootNode())
        TuningBenchmarker::GetSingleton().GetPacket()->tuning_problem_name = min_token;

    return my_option_id;
}

void EnumerateTrees(ExecPlan& execPlan)
{
    std::string archName = get_arch_name(execPlan.deviceProp);

    // TODO- plan-tuning: build tree several times to generate different trees
    {
        execPlan.rootPlan->RecursiveBuildTree();

        // Haven't supported type (real, bluestein...), return directly.
        // And tuner knows to skip work by testing "packet->total_nodes == 0"
        if(supported_prob_schemes.count(execPlan.rootPlan->scheme) == 0)
            return;

        assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->dimension);
        assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
        assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

        execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);

        // don't need to do SantiyCheck and KernelCheck now, since they are checking if
        // the kernels exist in function_pool which is not always true for RTC and tuning
        // execPlan.rootPlan->SanityCheck(rootScheme, execPlan.solution_kernels);

        if(TuningBenchmarker::GetSingleton().GetPacket()->tuning_phase == 0)
        {
            // Adding decompoistion solutions from this tree-decomposition
            SerializeTree(execPlan.rootPlan.get(), archName);
        }

        // Adding kernel candidates
        EnumerateKernelConfigs(execPlan);
    }
}