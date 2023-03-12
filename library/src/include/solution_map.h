/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef SOLUTION_MAP_H
#define SOLUTION_MAP_H

#include "tree_node.h"
#include <unordered_map>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

enum SolutionNodeType
{
    SOL_BUILTIN_KERNEL, // a solution indicating this kernel should be a built-in one, such as transpose, r2c, c2r...
    SOL_KERNEL_ONLY, // a solution representing a kernel only, nothing to do with problem
    SOL_LEAF_NODE, // a solution representing the tree-leaf-node
    SOL_INTERNAL_NODE, // a solution with tree decomposition
};

std::string      PrintSolutionNodeType(const SolutionNodeType snt);
SolutionNodeType StrToSolutionNodeType(const std::string& str);

// arch, problem-size token
struct ProblemKey
{
    std::string arch;
    std::string probToken;

    ProblemKey() = default;

    ProblemKey(const std::string& arch, const std::string& probToken)
        : arch(arch)
        , probToken(probToken)
    {
    }

    bool operator==(const ProblemKey& rhs) const
    {
        return (arch == rhs.arch) && (probToken == rhs.probToken);
    }
};

template <>
struct ToString<ProblemKey>;

template <>
struct FromString<ProblemKey>;

struct ProbKeyHash
{
    size_t operator()(const ProblemKey& k) const noexcept
    {
        size_t h = 0;
        h ^= std::hash<std::string>{}(k.arch);
        h ^= std::hash<std::string>{}(k.probToken);
        return h;
    }
};

struct SolutionPtr
{
    std::string child_token  = "";
    size_t      child_option = 0;
};

// Implementing the ToString / FromString (data_descriptor.h)
// for writing-to/reading-from texted-format solution map
template <>
struct ToString<SolutionPtr>;

template <>
struct FromString<SolutionPtr>;

struct SolutionNode
{
    SolutionNodeType sol_node_type = SOL_INTERNAL_NODE;
    ComputeScheme    using_scheme;
    FMKey            kernel_key = EmptyFMKey;
    // like the childnodes on tree-node, a childnode could be internal/leaf/kernel-node
    std::vector<SolutionPtr> solution_childnodes;

    SolutionNode()                    = default;
    SolutionNode(const SolutionNode&) = default;

    SolutionNode& operator=(const SolutionNode&) = default;
};

template <>
struct ToString<SolutionNode>;

template <>
struct FromString<SolutionNode>;

using SolutionNodeVec = std::vector<SolutionNode>;

using ProbSolMap = std::unordered_map<ProblemKey, SolutionNodeVec, ProbKeyHash>;

class solution_map
{
    ProbSolMap solution_nodes;

    ROCFFT_EXPORT solution_map() = default;

private:
    // check if two solution nodes have identical semantic
    bool SolutionNodesAreEqual(const SolutionNode& lhs,
                               const SolutionNode& rhs,
                               const std::string&  arch);

public:
    solution_map(const solution_map&) = delete;

    solution_map& operator=(const solution_map&) = delete;

    static solution_map& get_solution_map()
    {
        static solution_map sol_map;
        return sol_map;
    }

    ~solution_map() = default;

    void setup();

    bool has_solution_node(const ProblemKey& probKey, size_t option_id = 0);

    SolutionNode& get_solution_node(const ProblemKey& probKey, size_t option_id = 0);

    FMKey& get_solution_kernel(const ProblemKey& probKey, size_t option_id = 0);

    // add a solution of a problem to the map, should be called by a benchmarker
    bool add_solution(const ProblemKey&               probKey,
                      TreeNode*                       currentNode,
                      const std::vector<SolutionPtr>& children,
                      bool                            check_dup);

    // add a solution of a problem to the map, should be called by a benchmarker
    bool add_solution(const ProblemKey& probKey, const FMKey& kernel_key, bool check_dup);

    // directly insert a solution of a problem to the map, should be called by a benchmarker
    size_t add_solution(const ProblemKey& probKey, const SolutionNode& solution, bool check_dup);

    // read the map from input stream
    bool read_solution_map_data(const fs::path& sol_map_in_path);

    // write the map to output stream
    bool write_solution_map_data(const fs::path& sol_map_out_path,
                                 const fs::path& node_pool_out_path);
};

#endif // SOLUTION_MAP_H
