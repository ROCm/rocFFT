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

#include "solution_map.h"
#include "../../shared/environment.h"
#include "data_descriptor.h"
#include "library_path.h"
#include "logging.h"
#include "node_factory.h"

#include <fstream>

namespace fs = std::filesystem;

static std::regex regEx("[^:;,\"\\{\\}\\[\\s]+", std::regex_constants::optimize);

static const char* def_solution_map_path = "rocfft_solution_map.dat";

static std::map<SolutionNodeType, std::string> SolutionNodeTypetoStrMap()
{
    std::map<SolutionNodeType, std::string> SNTtoStr = {{SOL_INTERNAL_NODE, "SOL_INTERNAL_NODE"},
                                                        {SOL_LEAF_NODE, "SOL_LEAF_NODE"},
                                                        {SOL_KERNEL_ONLY, "SOL_KERNEL_ONLY"},
                                                        {SOL_BUILTIN_KERNEL, "SOL_BUILTIN_KERNEL"}};
    return SNTtoStr;
}

static std::map<std::string, SolutionNodeType> StrToSolutionNodeTypeMap()
{
    std::map<std::string, SolutionNodeType> StrToSNT;
    for(auto i : SolutionNodeTypetoStrMap())
        StrToSNT.emplace(i.second, i.first);
    return StrToSNT;
}

std::string PrintSolutionNodeType(const SolutionNodeType snt)
{
    static auto SNTtoString = SolutionNodeTypetoStrMap();
    return SNTtoString.at(snt);
}

SolutionNodeType StrToSolutionNodeType(const std::string& str)
{
    static auto str2SNT = StrToSolutionNodeTypeMap();
    return str2SNT.at(str);
}

static fs::path get_solution_map_path(const std::string& arch = "any")
{
    // if folder of data is set, find file in that folder,
    // else search in the library folder
    std::string folder_str = rocfft_getenv("ROCFFT_SOLUTION_MAP_FOLDER");
    fs::path    folder_path(folder_str.c_str());
    if(folder_str.empty())
    {
        auto lib_path = get_library_path();
        if(lib_path.empty())
            return {};

        folder_path = lib_path.parent_path();
    }

    // search the file with the arch prefix
    std::string prefix(arch + "_");
    fs::path    file_name(prefix.c_str());
    file_name += def_solution_map_path;

    return folder_path / file_name;
}

template <>
struct ToString<SolutionPtr>
{
    std::string print(const SolutionPtr& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("child_token", value.child_token) + ",";
        str += FieldDescriptor<size_t>().describe("child_option", value.child_option);
        str += "}";
        return str;
    }
};

template <>
struct FromString<SolutionPtr>
{
    void Get(SolutionPtr& ret, std::sregex_token_iterator& current) const
    {
        FieldParser<std::string>().parse("child_token", ret.child_token, current);
        FieldParser<size_t>().parse("child_option", ret.child_option, current);
    }
};

template <>
struct ToString<SolutionNode>
{
    std::string print(const SolutionNode& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("sol_node_type",
                                                       PrintSolutionNodeType(value.sol_node_type))
               + ",";

        if(value.sol_node_type != SOL_BUILTIN_KERNEL)
        {
            if(value.sol_node_type == SOL_KERNEL_ONLY)
            {
                str += FieldDescriptor<FMKey>().describe("kernel_key", value.kernel_key);
            }
            else
            {
                str += FieldDescriptor<std::string>().describe("using_scheme",
                                                               PrintScheme(value.using_scheme))
                       + ",";
                str += VectorFieldDescriptor<SolutionPtr>().describe("solution_childnodes",
                                                                     value.solution_childnodes);
            }
        }

        str += "}";
        return str;
    }
};

template <>
struct FromString<SolutionNode>
{
    void Get(SolutionNode& ret, std::sregex_token_iterator& current) const
    {
        std::string sol_node_type_str;
        std::string scheme_str;

        FieldParser<std::string>().parse("sol_node_type", sol_node_type_str, current);
        ret.sol_node_type = StrToSolutionNodeType(sol_node_type_str);

        if(ret.sol_node_type != SOL_BUILTIN_KERNEL)
        {
            if(ret.sol_node_type == SOL_KERNEL_ONLY)
            {
                FieldParser<FMKey>().parse("kernel_key", ret.kernel_key, current);
                ret.using_scheme = std::get<2>(ret.kernel_key);
            }
            else
            {
                FieldParser<std::string>().parse("using_scheme", scheme_str, current);
                ret.using_scheme = StrToComputeScheme(scheme_str);

                VectorFieldParser<SolutionPtr>().parse(
                    "solution_childnodes", ret.solution_childnodes, current);
            }
        }
    }
};

template <>
struct ToString<ProblemKey>
{
    std::string print(const ProblemKey& value) const
    {
        std::string str = "{";
        str += FieldDescriptor<std::string>().describe("arch", value.arch) + ",";
        str += FieldDescriptor<std::string>().describe("token", value.probToken);
        str += "}";
        return str;
    }
};

template <>
struct FromString<ProblemKey>
{
    void Get(ProblemKey& ret, std::sregex_token_iterator& current) const
    {
        std::string arch, token;
        FieldParser<std::string>().parse("arch", arch, current);
        FieldParser<std::string>().parse("token", token, current);
        ret = {arch, token};
    }
};

bool solution_map::SolutionNodesAreEqual(const SolutionNode& lhs,
                                         const SolutionNode& rhs,
                                         const std::string&  arch)
{
    // NB:
    // std::tie couldn't compare the .size() so we compare .size() outside std::tie
    bool members_equal = std::tie(lhs.sol_node_type, lhs.using_scheme, lhs.kernel_key)
                         == std::tie(rhs.sol_node_type, rhs.using_scheme, rhs.kernel_key);
    if((members_equal == false)
       || (lhs.solution_childnodes.size() != rhs.solution_childnodes.size()))
        return false;

    for(size_t i = 0; i < lhs.solution_childnodes.size(); ++i)
    {
        auto& lhs_child_ptr = lhs.solution_childnodes[i];
        auto& rhs_child_ptr = rhs.solution_childnodes[i];

        auto lhs_child_key = ProblemKey(arch, lhs_child_ptr.child_token);
        auto rhs_child_key = ProblemKey(arch, rhs_child_ptr.child_token);

        if(SolutionNodesAreEqual(get_solution_node(lhs_child_key, lhs_child_ptr.child_option),
                                 get_solution_node(rhs_child_key, rhs_child_ptr.child_option),
                                 arch)
           == false)
            return false;
    }

    return true;
}

void solution_map::setup()
{
    // set ROCFFT_READ_SOL_MAP_ENABLE to enable solution map reading
    // default is false for now
    if(!rocfft_getenv("ROCFFT_READ_SOL_MAP_ENABLE").empty())
    {
        // read data from any_arch
        auto sol_map_input = get_solution_map_path();
        read_solution_map_data(sol_map_input);

        // read data from current arch
        auto deviceProp = get_curr_device_prop();
        sol_map_input   = get_solution_map_path(get_arch_name(deviceProp));
        read_solution_map_data(sol_map_input);
    }
}

bool solution_map::has_solution_node(const ProblemKey& probKey, size_t option_id)
{
    ProbSolMap& dst_map = solution_nodes;

    // no this key
    if(dst_map.count(probKey) == 0)
        return false;

    // no this option_id
    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions.size() > option_id;
}

SolutionNode& solution_map::get_solution_node(const ProblemKey& probKey, size_t option_id)
{
    // be sure we have checked has_solution_node();
    ProbSolMap& dst_map = solution_nodes;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id];
}

FMKey& solution_map::get_solution_kernel(const ProblemKey& probKey, size_t option_id)
{
    // be sure we have checked has_solution_node();
    ProbSolMap& dst_map = solution_nodes;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id].kernel_key;
}

// add a solution of a problem to the map, should be called by a benchmarker
bool solution_map::add_solution(const ProblemKey&               probKey,
                                TreeNode*                       currentNode,
                                const std::vector<SolutionPtr>& children,
                                bool                            check_dup)
{
    SolutionNode solution;
    solution.using_scheme  = currentNode->scheme;
    solution.sol_node_type = (currentNode->nodeType == NT_LEAF) ? SOL_LEAF_NODE : SOL_INTERNAL_NODE;
    solution.solution_childnodes = children;

    return add_solution(probKey, solution, check_dup);
}

// add a solution of a problem to the map, should be called by a benchmarker
bool solution_map::add_solution(const ProblemKey& probKey, const FMKey& kernel_key, bool check_dup)
{
    SolutionNode solution;
    solution.using_scheme  = std::get<2>(kernel_key);
    solution.sol_node_type = SOL_KERNEL_ONLY;
    solution.kernel_key    = kernel_key;

    return add_solution(probKey, solution, check_dup);
}

// directly insert a solution of a problem to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey&   probKey,
                                  const SolutionNode& solution,
                                  bool                check_dup)
{
    ProbSolMap& dst_map = solution_nodes;

    // no this key, emplace one new vector
    if(dst_map.count(probKey) == 0)
        dst_map.emplace(probKey, SolutionNodeVec());

    auto& sol_vec = dst_map.at(probKey);

    // append solution to the same key
    if(check_dup)
    {
        const std::string& arch = probKey.arch;
        for(size_t option_id = 0; option_id < sol_vec.size(); ++option_id)
        {
            // find an existing solution that is identical, then don't insert, simply return that option
            if(SolutionNodesAreEqual(solution, sol_vec[option_id], arch))
                return option_id;
        }
    }

    // if we are here, it could be either check_dup but not found any indentical,
    // or force append, don't check_dup (for tuning)
    sol_vec.push_back(solution);

    return sol_vec.size() - 1;
}

// read the map from input stream
bool solution_map::read_solution_map_data(const fs::path& sol_map_in_path)
{
    if(LOG_TRACE_ENABLED())
        (*LogSingleton::GetInstance().GetTraceOS())
            << "reading solution map data from: " << sol_map_in_path.c_str() << std::endl;

    // Read text from the file. If file not found, do nothing
    std::string solution_map_text = "";
    if(fs::exists(sol_map_in_path))
    {
        std::ifstream in_file(sol_map_in_path.c_str());
        std::string   line;

        while(in_file.good())
        {
            std::getline(in_file, line);
            solution_map_text += line;
        }
    }

    if(solution_map_text.back() == ']')
        solution_map_text.resize(solution_map_text.size() - 1);

    std::sregex_token_iterator tokens{solution_map_text.begin(), solution_map_text.end(), regEx, 0};
    std::sregex_token_iterator endIt;
    for(; tokens != endIt; ++tokens)
    {
        ProblemKey                probKey;
        std::vector<SolutionNode> solutionVec;
        FieldParser<ProblemKey>().parse("Problem", probKey, tokens);
        VectorFieldParser<SolutionNode>().parse("Solutions", solutionVec, tokens);
        solution_nodes.emplace(probKey, solutionVec);
    }
    return true;
}

// write the map to output stream
bool solution_map::write_solution_map_data(const fs::path& sol_map_out_path,
                                           const fs::path& node_pool_out_path)
{
    std::stringstream ss;
    std::string       COMMA = "";
    ss << "[" << std::endl;

    std::string blanks(15, ' ');
    for(auto& [key, value] : solution_nodes)
    {
        ss << COMMA;
        ss << "{" << FieldDescriptor<ProblemKey>().describe("Problem", key) << "," << std::endl
           << "  "
           << VectorFieldDescriptor<SolutionNode>().describe("Solutions", value, true, blanks)
           << "}" << std::endl;

        if(COMMA.size() == 0)
            COMMA = ",";
    }
    ss << "]" << std::endl;

    std::cout << ss.str();
    return true;
}
