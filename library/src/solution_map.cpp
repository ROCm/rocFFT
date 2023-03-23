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

const char* solution_map::KERNEL_TOKEN_BUILTIN_KERNEL   = "kernel_token_builtin_kernel";
const char* solution_map::LEAFNODE_TOKEN_BUILTIN_KERNEL = "leafnode_token_builtin_kernel";

static std::map<SolutionNodeType, std::string> SolutionNodeTypetoStrMap()
{
    std::map<SolutionNodeType, std::string> SNTtoStr = {{SOL_DUMMY, "SOL_DUMMY"},
                                                        {SOL_INTERNAL_NODE, "SOL_INTERNAL_NODE"},
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
                                                       PrintSolutionNodeType(value.sol_node_type));

        if(value.sol_node_type != SOL_BUILTIN_KERNEL)
        {
            str += ",";
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

// private version called by constructor
size_t solution_map::add_solution_private(const ProblemKey& probKey, const SolutionNode& solution)
{
    // no this key, emplace one new vector
    if(primary_sol_map.count(probKey) == 0)
        primary_sol_map.emplace(probKey, SolutionNodeVec());

    auto& sol_vec = primary_sol_map.at(probKey);
    sol_vec.push_back(solution);
    return sol_vec.size() - 1;
}

bool solution_map::SolutionNodesAreEqual(const SolutionNode& lhs,
                                         const SolutionNode& rhs,
                                         const std::string&  arch,
                                         bool                primary_map)
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

        if(SolutionNodesAreEqual(
               get_solution_node(lhs_child_key, lhs_child_ptr.child_option, primary_map),
               get_solution_node(rhs_child_key, rhs_child_ptr.child_option, primary_map),
               arch,
               primary_map)
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
        // if we have speicified an explicit file-path, then read from it,
        std::string explict_read_path_str = rocfft_getenv("ROCFFT_READ_EXPLICIT_SOL_MAP_FILE");
        if(!explict_read_path_str.empty())
        {
            fs::path read_from_path(explict_read_path_str.c_str());
            read_solution_map_data(read_from_path);
        }
        // otherwise we read with a default file name arch_rocfft_solution_map.dat
        // under the folder set in envvar ROCFFT_SOLUTION_MAP_FOLDER
        else
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
}

bool solution_map::has_solution_node(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    // no this key
    if(dst_map.count(probKey) == 0)
        return false;

    // no this option_id
    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions.size() > option_id;
}

SolutionNode&
    solution_map::get_solution_node(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    // be sure we have checked has_solution_node();
    if(!has_solution_node(probKey, option_id, primary_map))
        throw std::runtime_error(
            "get_solution_node failed. the solution_node doesn't exist: ProbKey=(" + probKey.arch
            + "," + probKey.probToken + "), option_id=" + std::to_string(option_id));

    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id];
}

FMKey&
    solution_map::get_solution_kernel(const ProblemKey& probKey, size_t option_id, bool primary_map)
{
    // be sure we have checked has_solution_node();
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    SolutionNodeVec& solutions = dst_map.at(probKey);
    return solutions[option_id].kernel_key;
}

// setup a solution of a problem and insert to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey&               probKey,
                                  TreeNode*                       currentNode,
                                  const std::vector<SolutionPtr>& children,
                                  bool                            isRootProb,
                                  bool                            check_dup,
                                  bool                            primary_map)
{
    SolutionNode solution;
    solution.using_scheme  = currentNode->scheme;
    solution.sol_node_type = (currentNode->nodeType == NT_LEAF) ? SOL_LEAF_NODE : SOL_INTERNAL_NODE;
    solution.solution_childnodes = children;

    return add_solution(probKey, solution, isRootProb, check_dup, primary_map);
}

// setup a solution of a problem and insert to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey& probKey,
                                  const FMKey&      kernel_key,
                                  bool              check_dup,
                                  bool              primary_map)
{
    SolutionNode solution;
    solution.using_scheme  = (kernel_key == EmptyFMKey) ? CS_NONE : std::get<2>(kernel_key);
    solution.sol_node_type = (kernel_key == EmptyFMKey) ? SOL_BUILTIN_KERNEL : SOL_KERNEL_ONLY;
    solution.kernel_key    = kernel_key;

    return add_solution(probKey, solution, false, check_dup, primary_map);
}

// directly insert a solution of a problem to the map, should be called by a benchmarker
size_t solution_map::add_solution(const ProblemKey&   probKey,
                                  const SolutionNode& solution,
                                  bool                isRootProb,
                                  bool                check_dup,
                                  bool                primary_map)
{
    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    // CS_KERNEL_STOCKHAM could be a problem scheme but also a kernel.
    bool is_problem_scheme = ComputeSchemeIsAProblem(solution.using_scheme)
                             && (solution.sol_node_type != SOL_KERNEL_ONLY);

    // no this key, emplace one new vector
    if(dst_map.count(probKey) == 0)
    {
        dst_map.emplace(probKey, SolutionNodeVec());

        // if this is a solution for a problem (not kernel or non-problem)
        // then we insert a dummy one in the front, which is reserved for root-problem
        if(is_problem_scheme)
            dst_map.at(probKey).push_back(SolutionNode::DummySolutionNode());
    }

    auto& sol_vec = dst_map.at(probKey);

    // append solution to the same key
    // Root-solution never checks duplication since it will always be the first element
    // So adding a root-solution = simply overwrite the first one
    if(isRootProb)
    {
        // just a double-check
        assert(is_problem_scheme && (sol_vec.size() > 0));
        sol_vec[0] = solution;
        return 0;
    }
    else if(check_dup)
    {
        // if the solution is not a problem-solution (i.e. kernel solution or non-problem)
        // then there is no a "dummy solution" (or exclusive root-solution) in the vector,
        // so we still start from 0, otherwise, start from option 1.
        const std::string& arch            = probKey.arch;
        size_t             check_option_id = (is_problem_scheme) ? 1 : 0;
        for(; check_option_id < sol_vec.size(); ++check_option_id)
        {
            // find an existing solution that is identical, then don't insert, simply return that option
            if(SolutionNodesAreEqual(solution, sol_vec[check_option_id], arch, primary_map))
                return check_option_id;
        }
    }

    // if we are here, it could be either check_dup but not found any indentical,
    // or force append, don't check_dup (for tuning)
    sol_vec.push_back(solution);

    return sol_vec.size() - 1;
}

// read the map from input stream
bool solution_map::read_solution_map_data(const fs::path& sol_map_in_path, bool primary_map)
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

    ProbSolMap& dst_map = (primary_map) ? primary_sol_map : temp_working_map;

    std::sregex_token_iterator tokens{solution_map_text.begin(), solution_map_text.end(), regEx, 0};
    std::sregex_token_iterator endIt;
    for(; tokens != endIt; ++tokens)
    {
        ProblemKey                probKey;
        std::vector<SolutionNode> solutionVec;
        FieldParser<ProblemKey>().parse("Problem", probKey, tokens);
        VectorFieldParser<SolutionNode>().parse("Solutions", solutionVec, tokens);
        dst_map.emplace(probKey, solutionVec);
    }
    return true;
}

// write the map to output stream
bool solution_map::write_solution_map_data(const fs::path& sol_map_out_path,
                                           bool            sort,
                                           bool            primary_map)
{
    if(LOG_TUNING_ENABLED())
        (*LogSingleton::GetInstance().GetTuningOS())
            << "writing solution map data to: " << sol_map_out_path.c_str() << std::endl;

    std::ofstream outfile;
    outfile.open(sol_map_out_path.c_str(), (std::ios::out | std::ios::trunc));
    if(!outfile.is_open())
        throw std::runtime_error("Write solution map failed. Cannot open/create output file: "
                                 + sol_map_out_path.string());

    std::stringstream ss;
    ProbSolMap&       writing_map = (primary_map) ? primary_sol_map : temp_working_map;

    std::string COMMA = "";
    ss << "[" << std::endl;

    std::string blanks(15, ' ');
    if(sort)
    {
        std::vector<std::pair<ProblemKey, SolutionNodeVec>> sortVec;
        for(auto& [key, value] : writing_map)
            sortVec.push_back(std::make_pair(key, value));

        // sort !
        std::sort(sortVec.begin(), sortVec.end(), ProbSolCmp);

        for(auto& [key, value] : sortVec)
        {
            ss << COMMA;
            ss << "{" << FieldDescriptor<ProblemKey>().describe("Problem", key) << "," << std::endl
               << "  "
               << VectorFieldDescriptor<SolutionNode>().describe("Solutions", value, true, blanks)
               << "}" << std::endl;

            if(COMMA.size() == 0)
                COMMA = ",";
        }
    }
    else
    {
        for(auto& [key, value] : writing_map)
        {
            ss << COMMA;
            ss << "{" << FieldDescriptor<ProblemKey>().describe("Problem", key) << "," << std::endl
               << "  "
               << VectorFieldDescriptor<SolutionNode>().describe("Solutions", value, true, blanks)
               << "}" << std::endl;

            if(COMMA.size() == 0)
                COMMA = ",";
        }
    }

    ss << "]" << std::endl;

    outfile << ss.str();
    outfile.close();

    return true;
}

bool solution_map::merge_solutions_from_file(const fs::path&                src_file,
                                             const std::vector<ProblemKey>& root_probs)
{
    bool check_dup       = true;
    bool read_to_primary = false;
    if(read_solution_map_data(src_file, read_to_primary) == false)
        return false;

    // An important note is that we can't use SolutionNode& (alias) for the second arg,
    // since a node may be shared with others (e.g. RTRT, where 2 Rs shared the same one)
    // This Recur-Add-Solution is to add a sub-tree from mapA to mapB, so what we are doing
    // here is not just moving but also updating the "option_id" of a node's child (children
    // are also moved from mapA to mapB). If we pass by reference and update the option_id,
    // Then we also change the data of an "un-moved" node (in mapA). So using call by copy
    // is safer here.
    auto RecursivelyAddSolution = [&](const ProblemKey& key,
                                      SolutionNode      solution,
                                      bool              isRoot,
                                      bool              from_primary,
                                      bool              to_primary,
                                      auto&&            RecursivelyAddSolution) -> size_t {
        std::string archName = key.arch;
        for(auto& child : solution.solution_childnodes)
        {
            ProblemKey childKey(archName, child.child_token);

            // get the child solution object from current solution map (using existing child_option)
            auto& childSol = get_solution_node(childKey, child.child_option, from_primary);

            // since we are add solution to another solution map , so we have to update the child option
            child.child_option = RecursivelyAddSolution(
                childKey, childSol, false, from_primary, to_primary, RecursivelyAddSolution);
        }
        return add_solution(key, solution, isRoot, check_dup, to_primary);
    };

    // for each root-problem, adding the whole solution-tree. from one map to another map
    for(auto& rootProbKey : root_probs)
    {
        bool isRootProb     = true;
        bool getFromPrimary = read_to_primary;
        bool addToPrimary   = !getFromPrimary;

        SolutionNode& sol_node = get_solution_node(rootProbKey, 0, getFromPrimary);

        size_t option_id = RecursivelyAddSolution(rootProbKey,
                                                  sol_node,
                                                  isRootProb,
                                                  getFromPrimary,
                                                  addToPrimary,
                                                  RecursivelyAddSolution);

        // we are adding solution for a root-problem, so the root-solution should always be at index 0
        if(option_id != 0)
            throw std::runtime_error(
                "Merge failed. Inserting solution of a root-problem should return an index 0");
    }

    return true;
}
