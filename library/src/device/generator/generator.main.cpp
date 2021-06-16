/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#include "../../include/radix_table.h"
#include "../../include/tree_node.h"
#include "rocfft.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>

#include "generator.argument.hpp"
#include "generator.butterfly.hpp"
#include "generator.file.h"
#include "generator.kernel.hpp"
#include "generator.options_util.hpp"
#include "generator.param.h"
#include "generator.pass.hpp"
#include "generator.stockham.h"

using namespace StockhamGenerator;

// Returns 1 for single-precision, 2 for double precision
inline size_t PrecisionWidth(rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return 1;
    case rocfft_precision_double:
        return 2;
    default:
        assert(false);
        return 1;
    }
}

inline size_t Large1DThreshold(rocfft_precision precision)
{
    return 4096 / PrecisionWidth(precision);
}

generator_argument argument;

/* =====================================================================
    Ggenerate the support size according to the lower and upper bound
=================================================================== */
int generate_support_size_list(std::set<size_t>& support_size_set,
                               size_t            i_upper_bound,
                               size_t            j_upper_bound,
                               size_t            k_upper_bound,
                               bool              includeCommon)
{
    int    counter     = 0;
    size_t upper_bound = std::max(std::max(i_upper_bound, j_upper_bound), k_upper_bound);
    for(size_t i = 1; i <= i_upper_bound; i *= 5)
    {
        for(size_t j = 1; j <= j_upper_bound; j *= 3)
        {
            for(size_t k = 1; k <= k_upper_bound; k *= 2)
            {
                {
                    if(i * j * k <= upper_bound)
                    {
                        counter++;
                        // printf("Item %d: %d * %d * %d  = %d is below %d \n",
                        // (int)counter, (int)i, (int)j, (int)k, i*j*k, upper_bound);
                        size_t len = i * j * k;
                        support_size_set.insert(len);
                    }
                }
            }
        }
    }

    if(includeCommon)
    {
        // pick relatively common radix-7 sizes - radix-7 in general is
        // not common enough to justify generating every combination
        support_size_set.insert(7);
        support_size_set.insert(14);
        support_size_set.insert(21);
        support_size_set.insert(28);
        support_size_set.insert(49);
        support_size_set.insert(42);
        support_size_set.insert(56);
        support_size_set.insert(84);
        support_size_set.insert(112);
        support_size_set.insert(168);
        support_size_set.insert(224);
        support_size_set.insert(336);
        support_size_set.insert(343);

        // basic support for radix-11 and 13
        support_size_set.insert(11);
        support_size_set.insert(22);
        support_size_set.insert(44);
        support_size_set.insert(88);
        support_size_set.insert(121);
        support_size_set.insert(176);

        support_size_set.insert(13);
        support_size_set.insert(26);
        support_size_set.insert(52);
        support_size_set.insert(104);
        support_size_set.insert(169);
        support_size_set.insert(208);
    }

    // printf("Total, there are %d valid combinations\n", counter);
    return 0;
}

std::vector<std::tuple<size_t, size_t, ComputeScheme>>
    generate_support_size_list_2D(rocfft_precision precision)
{
    std::vector<std::tuple<size_t, size_t, ComputeScheme>> retval;
    KernelCoreSpecs                                        kcs;
    auto GetWGSAndNT = [&kcs](size_t length, size_t& workGroupSize, size_t& numTransforms) {
        return kcs.GetWGSAndNT(length, workGroupSize, numTransforms);
    };
    for(const auto& s : Single2DSizes(0, precision, GetWGSAndNT))
    {
        retval.push_back(std::make_tuple(s.first, s.second, CS_KERNEL_2D_SINGLE));
    }
    return retval;
}

std::set<size_t>
    get_dependent_1D_sizes(const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D)
{
    std::set<size_t> dependent_1D_set;
    for(auto& size2D : list_2D)
    {
        dependent_1D_set.insert(std::get<0>(size2D));
        dependent_1D_set.insert(std::get<1>(size2D));
    }
    return dependent_1D_set;
}

int main(int argc, char* argv[])
{
    // std::cout << argc << std::endl;
    // for (int i = 0; i < argc; ++i )
    //     std::cout << "[" << i << "] " << argv[i] << std::endl;
    /*
      std::string str;
      size_t rad = 10;
      for (size_t d = 0; d<2; d++)
      {
          bool fwd = d ? false : true;
          Butterfly<rocfft_precision_single> bfly1(rad, 1, fwd, true);
     bfly1.GenerateButterfly(str); str += "\n"; //TODO, does not work for 4,
     single or double precsion does not matter here.
      }
      printf("Generating rad %d butterfly \n", (int)rad);
      WriteButterflyToFile(str, rad);
      printf("===========================================================================\n");
  */

    // collection of supported 1D large sizes
    std::set<size_t> supported_large_set({50, 64, 81, 100, 128, 200, 256});

    /* =====================================================================
     Parsing Arguments
    =================================================================== */
    std::string typeArgStr;
    std::string precisionArgStr;
    std::string manualSmallArgStr;
    std::string manualLargeArgStr;
    std::string manual2DArgStr;
    std::string noSBCCArgStr;

    std::set<std::string> argStrList;

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    options_description opdesc("generator command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("type,t", value<std::string>(&typeArgStr),
         "Predefine Type (Separate by comma)")
        ("precision,p", value<std::string>(&precisionArgStr),
         "Precision (single, double, comma)")
        ("manual-small", value<std::string>(&manualSmallArgStr),
         "Manual 1D small sizes(Separate by comma)")
        ("manual-large", value<std::string>(&manualLargeArgStr),
         "Manual 1D large sizes(Separate by comma)")
        ("manual-2d", value<std::string>(&manual2DArgStr),
         "Manual 2D large sizes(Separate by comma and x)")
        ("no-sbcc", value<std::string>(&noSBCCArgStr),
         "gen large sizes with sbrc only, no sbcc (Separate by comma)")
        ("group,g", value<size_t>(&argument.group_num)->default_value(8),
         "Numbers of kernel launch cpp files for 1D small size");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, opdesc), vm);
    notify(vm);

    if(vm.count("manual-small"))
    {
        parse_arg_ints(manualSmallArgStr, argument.manualSize);
        // need to check if the sizes are supported by generator (pow2,3,5,7...)
        std::set<size_t> supported_small_set;
        generate_support_size_list(
            supported_small_set, 3125, 2187, Large1DThreshold(rocfft_precision_single), true);

        if(argument.filter_manual_small_size(supported_small_set) == 0)
        {
            std::cerr << "No valid manual small sizes!" << std::endl;
        }
    }
    if(vm.count("manual-large"))
    {
        parse_arg_ints(manualLargeArgStr, argument.manualSizeLarge);
        // need to check if the sizes are supported by generator
        if(argument.filter_manual_large_size(supported_large_set) == 0)
        {
            std::cerr << "No valid manual large sizes!" << std::endl;
        }
    }
    if(vm.count("manual-2d"))
    {
        // XXX just stick straight into valid...
        parse_arg_pairs(manual2DArgStr, argument.validManual2D);
    }
    // default type is ALL if not specified, else init_type from arg
    if(vm.count("type"))
    {
        parse_arg_strings(typeArgStr, argStrList);
        argument.init_type(argStrList);
    }
    // default precision is ALL if not specified, else init_precision from arg
    if(vm.count("precision"))
    {
        parse_arg_strings(precisionArgStr, argStrList);
        argument.init_precision(argStrList);
    }
    // default large sizes gen both sbcc and sbrc, except for those tagged with "no-sbcc"
    if(vm.count("no-sbcc"))
    {
        parse_arg_ints(noSBCCArgStr, argument.largeSizesWithoutSBCC);
    }

    if(argument.group_num <= 0)
    {
        std::cerr << "Invalid small kernels group number! Force set to 150" << std::endl;
        argument.group_num = 150;
    }

    std::cout << argument.str() << std::endl;

    if(!argument.check_valid())
    {
        return 0;
    }

    /*
      for(size_t i=7;i<=2401;i*=7){
          printf("Generating len %d FFT kernels\n", (int)i);
          generate_kernel(i);
          support_size_list.push_back(i);
      }
  */

    /* =====================================================================
     generate 1D small kernel: size, kernel *.h filse, launcher *.cpp files
    =================================================================== */
    std::vector<size_t> support_size_list;
    std::set<size_t>    all_small_sizes_to_gen; // a temp set for easier operation
    if(argument.has_predefine_type(EPredefineType::SMALL) || argument.has_manual_small_size())
    {
        // any of small type
        if(argument.has_predefine_type(EPredefineType::SMALL))
        {
            size_t pow2_bound = argument.has_predefine_type(EPredefineType::POW2)
                                    ? Large1DThreshold(rocfft_precision_single)
                                    : 1;
            size_t pow3_bound = argument.has_predefine_type(EPredefineType::POW3) ? 2187 : 1;
            size_t pow5_bound = argument.has_predefine_type(EPredefineType::POW5) ? 3125 : 1;

            generate_support_size_list(all_small_sizes_to_gen,
                                       pow5_bound,
                                       pow3_bound,
                                       pow2_bound,
                                       argument.has_predefine_type(EPredefineType::POW7));
        }

        // append manual small size (will not have duplicates since it's set)
        if(argument.has_manual_small_size())
        {
            all_small_sizes_to_gen.insert(argument.validManualSize.begin(),
                                          argument.validManualSize.end());
        }

        // convert to final vector
        support_size_list.assign(all_small_sizes_to_gen.begin(), all_small_sizes_to_gen.end());

        for(size_t i = 0; i < support_size_list.size(); i++)
        {
            // printf("Generating len %d FFT kernels\n", support_size_list[i]);
            generate_kernel(support_size_list[i], CS_KERNEL_STOCKHAM);
        }

        // printf("Wrtie small size CPU functions implemention to *.cpp files \n");
        // all the small size of the same precsion are in one single file
        if(argument.has_precision(EPrecision::SINGLE))
            write_cpu_function_small(support_size_list, "single", argument.group_num);
        if(argument.has_precision(EPrecision::DOUBLE))
            write_cpu_function_small(support_size_list, "double", argument.group_num);
    }

    /* =====================================================================
    large1D is not a single kernels but a bunch of small kernels combinations
    here we use a vector of tuple to store the supported sizes
    Initially available is 8K - 64K break into 64, 128, 256 combinations
  =================================================================== */

    std::vector<std::tuple<size_t, ComputeScheme>> large1D_list;
    if(argument.has_predefine_type(EPredefineType::LARGE) || argument.has_manual_large_size())
    {
        // --type large, gen all large
        if(argument.has_predefine_type(EPredefineType::LARGE))
        {
            for(auto i : supported_large_set)
            {
                if(argument.largeSizesWithoutSBCC.count(i) == 0)
                    large1D_list.push_back(std::make_tuple(i, CS_KERNEL_STOCKHAM_BLOCK_CC));
                large1D_list.push_back(std::make_tuple(i, CS_KERNEL_STOCKHAM_BLOCK_RC));
            }
        }
        // manual large size only,
        // we use else here since LARGE already contains all, no need to replicate
        // so do this else only if !LARGE but has_manual_large
        else /* if(argument.has_manual_large_size()) */
        {
            for(auto i : argument.validManualSizeLarge)
            {
                if(argument.largeSizesWithoutSBCC.count(i) == 0)
                    large1D_list.push_back(std::make_tuple(i, CS_KERNEL_STOCKHAM_BLOCK_CC));
                large1D_list.push_back(std::make_tuple(i, CS_KERNEL_STOCKHAM_BLOCK_RC));
            }
        }

        for(int i = 0; i < large1D_list.size(); i++)
        {
            auto my_tuple = large1D_list[i];
            generate_kernel(std::get<0>(my_tuple), std::get<1>(my_tuple));
        }

        // write big size CPU functions; one file for one size
        if(argument.has_precision(EPrecision::SINGLE))
            write_cpu_function_large(large1D_list, "single");
        if(argument.has_precision(EPrecision::DOUBLE))
            write_cpu_function_large(large1D_list, "double");
    }

    /* =====================================================================
      generate 2D fused kernels
    =================================================================== */
    std::vector<std::tuple<size_t, size_t, ComputeScheme>> support_size_list_2D;
    if(argument.has_predefine_type(EPredefineType::DIM2))
    {
        // // NOTICE: not all sizes in single_2D are supported in double_2D,
        // // for example, single 2D supports 125x25, 25x125, while double doesn't
        // // see kernel_launch_single_2D_pow5.cpp.h and kernel_launch_double_2D_pow5.cpp.h
        // if(argument.has_precision(EPrecision::SINGLE))
        //     support_size_list_2D_single = generate_support_size_list_2D(rocfft_precision_single);
        // if(argument.has_precision(EPrecision::DOUBLE))
        //     support_size_list_2D_double = generate_support_size_list_2D(rocfft_precision_double);

        // // generated code is all templated so we can generate the largest
        // // number of sizes and decide at runtime whether the
        // // double-precision variants can be used based on available LDS
        // if(argument.has_precision(EPrecision::SINGLE))
        // {
        //     generate_2D_kernels(support_size_list_2D_single);
        // }
        // // but if we want to build double "only", then single list is empty,
        // // so we build from double list
        // else
        // {
        //     generate_2D_kernels(support_size_list_2D_double);
        // }

        for(auto i : argument.validManual2D)
        {
            support_size_list_2D.push_back(std::make_tuple(i.first, i.second, CS_KERNEL_2D_SINGLE));
        }
        generate_2D_kernels(support_size_list_2D);

        // write 2D fused kernels, list could be empty if the precision is not built
        if(argument.has_precision(EPrecision::SINGLE))
            write_cpu_function_2D(support_size_list_2D, "single");
        if(argument.has_precision(EPrecision::DOUBLE))
            write_cpu_function_2D(support_size_list_2D, "double");

        // if build 2D, we need to build the dependent 1D small kernels as well
        // this is essential if not all small sizes are specified in argument
        //
        // double is the subset of single, so we only get from one of them
        auto dep_1D_sizes = get_dependent_1D_sizes(support_size_list_2D);

        for(size_t len_1D : dep_1D_sizes)
        {
            // if this dependent 1D is not generated before, generate it
            if(!all_small_sizes_to_gen.count(len_1D))
                generate_kernel(len_1D, CS_KERNEL_STOCKHAM);
        }
    }

    /* =====================================================================
      write to kernel_launch_generator.h
    =================================================================== */

    // // printf("Write CPU functions declaration to *.h file \n");
    // WriteCPUHeaders(support_size_list,
    //                 large1D_list,
    //                 support_size_list_2D_single,
    //                 support_size_list_2D_double,
    //                 argument.has_precision(EPrecision::SINGLE),
    //                 argument.has_precision(EPrecision::DOUBLE));

    // /* =====================================================================
    //   write to function_pool.cpp.h
    // =================================================================== */

    // // printf("Add CPU function into hash map \n");
    // AddCPUFunctionToPool(support_size_list,
    //                      large1D_list,
    //                      support_size_list_2D_single,
    //                      support_size_list_2D_double,
    //                      argument.has_precision(EPrecision::SINGLE),
    //                      argument.has_precision(EPrecision::DOUBLE));
}
