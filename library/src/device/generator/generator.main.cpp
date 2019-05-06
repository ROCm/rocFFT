/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#include "../../include/radix_table.h"
#include "../../include/tree_node.h"
#include "rocfft.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>

#include "generator.butterfly.hpp"
#include "generator.file.h"
#include "generator.kernel.hpp"
#include "generator.param.h"
#include "generator.pass.hpp"
#include "generator.stockham.h"

using namespace StockhamGenerator;

/* =====================================================================
    Ggenerate the support size according to the lower and upper bound
=================================================================== */

int generate_support_size_list(std::vector<size_t>& support_size_list,
                               size_t               i_upper_bound,
                               size_t               j_upper_bound,
                               size_t               k_upper_bound)
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
                        support_size_list.push_back(len);
                    }
                }
            }
        }
    }

    // printf("Total, there are %d valid combinations\n", counter);
    return 0;
}

int main(int argc, char* argv[])
{

    std::string str;
    /*
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

    std::vector<size_t> support_size_list;

    int small_kernels_group_num = 8; // default

    if(argc > 1)
    {
        if(strcmp(argv[1], "pow2") == 0)
        {
            // printf("Generating len pow2 FFT kernels\n");
            generate_support_size_list(
                support_size_list, 1, 1, Large1DThreshold(rocfft_precision_single));
        }
        else if(strcmp(argv[1], "pow3") == 0)
        {
            // printf("Generating len pow3 FFT kernels\n");
            generate_support_size_list(support_size_list, 1, 2187, 1);
        }
        else if(strcmp(argv[1], "pow5") == 0)
        {
            // printf("Generating len pow5 FFT kernels\n");
            generate_support_size_list(support_size_list, 3125, 1, 1);
        }
        else if(strcmp(argv[1], "pow2,3") == 0)
        {
            // printf("Generating len pow2 and pow3 FFT kernels\n");
            generate_support_size_list(
                support_size_list, 1, 2187, Large1DThreshold(rocfft_precision_single));
        }
        else if(strcmp(argv[1], "pow2,5") == 0)
        {
            // printf("Generating len pow2 and pow5 FFT kernels\n");
            generate_support_size_list(
                support_size_list, 3125, 1, Large1DThreshold(rocfft_precision_single));
        }
        else if(strcmp(argv[1], "pow3,5") == 0)
        {
            // printf("Generating len pow3 and pow5 FFT kernels\n");
            generate_support_size_list(support_size_list, 3125, 2187, 1);
        }
        else if(strcmp(argv[1], "all") == 0)
        {
            // printf("Generating len mix of 2,3,5 FFT kernels\n");
            generate_support_size_list(
                support_size_list, 3125, 2187, Large1DThreshold(rocfft_precision_single));
        }
    }
    else
    { // if no arguments, generate all possible sizes
        // printf("Generating len mix of 2,3,5 FFT kernels\n");
        generate_support_size_list(
            support_size_list, 3125, 2187, Large1DThreshold(rocfft_precision_single));
    }

    if(argc > 2)
    {
        small_kernels_group_num = std::stoi(argv[2]);
        if(small_kernels_group_num <= 0 || small_kernels_group_num > 128)
        {
            std::cerr << "Invalid small kernels group number!" << std::endl;
            return 0;
        }
    }

    /*
      for(size_t i=7;i<=2401;i*=7){
          printf("Generating len %d FFT kernels\n", (int)i);
          generate_kernel(i);
          support_size_list.push_back(i);
      }
  */

    /* =====================================================================
     generate small kernel into *.h file
  =================================================================== */

    for(size_t i = 0; i < support_size_list.size(); i++)
    {
        // printf("Generating len %d FFT kernels\n", support_size_list[i]);
        generate_kernel(support_size_list[i], CS_KERNEL_STOCKHAM);
    }

    // printf("Wrtie small size CPU functions implemention to *.cpp files \n");
    // all the small size of the same precsion are in one single file
    write_cpu_function_small(support_size_list, "single", small_kernels_group_num);
    write_cpu_function_small(support_size_list, "double", small_kernels_group_num);

    /* =====================================================================

    large1D is not a single kernels but a bunch of small kernels combinations
    here we use a vector of tuple to store the supported sizes
    Initially available is 8K - 64K break into 64, 128, 256 combinations
  =================================================================== */

    std::vector<std::tuple<size_t, ComputeScheme>> large1D_list;
    large1D_list.push_back(std::make_tuple(64, CS_KERNEL_STOCKHAM_BLOCK_CC));
    large1D_list.push_back(std::make_tuple(128, CS_KERNEL_STOCKHAM_BLOCK_CC));
    large1D_list.push_back(std::make_tuple(256, CS_KERNEL_STOCKHAM_BLOCK_CC));

    large1D_list.push_back(std::make_tuple(64, CS_KERNEL_STOCKHAM_BLOCK_RC));
    large1D_list.push_back(std::make_tuple(128, CS_KERNEL_STOCKHAM_BLOCK_RC));
    large1D_list.push_back(std::make_tuple(256, CS_KERNEL_STOCKHAM_BLOCK_RC));

    for(int i = 0; i < large1D_list.size(); i++)
    {
        auto my_tuple = large1D_list[i];
        generate_kernel(std::get<0>(my_tuple), std::get<1>(my_tuple));
    }

    write_cpu_function_large(large1D_list, "single");
    write_cpu_function_large(large1D_list, "double");

    // write big size CPU functions; one file for one size

    // printf("Write CPU functions declaration to *.h file \n");
    WriteCPUHeaders(support_size_list, large1D_list);

    // printf("Add CPU function into hash map \n");
    AddCPUFunctionToPool(support_size_list, large1D_list);
}
