/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
#include "../../include/plan.h"
#include "../../include/radix_table.h"
#include "../../include/tree_node.h"
#include "rocfft.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

#include "generator.kernel.hpp"
#include "generator.param.h"
#include "generator.pass.hpp"
#include "generator.stockham.h"

using namespace StockhamGenerator;

/* =====================================================================
            Initial parameter used to generate kernels
=================================================================== */

rocfft_status initParams(FFTKernelGenKeyParams& params,
                         std::vector<size_t>    fft_N,
                         bool                   blockCompute,
                         BlockComputeType       blockComputeType)
{
    /* =====================================================================
      Parameter : basic plan info
     =================================================================== */

    params.blockCompute = blockCompute;

    params.blockComputeType = blockComputeType;

    // bool real_transform = ((params.fft_inputLayout == rocfft_array_type_real)
    // || (params.fft_outputLayout == rocfft_array_type_real));

    /* =====================================================================
      Parameter : dimension
     =================================================================== */

    params.fft_DataDim = fft_N.size() + 1;

    // TODO: fft_N does not need to know the other dimension
    for(int i = 0; i < fft_N.size(); i++)
    {
        params.fft_N[i] = fft_N[i];
    }

    params.fft_N[0] = fft_N[0];
    /* =====================================================================
      Parameter: forward, backward scale
     =================================================================== */

    double forwardScale  = 1.0;
    double backwardScale = 1.0;

    params.fft_fwdScale  = forwardScale;
    params.fft_backScale = backwardScale;

    /* =====================================================================
      Parameter: real FFT
     =================================================================== */

    // Real-Complex simple flag
    // if this is set we do real to-and-from full complex using simple algorithm
    // where imaginary of input is set to zero in forward and imaginary not
    // written in backward
    bool RCsimple = false;
    // Real FFT special flag
    // if this is set it means we are doing the 4th step in the 5-step real FFT
    // breakdown algorithm
    bool realSpecial = false;
    // size_t realSpecial_Nr; // this value stores the logical column height (N0)
    // of matrix in the 4th step
    // length[1] should be 1 + N0/2

    params.fft_RCsimple    = RCsimple;
    params.fft_realSpecial = realSpecial;
    // params.fft_realSpecial_Nr = realSpecial_Nr;

    // do twiddle scaling at the beginning pass
    bool twiddleFront       = false;
    params.fft_twiddleFront = twiddleFront;

    /* =====================================================================
      Parameter :  grid and thread blocks (work groups, work items)
                   wgs: work (w) group (g) size (s)
                   nt: number (n) of transforms (t)
                   t_ : temporary
     =================================================================== */

    size_t wgs, nt;
    size_t t_wgs, t_nt;

    KernelCoreSpecs kcs;
    kcs.GetWGSAndNT(params.fft_N[0], t_wgs, t_nt);

    if((t_wgs != 0) && (t_nt != 0) && (MAX_WORK_GROUP_SIZE >= 256))
    {
        wgs = t_wgs;
        nt  = t_nt;
    }
    else
    {
        // determine wgs, nt if not in the predefined table
        DetermineSizes(params.fft_N[0], wgs, nt);
    }

    assert((nt * params.fft_N[0]) >= wgs);
    assert((nt * params.fft_N[0]) % wgs == 0);

    params.fft_numTrans      = nt;
    params.fft_workGroupSize = wgs;

    return rocfft_status_success;
}

/* =====================================================================
   Write butterfly device function to *.h file
=================================================================== */
void WriteButterflyToFile(std::string& str, int LEN)
{

    std::ofstream file;
    std::string   fileName = "rocfft_butterfly_" + std::to_string(LEN) + ".h";
    file.open(fileName);

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }

    file << str;
    file.close();
}

/* =====================================================================
   Write CPU functions (launching kernel) header to file
=================================================================== */
void WriteCPUHeaders(const std::vector<size_t>&                                    support_list,
                     const std::vector<std::tuple<size_t, ComputeScheme>>&         large1D_list,
                     const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D_single,
                     const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D_double,
                     bool                                                          buildSingle,
                     bool                                                          buildDouble)
{
    const std::string str_interface = "(const void *data_p, void *back_p);\n";

    std::string str;

    str += "\n";
    str += "#pragma once\n";
    str += "#if !defined( kernel_launch_generator_H )\n";
    str += "#define kernel_launch_generator_H \n";

    str += "//generated CPU function headers which call GPU kernels\n";
    str += "\n";
    str += "extern \"C\"\n";
    str += "{\n";

    // write small 1D kernels, single precision
    str += "\n // 1D small size, single precision\n";
    for(size_t i = 0; i < support_list.size() && buildSingle; i++)
    {
        std::string str_len = std::to_string(support_list[i]);
        str += "void rocfft_internal_dfn_sp_ci_ci_stoc_";
        str += str_len + str_interface;
    }

    // write small 1D kernels, double precision
    str += "\n // 1D small size, double precision\n";
    for(size_t i = 0; i < support_list.size() && buildDouble; i++)
    {
        std::string str_len = std::to_string(support_list[i]);
        str += "void rocfft_internal_dfn_dp_ci_ci_stoc_";
        str += str_len + str_interface;
    }

    // write large 1D kernels, single precision
    str += "\n // 1D large size, single precision\n";
    for(size_t i = 0; i < large1D_list.size() && buildSingle; i++)
    {
        auto          my_tuple = large1D_list[i];
        auto          len      = std::get<0>(my_tuple);
        std::string   str_len  = std::to_string(len);
        ComputeScheme scheme   = std::get<1>(my_tuple);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            str += "void rocfft_internal_dfn_sp_ci_ci_sbcc_";
            str += str_len + str_interface;
        }
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_";
            str += str_len + str_interface;
            if(is_diagonal_sbrc_3D_length(len))
            {
                str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_";
                str += str_len + str_interface;
                str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_";
                str += str_len + str_interface;
                str += "void rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_";
                str += str_len + str_interface;
            }
        }
    }

    // write large 1D kernels, double precision
    str += "\n // 1D large size, double precision\n";
    for(size_t i = 0; i < large1D_list.size() && buildDouble; i++)
    {
        auto          my_tuple = large1D_list[i];
        auto          len      = std::get<0>(my_tuple);
        std::string   str_len  = std::to_string(len);
        ComputeScheme scheme   = std::get<1>(my_tuple);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            str += "void rocfft_internal_dfn_dp_ci_ci_sbcc_";
            str += str_len + str_interface;
        }
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_";
            str += str_len + str_interface;
            str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_";
            str += str_len + str_interface;
            if(is_diagonal_sbrc_3D_length(len))
            {
                str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_";
                str += str_len + str_interface;
                str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_";
                str += str_len + str_interface;
                str += "void rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_";
                str += str_len + str_interface;
            }
        }
    }

    str += "\n";
    // write 2d fused single, empty if not buildSingle
    str += "\n // 2D size, single precision\n";
    for(const auto& kernel : list_2D_single)
    {
        std::string str_len_1 = std::to_string(std::get<0>(kernel));
        std::string str_len_2 = std::to_string(std::get<1>(kernel));

        std::string suffix = str_len_1;
        suffix += "_";
        suffix += str_len_2;
        suffix += "(const void *data_p, void *back_p);\n";

        ComputeScheme scheme = std::get<2>(kernel);
        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            str += "void rocfft_internal_dfn_sp_ci_ci_2D_" + suffix;
        }
    }

    // write 2d fused double, empty if not buildDouble
    str += "\n // 2D size, double precision\n";
    for(const auto& kernel : list_2D_double)
    {
        std::string str_len_1 = std::to_string(std::get<0>(kernel));
        std::string str_len_2 = std::to_string(std::get<1>(kernel));

        std::string suffix = str_len_1;
        suffix += "_";
        suffix += str_len_2;
        suffix += "(const void *data_p, void *back_p);\n";

        ComputeScheme scheme = std::get<2>(kernel);
        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            str += "void rocfft_internal_dfn_dp_ci_ci_2D_" + suffix;
        }
    }

    str += "\n";
    str += "}\n";

    str += "\n";
    str += "#endif";

    std::ofstream file;
    std::string   fileName = "kernel_launch_generator.h";
    file.open(fileName);

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }
    file << str;
    file.close();
}

/* =====================================================================
   Write CPU functions (launching a single kernel)
   implementation to *.cpp.h file for small sizes
=================================================================== */
void write_cpu_function_small(std::vector<size_t> support_list,
                              std::string         precision,
                              int                 group_num)
{
    if(support_list.size() < group_num)
    {
        std::cout << "Not enough kernels to generate with " << group_num
                  << " groups. Some group headers might be empty" << std::endl;
    }

    std::string large_case_precision = "SINGLE";
    std::string short_name_precision = "sp";

    std::string complex_case_precision = "float2";

    if(precision == "double")
    {
        large_case_precision   = "DOUBLE";
        short_name_precision   = "dp";
        complex_case_precision = "double2";
    }

    size_t group_size = (support_list.size() + group_num - 1) / group_num;
    for(size_t j = 0; j < group_num; j++)
    {
        size_t remaining_groups = group_num - j - 1;
        // if we're out of sizes, write an empty file to keep cmake happy
        if(support_list.empty())
            group_size = 0;
        // Just put one size in each group if there are enough groups
        // left
        else if(support_list.size() <= remaining_groups)
            group_size = 1;
        // if the remaining sizes is less than the group_size (last non-empty group)
        // be sure to fit the size...or the loop would access the unwanted entries.
        else if(support_list.size() < group_size)
            group_size = support_list.size();

        size_t i_start = 0;
        size_t i_end   = group_size;

        std::string str;

        str += "\n";
        str += "#include \"kernel_launch.h\" \n"; // kernel_launch.h has the
        // required macros
        str += "\n";
        for(size_t i = i_start; i < i_end; i++)
        {

            std::string str_len = std::to_string(support_list[i]);

            str += "#include \"rocfft_kernel_" + str_len + ".h\" \n";
        }

        str += "\n";

        str += "//" + precision + " precision \n";
        for(size_t i = i_start; i < i_end; i++)
        {

            std::string str_len = std::to_string(support_list[i]);
            str += "POWX_SMALL_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_ci_ci_stoc_" + str_len + ", fft_fwd_ip_len" + str_len + ", fft_back_ip_len"
                   + str_len + ", fft_fwd_op_len" + str_len + ", fft_back_op_len" + str_len + ", "
                   + complex_case_precision + ")\n";
        }

        std::ofstream file;
        std::string   headerFileName
            = "kernel_launch_" + precision + "_" + std::to_string(j) + ".cpp.h";
        file.open(headerFileName);

        if(!file.is_open())
        {
            std::cout << "File: " << headerFileName << " could not be opened, exiting ...."
                      << std::endl;
        }
        file << str;
        file.close();

        std::string sourceFileName
            = "kernel_launch_" + precision + "_" + std::to_string(j) + ".cpp";
        file.open(sourceFileName);
        if(!file.is_open())
        {
            std::cout << "File: " << sourceFileName << " could not be opened, exiting ...."
                      << std::endl;
        }
        file << "#include \"" << headerFileName << "\"";
        file.close();

        // remove the sizes that we wrote
        // the min() should be un-neccessary after we checked "if (support_list.size() < group_size)
        support_list.erase(support_list.begin(),
                           support_list.begin() + std::min(group_size, support_list.size()));
    }
}

/* =====================================================================
   Write CPU functions (launching multiple kernels to finish a transformation)
   to *.cpp.h file for large sizes
=================================================================== */

void write_cpu_function_large(std::vector<std::tuple<size_t, ComputeScheme>> large1D_list,
                              std::string                                    precision)
{
    if(large1D_list.size() == 0)
    {
        std::cout << "No large 1D kernels to generate." << std::endl;
        return;
    }

    std::string str;

    std::string complex_case_precision = "float2";
    std::string short_name_precision   = "sp";

    if(precision == "double")
    {
        complex_case_precision = "double2";
        short_name_precision   = "dp";
    }

    str += "\n";
    str += "#include \"kernel_launch.h\" \n"; // kernel_launch.h has the required
    // macros
    str += "\n";

    str += "//" + precision + " precision \n";
    str += "\n";

    for(size_t i = 0; i < large1D_list.size(); i++)
    {

        auto          my_tuple = large1D_list[i];
        auto          len      = std::get<0>(my_tuple);
        std::string   str_len  = std::to_string(len);
        ComputeScheme scheme   = std::get<1>(my_tuple);

        std::string name_suffix;

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            name_suffix = "_sbcc";
            str += "#include \"rocfft_kernel_" + str_len + name_suffix + ".h\" \n";
            str += "POWX_LARGE_SBCC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_ci_ci_sbcc_" + str_len + ", fft_fwd_ip_len" + str_len + name_suffix
                   + ", fft_back_ip_len" + str_len + name_suffix + ", fft_fwd_op_len" + str_len
                   + name_suffix + ", fft_back_op_len" + str_len + name_suffix + ", "
                   + complex_case_precision + ")\n";
        }
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            name_suffix = "_sbrc";
            str += "#include \"rocfft_kernel_" + str_len + name_suffix + ".h\" \n";
            str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_op_ci_ci_sbrc_" + str_len + ", fft_fwd_op_len" + str_len + name_suffix
                   + ", fft_back_op_len" + str_len + name_suffix + ", " + complex_case_precision
                   + ", SBRC_2D, TILE_ALIGNED)\n";
            str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_" + str_len + ", fft_fwd_op_len"
                   + str_len + name_suffix + ", fft_back_op_len" + str_len + name_suffix + ", "
                   + complex_case_precision + ", SBRC_3D_FFT_TRANS_XY_Z, TILE_ALIGNED)\n";
            str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_" + str_len + ", fft_fwd_op_len"
                   + str_len + name_suffix + ", fft_back_op_len" + str_len + name_suffix + ", "
                   + complex_case_precision + ", SBRC_3D_FFT_TRANS_Z_XY, TILE_ALIGNED)\n";
            str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                   + "_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_" + str_len
                   + ", fft_fwd_op_len" + str_len + name_suffix + ", fft_back_op_len" + str_len
                   + name_suffix + ", " + complex_case_precision
                   + ", SBRC_3D_FFT_ERC_TRANS_Z_XY, TILE_ALIGNED)\n";
            // add diagonal transpose if possible
            if(is_diagonal_sbrc_3D_length(len))
            {
                str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                       + "_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_" + str_len + ", fft_fwd_op_len"
                       + str_len + name_suffix + ", fft_back_op_len" + str_len + name_suffix + ", "
                       + complex_case_precision + ", SBRC_3D_FFT_TRANS_XY_Z, DIAGONAL)\n";
                str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                       + "_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_" + str_len + ", fft_fwd_op_len"
                       + str_len + name_suffix + ", fft_back_op_len" + str_len + name_suffix + ", "
                       + complex_case_precision + ", SBRC_3D_FFT_TRANS_Z_XY, DIAGONAL)\n";
                str += "POWX_LARGE_SBRC_GENERATOR( rocfft_internal_dfn_" + short_name_precision
                       + "_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_" + str_len
                       + ", fft_fwd_op_len" + str_len + name_suffix + ", fft_back_op_len" + str_len
                       + name_suffix + ", " + complex_case_precision
                       + ", SBRC_3D_FFT_ERC_TRANS_Z_XY, DIAGONAL)\n";
            }
        }
    }

    std::ofstream file;
    std::string   headerFileName = "kernel_launch_" + precision + "_large.cpp.h";
    file.open(headerFileName);

    if(!file.is_open())
    {
        std::cout << "File: " << headerFileName << " could not be opened, exiting ...."
                  << std::endl;
    }
    file << str;
    file.close();

    std::string sourceFileName = "kernel_launch_" + precision + "_large.cpp";
    file.open(sourceFileName);
    if(!file.is_open())
    {
        std::cout << "File: " << sourceFileName << " could not be opened, exiting ...."
                  << std::endl;
    }
    file << "#include \"" << headerFileName << "\"";
    file.close();
}

/* =====================================================================
   Write CPU functions for launching fused 2D kernels to *.cpp.h
=================================================================== */
// split fused kernels into separate files
std::string get_2D_type(const std::tuple<size_t, size_t, ComputeScheme>& dim)
{
    // power of 2
    if(IsPo2(std::get<0>(dim)) && IsPo2(std::get<1>(dim)))
    {
        return "pow2";
    }
    // power of 3
    else if(IsPow<3>(std::get<0>(dim)) && IsPow<3>(std::get<1>(dim)))
    {
        return "pow3";
    }
    // power of 5
    else if(IsPow<5>(std::get<0>(dim)) && IsPow<5>(std::get<1>(dim)))
    {
        return "pow5";
    }
    // mixed pow2+pow3
    else if(IsPo2(std::get<0>(dim)) && IsPow<3>(std::get<1>(dim)))
    {
        return "mix_pow2_3";
    }
    // mixed pow3+pow2
    else if(IsPow<3>(std::get<0>(dim)) && IsPo2(std::get<1>(dim)))
    {
        return "mix_pow3_2";
    }
    // mixed pow3+pow5
    else if(IsPow<3>(std::get<0>(dim)) && IsPow<5>(std::get<1>(dim)))
    {
        return "mix_pow3_5";
    }
    // mixed pow5+pow3
    else if(IsPow<5>(std::get<0>(dim)) && IsPow<3>(std::get<1>(dim)))
    {
        return "mix_pow5_3";
    }
    // mixed pow2+pow5
    else if(IsPo2(std::get<0>(dim)) && IsPow<5>(std::get<1>(dim)))
    {
        return "mix_pow2_5";
    }
    // mixed pow5+pow2
    else if(IsPow<5>(std::get<0>(dim)) && IsPo2(std::get<1>(dim)))
    {
        return "mix_pow5_2";
    }
    // not implemented, fail the build
    abort();
}

std::ofstream& open_2D_file(const std::tuple<size_t, size_t, ComputeScheme>& dim,
                            const std::string&                               precision,
                            std::map<std::string, std::ofstream>&            files)
{
    std::string type = get_2D_type(dim);

    std::string    headerFileName = "kernel_launch_" + precision + "_2D_" + type + ".cpp.h";
    auto           result         = files.emplace(type, headerFileName);
    std::ofstream& file           = result.first->second;

    // if it was newly opened, initialize the file
    if(result.second)
    {
        if(!file.is_open())
        {
            // can't continue, fail the build
            std::cout << "Failed to open " << headerFileName << " for writing, aborting\n";
            abort();
        }
        file << "#include \"kernel_launch.h\"\n";

        // write source file to include this header
        std::string   sourceFileName = "kernel_launch_" + precision + "_2D_" + type + ".cpp";
        std::ofstream sourceFile(sourceFileName);
        if(!file.is_open())
        {
            // fail build
            std::cout << "File: " << sourceFileName << " could not be opened, exiting ...."
                      << std::endl;
            abort();
        }
        sourceFile << "#include \"" << headerFileName << "\"";
    }
    return file;
}

void write_cpu_function_2D(const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D,
                           const std::string&                                            precision)
{
    std::string complex_case_precision = "float2";
    std::string short_name_precision   = "sp";

    if(precision == "double")
    {
        complex_case_precision = "double2";
        short_name_precision   = "dp";
    }

    std::map<std::string, std::ofstream> files;
    for(const auto& kernel : list_2D)
    {
        std::ofstream& file          = open_2D_file(kernel, precision, files);
        std::string    str_len_1     = std::to_string(std::get<0>(kernel));
        std::string    str_len_2     = std::to_string(std::get<1>(kernel));
        std::string    length_suffix = "_2D_" + str_len_1 + "_" + str_len_2;

        file << "#include \"rocfft_kernel" << length_suffix << ".h\"\n";

        ComputeScheme scheme = std::get<2>(kernel);
        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            // reuse the POWX_SMALL_GENERATOR because we're ultimately
            // calling those kernels in the same way
            file << "POWX_SMALL_GENERATOR(rocfft_internal_dfn_" << short_name_precision << "_ci_ci"
                 << length_suffix << ", fft_fwd_ip" << length_suffix << ", fft_back_ip"
                 << length_suffix << ", fft_fwd_op" << length_suffix << ", fft_back_op"
                 << length_suffix << ", " << complex_case_precision << ")\n";
        }
        else
        {
            // not implemented yet
            abort();
        }
    }
}

/* =====================================================================
   Add CPU funtions to function pools (a hash map)
=================================================================== */
void AddCPUFunctionToPool(
    const std::vector<size_t>&                                    support_list,
    const std::vector<std::tuple<size_t, ComputeScheme>>&         large1D_list,
    const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& support_list_2D_single,
    const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& support_list_2D_double,
    bool                                                          buildSingle,
    bool                                                          buildDouble)
{
    std::string str;

    str += "\n";
    str += "#include <iostream> \n";
    str += "#include \"../include/function_pool.h\" \n";
    str += "#include \"kernel_launch_generator.h\" \n";
    str += "\n";
    str += "//build hash map to store the function pointers\n";
    str += "function_pool::function_pool()\n";

    // write small 1D kernels single
    str += "{\n";
    str += "\t//small 1D, single precision \n";
    for(size_t i = 0; i < support_list.size() && buildSingle; i++)
    {
        std::string str_len = std::to_string(support_list[i]);
        str += "\tfunction_map_single[std::make_pair(" + str_len
               + ",CS_KERNEL_STOCKHAM)] = &rocfft_internal_dfn_sp_ci_ci_stoc_";
        str += str_len + ";\n";
    }

    // write small 1D kernels double
    str += "\n";
    str += "\t//small 1D, double precision \n";
    for(size_t i = 0; i < support_list.size() && buildDouble; i++)
    {
        std::string str_len = std::to_string(support_list[i]);
        str += "\tfunction_map_double[std::make_pair(" + str_len
               + ",CS_KERNEL_STOCKHAM)] = &rocfft_internal_dfn_dp_ci_ci_stoc_";
        str += str_len + ";\n";
    }

    str += "\n";

    // write larege 1D kernels single
    str += "\n";
    str += "\t//large 1D, single precision \n";
    for(size_t i = 0; i < large1D_list.size() && buildSingle; i++)
    {
        auto          my_tuple = large1D_list[i];
        auto          len      = std::get<0>(my_tuple);
        std::string   str_len  = std::to_string(len);
        ComputeScheme scheme   = std::get<1>(my_tuple);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            str += "\tfunction_map_single[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_BLOCK_CC)] = "
                     "&rocfft_internal_dfn_sp_ci_ci_sbcc_"
                   + str_len + ";\n";
            ;
        }
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            str += "\tfunction_map_single[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_BLOCK_RC)] = "
                     "&rocfft_internal_dfn_sp_op_ci_ci_sbrc_"
                   + str_len + ";\n";
            // For every SBRC kernel, also generate one that fuses
            // transpose for 3D transforms
            str += "\tfunction_map_single_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)] = "
                     "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_"
                   + str_len + ";\n";
            str += "\tfunction_map_single_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)] = "
                     "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_"
                   + str_len + ";\n";
            str += "\tfunction_map_single_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)] = "
                     "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_"
                   + str_len + ";\n";
            // add diagonal transpose if possible
            if(is_diagonal_sbrc_3D_length(len))
            {
                str += "\tfunction_map_single_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)] = "
                         "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_"
                       + str_len + ";\n";
                str += "\tfunction_map_single_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)] = "
                         "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_"
                       + str_len + ";\n";
                str += "\tfunction_map_single_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)] = "
                         "&rocfft_internal_dfn_sp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_"
                       + str_len + ";\n";
            }
        }
    }

    // write large 1D kernels double
    str += "\n";
    str += "\t//large 1D, double precision \n";
    for(size_t i = 0; i < large1D_list.size() && buildDouble; i++)
    {
        auto          my_tuple = large1D_list[i];
        auto          len      = std::get<0>(my_tuple);
        std::string   str_len  = std::to_string(len);
        ComputeScheme scheme   = std::get<1>(my_tuple);

        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        {
            str += "\tfunction_map_double[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_BLOCK_CC)] = "
                     "&rocfft_internal_dfn_dp_ci_ci_sbcc_"
                   + str_len + ";\n";
            ;
        }
        else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
        {
            str += "\tfunction_map_double[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_BLOCK_RC)] = "
                     "&rocfft_internal_dfn_dp_op_ci_ci_sbrc_"
                   + str_len + ";\n";
            // For every SBRC kernel, also generate one that fuses
            // transpose for 3D transforms
            str += "\tfunction_map_double_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)] = "
                     "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_"
                   + str_len + ";\n";
            str += "\tfunction_map_double_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)] = "
                     "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_"
                   + str_len + ";\n";
            str += "\tfunction_map_double_transpose_tile_aligned[std::make_pair(" + str_len
                   + ", CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)] = "
                     "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_"
                   + str_len + ";\n";
            // add diagonal transpose if possible
            if(is_diagonal_sbrc_3D_length(len))
            {
                str += "\tfunction_map_double_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)] = "
                         "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_"
                       + str_len + ";\n";
                str += "\tfunction_map_double_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)] = "
                         "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_"
                       + str_len + ";\n";
                str += "\tfunction_map_double_transpose_diagonal[std::make_pair(" + str_len
                       + ", CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)] = "
                         "&rocfft_internal_dfn_dp_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_"
                       + str_len + ";\n";
            }
        }
    }

    // write 2D kernels single
    str += "\n";
    str += "\t//2D, single precision \n";
    for(const auto& kernel : support_list_2D_single)
    {
        std::string   str_len_1 = std::to_string(std::get<0>(kernel));
        std::string   str_len_2 = std::to_string(std::get<1>(kernel));
        ComputeScheme scheme    = std::get<2>(kernel);
        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            str += "\tfunction_map_single_2D[std::make_tuple(" + str_len_1 + ", " + str_len_2
                   + ", CS_KERNEL_2D_SINGLE)] = "
                     "&rocfft_internal_dfn_sp_ci_ci_2D_"
                   + str_len_1 + "_" + str_len_2 + ";\n";
        }
        else
        {
            // not implemented yet!
            abort();
        }
    }

    // write 2D kernels double
    str += "\n";
    str += "\t//2D, double precision \n";
    for(const auto& kernel : support_list_2D_double)
    {
        std::string   str_len_1 = std::to_string(std::get<0>(kernel));
        std::string   str_len_2 = std::to_string(std::get<1>(kernel));
        ComputeScheme scheme    = std::get<2>(kernel);
        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            str += "\tfunction_map_double_2D[std::make_tuple(" + str_len_1 + ", " + str_len_2
                   + ", CS_KERNEL_2D_SINGLE)] = "
                     "&rocfft_internal_dfn_dp_ci_ci_2D_"
                   + str_len_1 + "_" + str_len_2 + ";\n";
        }
        else
        {
            // not implemented yet!
            abort();
        }
    }

    str += "}\n";

    std::ofstream file;
    std::string   headerFileName = "function_pool.cpp.h";
    file.open(headerFileName);

    if(!file.is_open())
    {
        std::cout << "File: " << headerFileName << " could not be opened, exiting ...."
                  << std::endl;
    }
    file << str;
    file.close();

    std::string sourceFileName = "function_pool.cpp";
    file.open(sourceFileName);
    if(!file.is_open())
    {
        std::cout << "File: " << sourceFileName << " could not be opened, exiting ...."
                  << std::endl;
    }
    file << "#include \"" << headerFileName << "\"";
    file.close();
}

/* =====================================================================
    Ggenerate the kernels and write to *.h files
=================================================================== */

void WriteKernelToFile(std::string& str, std::string LEN)
{

    std::ofstream file;
    std::string   fileName = "rocfft_kernel_" + LEN + ".h";
    file.open(fileName);

    if(!file.is_open())
    {
        std::cout << "File: " << fileName << " could not be opened, exiting ...." << std::endl;
    }

    // multiple include protection
    file << "#pragma once\n";

    file << str;
    file.close();
}

void generate_kernel(size_t len, ComputeScheme scheme)
{
    std::string           programCode;
    FFTKernelGenKeyParams params;

    if(scheme == CS_KERNEL_STOCKHAM) // for small size
    {
        std::vector<size_t> fft_N(1);
        fft_N[0] = len;
        initParams(params, fft_N, false, BCT_C2C); // here the C2C is not enabled,
        // as the third parameter is set
        // as false

        Kernel<rocfft_precision_single> kernel(
            params); // generate data type template kernels regardless of precision
        kernel.GenerateKernel(programCode);

        WriteKernelToFile(programCode, std::to_string(len));
    }
    else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
    {
        // length of the FFT in each dimension, <= 3
        // generate different combinations, like 8192=64(C2C)*128(R2C).
        // 32768=128(C2C)*256(R2C), notice,128(C2C) != 128(R2C)
        // the first dim is always type C2C with fft_2StepTwiddle true (1), the
        // second is always R2C with fft_2StepTwiddle false (0)
        bool                blockCompute = true; // enable blockCompute in large 1D
        std::vector<size_t> fft_N        = {1, 1};
        // generate C2C type kernels
        fft_N[0]                = len;
        params.fft_3StepTwiddle = true;
        params.name_suffix      = "_sbcc";
        initParams(params, fft_N, blockCompute, BCT_C2C);

        Kernel<rocfft_precision_single> kernel(
            params); // generate data type template kernels regardless of precision
        kernel.GenerateKernel(programCode);

        WriteKernelToFile(programCode, std::to_string(len) + params.name_suffix);
    }
    else if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
    {
        bool                blockCompute = true; // enable blockCompute in large 1D
        std::vector<size_t> fft_N        = {1, 1};
        // generate R2C type kernels
        fft_N[0]                = len;
        params.fft_3StepTwiddle = false;
        params.name_suffix      = "_sbrc";
        initParams(params, fft_N, blockCompute, BCT_R2C);

        Kernel<rocfft_precision_single> kernel(
            params); // generate data type template kernels regardless of precision
        kernel.GenerateKernel(programCode);

        WriteKernelToFile(programCode, std::to_string(len) + params.name_suffix);
    }
}
void generate_2D_kernels(const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& kernels)
{
    for(const auto& kernel : kernels)
    {
        std::string   programCode;
        size_t        len1   = std::get<0>(kernel);
        size_t        len2   = std::get<1>(kernel);
        ComputeScheme scheme = std::get<2>(kernel);

        // if we were able to insert, this size must be new
        programCode += "#include \"rocfft_kernel_" + std::to_string(len1) + ".h\"\n";
        if(len1 != len2)
            programCode += "#include \"rocfft_kernel_" + std::to_string(len2) + ".h\"\n";

        if(scheme == CS_KERNEL_2D_SINGLE)
        {
            // parameters for each dimension
            FFTKernelGenKeyParams params1;
            FFTKernelGenKeyParams params2;
            // column-by-column transform can't possibly be unit stride
            params2.forceNonUnitStride = true;

            std::vector<size_t> fft_N(1, len1);
            // here the C2C is not enabled,
            // as the third parameter is set
            // as false
            initParams(params1, fft_N, false, BCT_C2C);
            fft_N.front() = len2;
            initParams(params2, fft_N, false, BCT_C2C);

            Kernel2D kernel(params1, params2);
            kernel.GenerateGlobalKernel(programCode);

            std::string file_suffix = "2D_" + std::to_string(len1) + "_" + std::to_string(len2);
            WriteKernelToFile(programCode, file_suffix);
        }
        else
        {
            // not handled yet
            abort();
        }
    }
}
