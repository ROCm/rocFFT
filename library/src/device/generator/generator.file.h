/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(generator_file_H)
#define generator_file_H

#include "generator.param.h"

rocfft_status initParams(FFTKernelGenKeyParams& params,
                         std::vector<size_t>    fft_N,
                         bool                   blockCompute,
                         BlockComputeType       blockComputeType);

void WriteButterflyToFile(std::string& str, int LEN);

void WriteCPUHeaders(const std::vector<size_t>&                                    support_list,
                     const std::vector<std::tuple<size_t, ComputeScheme>>&         large1D_list,
                     const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D_single,
                     const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D_double,
                     bool                                                          buildSingle,
                     bool                                                          buildDouble);

void write_cpu_function_small(std::vector<size_t> support_list,
                              std::string         precision,
                              int                 group_num);

void write_cpu_function_large(std::vector<std::tuple<size_t, ComputeScheme>> large1D_list,
                              std::string                                    precision);

void write_cpu_function_2D(const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& list_2D,
                           const std::string&                                            precision);

void AddCPUFunctionToPool(
    const std::vector<size_t>&                                    support_list,
    const std::vector<std::tuple<size_t, ComputeScheme>>&         large1D_list,
    const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& support_list_2D_single,
    const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& support_list_2D_double,
    bool                                                          buildSingle,
    bool                                                          buildDouble);

void generate_kernel(size_t len, ComputeScheme scheme);

void generate_2D_kernels(const std::vector<std::tuple<size_t, size_t, ComputeScheme>>& kernels);

#endif // generator_file_H
