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

void WriteCPUHeaders(std::vector<size_t>                            support_list,
                     std::vector<std::tuple<size_t, ComputeScheme>> large1D_list);

void write_cpu_function_small(std::vector<size_t> support_list,
                              std::string         precision,
                              int                 group_num);

void write_cpu_function_large(std::vector<std::tuple<size_t, ComputeScheme>> large1D_list,
                              std::string                                    precision);

void AddCPUFunctionToPool(std::vector<size_t>                            support_list,
                          std::vector<std::tuple<size_t, ComputeScheme>> large1D_list);

void generate_kernel(size_t len, ComputeScheme scheme);

#endif // generator_file_H
