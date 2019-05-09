/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(generator_file_H)
#define generator_file_H

#include "generator.param.h"

extern "C" rocfft_status initParams(FFTKernelGenKeyParams& params,
                                    std::vector<size_t>    fft_N,
                                    bool                   blockCompute,
                                    BlockComputeType       blockComputeType);

extern "C" void WriteButterflyToFile(std::string& str, int LEN);

extern "C" void WriteCPUHeaders(std::vector<size_t>                            support_list,
                                std::vector<std::tuple<size_t, ComputeScheme>> large1D_list);

extern "C" void write_cpu_function_small(std::vector<size_t> support_list,
                                         std::string         precision,
                                         int                 group_num);

extern "C" void
    write_cpu_function_large(std::vector<std::tuple<size_t, ComputeScheme>> large1D_list,
                             std::string                                    precision);

extern "C" void AddCPUFunctionToPool(std::vector<size_t>                            support_list,
                                     std::vector<std::tuple<size_t, ComputeScheme>> large1D_list);

extern "C" void generate_kernel(size_t len, ComputeScheme scheme);

#endif // generator_file_H
