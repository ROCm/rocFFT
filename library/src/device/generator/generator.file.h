/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined( generator_file_H )
#define generator_file_H

extern "C"
rocfft_status initParams (FFTKernelGenKeyParams &params, std::vector<size_t> fft_N,  bool blockCompute, BlockComputeType blockComputeType);

extern "C" 
void WriteButterflyToFile(std::string &str, int LEN);

extern "C" 
void WriteCPUHeaders(std::vector<size_t> support_list);

extern "C" 
void write_cpu_function_small(std::vector<size_t> support_list, std::string precision);

extern "C" 
void write_cpu_function_large(std::vector<size_t> support_list, std::string precision);

extern "C" 
void AddCPUFunctionToPool(std::vector<size_t> support_list);

extern "C" 
void generate_kernel_small(size_t len);

extern "C" 
void generate_kernel_large(std::vector<size_t> large1D_first_dim, std::vector<size_t> large1D_second_dim);

#endif // generator_file_H

