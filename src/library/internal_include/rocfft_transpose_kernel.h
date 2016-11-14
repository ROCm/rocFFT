/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef __ROCFFT_TRANSPOSE_KERNEL_H__
#define __ROCFFT_TRANSPOSE_KERNEL_H__
#include <hip/hip_runtime.h>

//defined in rocfft_transpose_kernel.cu
template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t batch_size);

//defined in rocfft_transpose_kernel_check_boundary.cu
template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_check_boundary(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t batch_size);

//defined in rocfft_transpose_complex_kernel.cu
template<typename T, int micro_tile_col_size, int micor_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_complex_planar_to_complex_planar(hipLaunchParm lp,
                                                                           T *input_matrix_real,
									   T *input_matrix_imag,
									   T *output_matrix_real,
									   T *output_matrix_imag,
		              						   size_t input_row_size,
									   size_t input_col_size,
		               						   size_t input_leading_dim_size,
									   size_t output_leading_dim_size,
						                           size_t batch_size);

//defined in rocfft_transpose_complex_planar_to_interleavd_kernel.cu
template<typename T, typename T2, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_complex_planar_to_complex_interleaved(hipLaunchParm lp,
                                                                                T *input_matrix_real,
                                                                                T *input_matrix_imag,
                                                                                T2 *output_matrix,
                                                                                size_t input_row_size,
                                                                                size_t input_col_size,
                                                                                size_t input_leading_dim_size,
                                                                                size_t output_leading_dim_size,
                                                                                size_t batch_size);

//defined in rocfft_transpose_complex_interleaved_to_planar_kernel.cu
template<typename T, typename T2, int micor_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_complex_interleaved_to_complex_planar(hipLaunchParm lp,
                                                                                T *input_matrix,
                                                                                T2 *output_matrix_real,
                                                                                T2 *output_matrix_imag,
                                                                                size_t input_row_size,
                                                                                size_t input_leading_dim_size,
                                                                                size_t output_leading_dim_size,
                                                                                size_t batch_size);

//include the definations of kernel templates here
#include "rocfft_transpose_kernel.cu"
#include "rocfft_transpose_kernel_check_boundary.cu"
#include "rocfft_transpose_complex_planar_kernel.cu"
#include "rocfft_transpose_complex_planar_to_interleaved_kernel.cu"
#include "rocfft_transpose_complex_interleaved_to_planar_kernel.cu"
#endif
