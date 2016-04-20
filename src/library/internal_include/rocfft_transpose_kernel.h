#ifndef __ROCFFT_TRANSPOSE_KERNEL_H__
#define __ROCFFT_TRANSPOSE_KERNEL_H__
#include "hip_runtime.h"

//defined in rocfft_transpose_kernel.cu
template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t batch_size);

//defined in rocfft_transpose_kernel_check_boundary.cu
template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_check_boundary(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t batch_size);

//include the definations of kernel templates here
#include "rocfft_transpose_kernel.cu"
#include "rotfft_transpose_kernel_check_boundary.cu"
#endif
