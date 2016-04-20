#ifndef __ROCFFT_TRANSPOSE_KERNEL_H__
#define __ROCFFT_TRANSPOSE_KERNEL_H__
#include "hip_runtime.h"

template<typename T, int micro_tile_col_size, int micro_tile_row_size>
__global__ void transpose_kernel_outplace(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t batch_size);

#include "rocfft_transpose_kernel.cu"
#endif
