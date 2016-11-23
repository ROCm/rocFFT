

template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_check_boundary(hipLaunchParm lp, T *input_matrix, T *output_matrix, size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{
//TODO
}
