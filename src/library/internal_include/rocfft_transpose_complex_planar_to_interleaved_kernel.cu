
//works for complex planar to complex interleaved; T should be float or double; T2 should be float2 or double2
template<typename T, typename T2, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace_complex_planar_to_complex_interleaved(hipLaunchParm lp,
                                                                                T *input_matrix_real,
	            								T *input_matrix_imag,
									        T2 *output_matrix,
										size_t input_row_size,
										size_t input_col_size,
										size_t input_leading_dim_size,
										size_t output_leading_dim_size,
										size_t batch_size)
{
    // WG size can be assumed to be 16 by 16 in most cases
    size_t local_idx_0 = hipThreadIdx_x;// 0-15
    size_t local_idx_1 = hipThreadIdx_y;// 0-15
    size_t block_idx_0 = hipBlockIdx_x;// index of work groups
    size_t block_idx_1 = hipBlockIdx_y;
    size_t block_dim_0 = hipBlockDim_x;// size of work groups 16
    size_t block_dim_1 = hipBlockDim_y;// size of work groups 16
    size_t grid_dim_0 = hipGridDim_x;// number of blocks only the first dimension is loaded to handle batch size

    // for 64 x 64 macro tile size we will need 16 x 4 x 64 blocks (4 x 64 == 16 x 16)
    // for 32 x 32 macro tile size we will need  4 x 8 x 32 blocks
    const size_t macro_tile_col_size = micro_tile_col_size * wg_col_size;
    const size_t macro_tile_row_size = micro_tile_row_size * wg_row_size;
    const size_t reshape_factor = macro_tile_col_size / block_dim_0; // 64 / 16 = 4 need to fit 4 rows into one row in LDS; 32 / 16 = 2
    const size_t unroll_factor = macro_tile_row_size / (block_dim_1 / reshape_factor); // 64 / (16 / 4) = 16; 32 / (16 / 2) = 4

	__shared__ T lds_real[macro_tile_row_size][macro_tile_col_size];
	__shared__ T lds_imag[macro_tile_row_size][macro_tile_col_size];

	size_t blocks_per_batch = grid_dim_0 / batch_size;
	size_t batch_idx = block_idx_0 / blocks_per_batch;

	input_matrix_real += batch_idx * input_leading_dim_size * input_row_size;
	input_matrix_imag += batch_idx * input_leading_dim_size * input_row_size;

	size_t input_offset = 0;
	input_offset += input_leading_dim_size * block_idx_1 * macro_tile_row_size;
	input_offset += (block_idx_0 % blocks_per_batch) * macro_tile_col_size;

	input_matrix_real += input_offset;
	input_matrix_imag += input_offset;

	for(int i = 0; i < unroll_factor; i++)
	{
	        //each iteration 256 work items will read from a 4 x 64 subblock
                //there are 16 iterations
		size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % reshape_factor) * block_dim_0; // local_idx_0 + (local_idx_1 % 4) * 16
                size_t subblock_idx_1 = local_idx_1 / reshape_factor + i * (block_dim_1 / reshape_factor);
		//transpose happened here
		lds_real[subblock_idx_0][subblock_idx_1] = input_matrix_real[subblock_idx_1 * input_leading_dim_size + subblock_idx_0];
		lds_imag[subblock_idx_0][subblock_idx_1] = input_matrix_imag[subblock_idx_1 * input_leading_dim_size + subblock_idx_0];
	}

	__syncthreads();

	output_matrix += batch_idx * input_col_size * output_leading_dim_size;
	size_t output_offset = 0;
        output_offset += output_leading_dim_size * (block_idx_0 % blocks_per_batch) * macro_tile_row_size;//input_row_size == ouput_col_size
        output_offset += block_idx_1 * macro_tile_col_size;

	output_matrix += output_offset;

	for(int i = 0; i < unroll_factor; i++)
	{
	    size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % reshape_factor) * block_dim_0;// 0-63
            size_t subblock_idx_1 = local_idx_1 / reshape_factor + i * (block_dim_1 / reshape_factor);// 0-3, 4-7 ... 60-63
	    output_matrix[subblock_idx_1 * output_leading_dim_size + subblock_idx_0].x = lds_real[subblock_idx_1][subblock_idx_0];
	    output_matrix[subblock_idx_1 * output_leading_dim_size + subblock_idx_0].y = lds_imag[subblock_idx_1][subblock_idx_0];
	}
}
