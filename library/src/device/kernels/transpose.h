#include <hip/hip_runtime.h>
#include "butterfly.h"

//works for real to real and complex interleaved to complex interleaved 
template<typename T, int micro_tile_col_size, int micro_tile_row_size, int wg_col_size, int wg_row_size>
__global__ void transpose_kernel_outplace(hipLaunchParm lp, T *input_matrix, T *output_matrix, T *twiddles_large,
        const size_t batch_count, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out)
{
    const size_t input_row_size = lengths[1];
    const size_t input_col_size = lengths[0];
    const size_t input_leading_dim_size = stride_in[1];
    const size_t output_leading_dim_size = stride_out[1];

    // WG size can be assumed to be 16 by 16
    size_t local_idx_0 = hipThreadIdx_x;// 0-15
    size_t local_idx_1 = hipThreadIdx_y;// 0-15
    size_t block_idx_0 = hipBlockIdx_x;// index of work groups
    size_t block_idx_1 = hipBlockIdx_y;
    size_t block_dim_0 = hipBlockDim_x;// size of work groups 16
    size_t block_dim_1 = hipBlockDim_y;// size of work groups 16
    size_t grid_dim_0 = hipGridDim_x;// number of blocks

    // for 64 x 64 macro tile size we will need 16 x 4 x 64 blocks (4 x 64 == 16 x 16)
    // for 32 x 32 macro tile size we will need  4 x 8 x 32 blocks
    const size_t macro_tile_col_size = micro_tile_col_size * wg_col_size;
    const size_t macro_tile_row_size = micro_tile_row_size * wg_row_size;
    const size_t reshape_factor = macro_tile_col_size / block_dim_0; // 64 / 16 = 4 need to fit 4 rows into one row in LDS; 32 / 16 = 2
    const size_t unroll_factor = macro_tile_row_size / (block_dim_1 / reshape_factor); // 64 / (16 / 4) = 16; 32 / (16 / 2) = 4

    __shared__ T lds[macro_tile_row_size][macro_tile_col_size];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
    size_t blocks_per_matrix = (input_col_size / macro_tile_col_size);

    {
    size_t counter_mod = block_idx_0 / blocks_per_matrix;
    
    for(int i = dim; i>2; i--){
        int currentLength = 1;
        for(int j=2; j<i; j++){
            currentLength *= lengths[j];
        }
    
        iOffset += (counter_mod / currentLength)*stride_in[i];
        oOffset += (counter_mod / currentLength)*stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset+= counter_mod * stride_in[2];
    oOffset+= counter_mod * stride_out[2];
    }


    input_matrix += iOffset;

    size_t input_offset = 0;
    input_offset += input_leading_dim_size * block_idx_1 * macro_tile_row_size;// each WG works on 64 by 64 block or 32 by 32
    input_offset += (block_idx_0 % blocks_per_matrix) * macro_tile_col_size;

    input_matrix += input_offset;
    for(int i = 0; i < unroll_factor; i++)
    {
        //each iteration 256 work items will read from a 4 x 64 subblock
        //there are 16 iterations
        size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % reshape_factor) * block_dim_0; // local_idx_0 + (local_idx_1 % 4) * 16
        size_t subblock_idx_1 = local_idx_1 / reshape_factor + i * (block_dim_1 / reshape_factor);
        //transpose happened here

        T tmp = input_matrix[subblock_idx_1 * input_leading_dim_size + subblock_idx_0];
        if(twiddles_large != 0)
        {
            TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, subblock_idx_0*subblock_idx_1, tmp);
        }
        lds[subblock_idx_0][subblock_idx_1] = tmp;

    }

    __syncthreads();

    output_matrix += oOffset;

    size_t output_offset = 0;
    output_offset += output_leading_dim_size * (block_idx_0 % blocks_per_matrix) * macro_tile_row_size;//input_row_size == ouput_col_size
    output_offset += block_idx_1 * macro_tile_col_size;

    output_matrix += output_offset;

    for(int i = 0; i < unroll_factor; i++)
    {
        size_t subblock_idx_0 = local_idx_0 + (local_idx_1 % reshape_factor) * block_dim_0;// 0-63
        size_t subblock_idx_1 = local_idx_1 / reshape_factor + i * (block_dim_1 / reshape_factor);// 0-3, 4-7 ... 60-63
        T  temp = lds[subblock_idx_1][subblock_idx_0];
        output_matrix[subblock_idx_1 * output_leading_dim_size + subblock_idx_0] = temp;//lds[subblock_idx_1][subblock_idx_0];
    }

}



/*
   transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
   input, output are in device memory
   shared memory of size DIM_X*DIM_X is allocated internally as working space

   Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
   DIM_X is divisible by DIM_Y
*/

template<typename T, int DIM_X, int DIM_Y>
__device__ void
transpose_tile_device(const T* input, T* output, int m, int n, int input_lda, int output_lda )
{

    __shared__ T shared_A[DIM_X][DIM_X+1];//avoid bank conflicts

    int tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int tx1 = tid % DIM_X;
    int ty1 = tid / DIM_X;

    #pragma unroll
    for(int i=0; i<n; i+=DIM_Y)
    {
        if( tx1 < m && (ty1 + i) < n)
        {
            shared_A[ty1+i][tx1] = input[tx1 + (ty1 + i) * input_lda];   // the transpose taking place here
        }
    }
    __syncthreads();// ?

    for(int i=0; i<m; i+=DIM_Y)
    {
        //reconfigure the threads
        if( tx1 < n && (ty1 + i)< m)
        {
            output[tx1 + (i + ty1) * output_lda] = shared_A[tx1][ty1+i];
        }
    }

}

/*
   transpose input of size m * n to output of size n * m
   input, output are in device memory

   2D grid and 2D thread block (DIM_X, DIM_Y)

   Assume DIM_X by DIM_Y threads are transposing a tile DIM_X * DIM_X
*/



template<typename T, int DIM_X, int DIM_Y>
__global__ void
transpose_kernel( hipLaunchParm lp, int m, int n, const T* input, T* output, int input_lda, int output_lda )
{

    input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * input_lda + (hipBlockIdx_z * input_lda * n);
    output += hipBlockIdx_x * DIM_X * output_lda + hipBlockIdx_y * DIM_X + (hipBlockIdx_z * output_lda * m);

    int mm = min(m - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along m
    int nn = min(n - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along n

    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, input_lda, output_lda );
}





