#include "rocfft_hip.h"
#include "common.h"



/*
   transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
   input, output are in device memory
   shared memory of size DIM_X*DIM_X is allocated size_ternally as working space

   Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
   DIM_X is divisible by DIM_Y
*/

template<typename T, size_t DIM_X, size_t DIM_Y>
__device__ void
transpose_tile_device(const T* input, T* output, const size_t m, const size_t n, size_t gx, size_t gy, size_t ld_in, size_t ld_out, T *twiddles_large, const int twl, const int dir )
{

    __shared__ T shared_A[DIM_X][DIM_X];

    size_t tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    size_t tx1 = tid % DIM_X;
    size_t ty1 = tid / DIM_X;
            
    for(size_t i=0; i<m; i+=DIM_Y)
    {
        if( tx1 < n && (ty1 + i) < m)
        {
            T tmp = input[tx1 + (ty1 + i) * ld_in];
            if(twl == 2)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep2, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp); 
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep2, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp);
                }
            }
            else if(twl == 3)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp);
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp);
                }
            }
            else if(twl == 4)
            {
                if(dir == -1)
                {
                    TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp);
                }
                else
                {
                    TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, (gx + tx1)*(gy + ty1 + i), tmp);
                }
     
            }

            shared_A[tx1][ty1+i] = tmp;   // the transpose taking place here
        }
    }
    __syncthreads();// ?

    for(size_t i=0; i<n; i+=DIM_Y)
    {
        //reconfigure the threads
        if( tx1 < m && (ty1 + i)< n)
        {
            output[tx1 + (i + ty1) * ld_out] = shared_A[ty1+i][tx1];
        }
    }

}

/*
   transpose input of size m * n to output of size n * m
   input, output are in device memory

   2D grid and 2D thread block (DIM_X, DIM_Y)

   Assume DIM_X by DIM_Y threads are transposing a tile DIM_X * DIM_X
*/



template<typename T, size_t DIM_X, size_t DIM_Y>
__global__ void
transpose_kernel2( hipLaunchParm lp, const T* input, T* output, T *twiddles_large, size_t dim, size_t *lengths, size_t *stride_in, size_t *stride_out, const int twl, const int dir )
{
    size_t m = lengths[1];
    size_t n = lengths[0];
    size_t ld_in = stride_in[1];
    size_t ld_out = stride_out[1];

    size_t iOffset = 0;
    size_t oOffset = 0;
 
    size_t counter_mod = hipBlockIdx_z;
    
    for(size_t i = dim; i>2; i--){
        size_t currentLength = 1;
        for(size_t j=2; j<i; j++){
            currentLength *= lengths[j];
        }
    
        iOffset += (counter_mod / currentLength)*stride_in[i];
        oOffset += (counter_mod / currentLength)*stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset+= counter_mod * stride_in[2];
    oOffset+= counter_mod * stride_out[2];
        

    input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in + iOffset;
    output += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X + oOffset;

    size_t mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
    size_t nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n
    
    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, hipBlockIdx_x * DIM_X, hipBlockIdx_y * DIM_X, ld_in, ld_out, twiddles_large, twl, dir);
}

template<typename T, size_t DIM_X, size_t DIM_Y>
__global__ void
transpose_kernel2_scheme( hipLaunchParm lp, const T* input, T* output, T *twiddles_large, size_t dim, size_t *lengths, size_t *stride_in, size_t *stride_out, const size_t scheme)
{
    size_t m = scheme == 1 ? lengths[2] : lengths[1]*lengths[2];
    size_t n = scheme == 1 ? lengths[0]*lengths[1] : lengths[0];
    size_t ld_in = scheme == 1 ? stride_in[2] : stride_in[1];
    size_t ld_out = scheme == 1 ? stride_out[1] : stride_out[2];

    size_t iOffset = 0;
    size_t oOffset = 0;
 
    size_t counter_mod = hipBlockIdx_z;
    
    for(size_t i = dim; i>3; i--){
        size_t currentLength = 1;
        for(size_t j=3; j<i; j++){
            currentLength *= lengths[j];
        }
    
        iOffset += (counter_mod / currentLength)*stride_in[i];
        oOffset += (counter_mod / currentLength)*stride_out[i];
        counter_mod = counter_mod % currentLength;
    }
    iOffset+= counter_mod * stride_in[3];
    oOffset+= counter_mod * stride_out[3];



    input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in + iOffset;
    output += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X + oOffset;

    size_t mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
    size_t nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n

    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, hipBlockIdx_x * DIM_X, hipBlockIdx_y * DIM_X, ld_in, ld_out, twiddles_large, 0, 0);
}

