#include "rocfft_hip.h"
#include "common.h"



/*
   transpose input of size m * n (up to DIM_X * DIM_X) to output of size n * m
   input, output are in device memory
   shared memory of size DIM_X*DIM_X is allocated internally as working space

   Assume DIM_X by DIM_Y threads are reading & wrting a tile size DIM_X * DIM_X
   DIM_X is divisible by DIM_Y
*/

template<typename T, int DIM_X, int DIM_Y>
__device__ void
transpose_tile_device(const T* input, T* output, int m, int n, int gx, int gy, int ld_in, int ld_out, T *twiddles_large, const int twl, const int dir )
{

    __shared__ T shared_A[DIM_X][DIM_X];

    int tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int tx1 = tid % DIM_X;
    int ty1 = tid / DIM_X;

    #pragma unroll
    for(int i=0; i<m; i+=DIM_Y)
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

    for(int i=0; i<n; i+=DIM_Y)
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



template<typename T, int DIM_X, int DIM_Y>
__global__ void
transpose_kernel2( hipLaunchParm lp, const T* input, T* output, T *twiddles_large, size_t dim, size_t *lengths, size_t *stride_in, size_t *stride_out, const int twl, const int dir )
{
    int m = lengths[1];
    int n = lengths[0];
    int ld_in = stride_in[1];
    int ld_out = stride_out[1];

    size_t iOffset = 0;
    size_t oOffset = 0;
 
    size_t counter_mod = hipBlockIdx_z;
    
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



    input += hipBlockIdx_x * DIM_X + hipBlockIdx_y * DIM_X * ld_in + iOffset;
    output += hipBlockIdx_x * DIM_X * ld_out + hipBlockIdx_y * DIM_X + oOffset;

    int mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
    int nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n

    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, hipBlockIdx_x * DIM_X, hipBlockIdx_y * DIM_X, ld_in, ld_out, twiddles_large, twl, dir);
}

template<typename T, int DIM_X, int DIM_Y>
__global__ void
transpose_kernel2_scheme( hipLaunchParm lp, const T* input, T* output, T *twiddles_large, size_t dim, size_t *lengths, size_t *stride_in, size_t *stride_out, const int scheme)
{
    int m = scheme == 1 ? lengths[2] : lengths[1]*lengths[2];
    int n = scheme == 1 ? lengths[0]*lengths[1] : lengths[0];
    int ld_in = scheme == 1 ? stride_in[2] : stride_in[1];
    int ld_out = scheme == 1 ? stride_out[1] : stride_out[2];

    size_t iOffset = 0;
    size_t oOffset = 0;
 
    size_t counter_mod = hipBlockIdx_z;
    
    for(int i = dim; i>3; i--){
        int currentLength = 1;
        for(int j=3; j<i; j++){
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

    int mm = min(m - hipBlockIdx_y * DIM_X, DIM_X); // the corner case along m
    int nn = min(n - hipBlockIdx_x * DIM_X, DIM_X); // the corner case along n

    transpose_tile_device<T, DIM_X, DIM_Y>(input, output, mm, nn, hipBlockIdx_x * DIM_X, hipBlockIdx_y * DIM_X, ld_in, ld_out, twiddles_large, 0, 0);
}


/*
   transpose input of pow of 2 matrices with 64 * 64 block
   Notice: Only works for single complex precsion. double complex precsion will overflow shared memory (LDS) and fail 
*/

// Local structure to embody/capture tile dimensions
typedef struct tag_Tile
{
   size_t x;
   size_t y;
} Tile;


template<typename T, int dir, int twl, TransTileDir ttd>
__global__
void
transpose_var1( hipLaunchParm lp, T *twiddles_large, T* pmComplexIn, T* pmComplexOut,
			const ulong numGroupsY,  const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = numGroupsY;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];

   constexpr size_t TWIDTH = 64/(sizeof(T)/8);
   __shared__ T lds[ TWIDTH ][ TWIDTH ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*dist_i;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = stride_i;
   
   if(ttd == TTD_IP_HOR)
   {
   	iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
	iOffset += groupIndex.x * wgTileExtent.x;
   }
   else
   {
   	iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
   	iOffset += currDimIndex * wgTileExtent.x;
   }
   
   T* tileIn = pmComplexIn + iOffset;
   T tmp;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here

	if(ttd == TTD_IP_HOR)
	{		 
		if(twl == 3)
		{
			if(dir == -1)
			{
				TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
			}
			else
			{
				TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
			}
		}
		else if(twl == 4)
		{
			if(dir == -1)
			{
				TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
			}
			else
			{
				TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
			}
		}
	}
	else
	{
		if(twl == 3)
		{
			if(dir == -1)
			{
				TWIDDLE_STEP_MUL_FWD(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.y * wgUnroll + yInd) * (currDimIndex * wgTileExtent.x + xInd), tmp)
			}
			else
			{
				TWIDDLE_STEP_MUL_INV(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.y * wgUnroll + yInd) * (currDimIndex * wgTileExtent.x + xInd), tmp)
			}
		}
		else if(twl == 4)
		{
			if(dir == -1)
			{
				TWIDDLE_STEP_MUL_FWD(TWLstep4, twiddles_large, (groupIndex.x * wgTileExtent.y * wgUnroll + yInd) * (currDimIndex * wgTileExtent.x + xInd), tmp)
			}
			else
			{
				TWIDDLE_STEP_MUL_INV(TWLstep4, twiddles_large, (groupIndex.x * wgTileExtent.y * wgUnroll + yInd) * (currDimIndex * wgTileExtent.x + xInd), tmp)
			}
		}
	}

 
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*dist_o;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = stride_o;

   if(ttd == TTD_IP_HOR)
   {
   	oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   	oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   }
   else
   {
   	oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
   	oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
   }
   
   T* tileOut = pmComplexOut + oOffset;

   const size_t transposeRatio = wgTileExtent.x / ( wgTileExtent.y * wgUnroll );
   const size_t groupingPerY = wgUnroll / wgTileExtent.y;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % groupingPerY ); 
         size_t yInd = localIndex.y/groupingPerY + t * (wgTileExtent.y * transposeRatio); 
         tmp = lds[ yInd ][ xInd ]; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tileOut[ gInd ] = tmp;
      }
}



