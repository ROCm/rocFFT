#ifndef FFT_POW2_LARGE_ENTRY_HIP_H
#define FFT_POW2_LARGE_ENTRY_HIP_H

#include "fft_pow2_large_hip.h"

template<int dir>
__global__
void fft_64_128_bcc_pk(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_8192, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/8)*8192 + (batch%8)*16;
	oOffset = (batch/8)*8192 + (batch%8)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%16) + (me/16)*128 + t*1024];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();

	for(uint t=0; t<2; t++)
	{
		b = (batch%8)*16 + t*8 + (me/16);
		
		lfft_64<dir, 2>(twiddles_64, twiddles_8192, lds + t*512 + (me/16)*64, me%16, b);
		
		__syncthreads();
	}

	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*128 + t*1024] = R0;
	}

}


template<int dir>
__global__
void fft_128_64_brc_pk(hipLaunchParm lp, float2 *twiddles_128, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	iOffset = (batch/8)*8192 + (batch%8)*1024;
	oOffset = (batch/8)*8192 + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[me + t*128];
		lds[t*128 + me] = R0;
	}

	__syncthreads();

	lfft_128<dir, 0>(twiddles_128, 0, lds + (me/16)*128, me%16, 0);

	__syncthreads();


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*64 + t*1024] = R0;
	}

}


template<int dir>
__global__
void fft_64_256_bcc_pk(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_16384, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/16)*16384 + (batch%16)*16;
	oOffset = (batch/16)*16384 + (batch%16)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%16) + (me/16)*256 + t*2048];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();

	for(uint t=0; t<2; t++)
	{
		b = (batch%16)*16 + t*8 + (me/16);

		lfft_64<dir, 2>(twiddles_64, twiddles_16384, lds + t*512 + (me/16)*64, me%16, b);

		__syncthreads();
	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*256 + t*2048] = R0;
	}

}


template<int dir>
__global__
void fft_256_64_brc_pk(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[2048];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	iOffset = (batch/8)*16384 + (batch%8)*2048;
	oOffset = (batch/8)*16384 + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{
		lfft_256<dir, 0>(twiddles_256, 0, lds + t*1024 + (me/64)*256, me%64, 0);

		__syncthreads();
	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*64 + t*2048] = R0;
	}

}


template<int dir>
__global__
void fft_128_256_bcc_pk(hipLaunchParm lp, float2 *twiddles_128, float2 *twiddles_32768, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/32)*32768 + (batch%32)*8;
	oOffset = (batch/32)*32768 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%8) + (me/8)*256 + t*4096];
		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	b = (batch%32)*8 + (me/16);

	lfft_128<dir, 2>(twiddles_128, twiddles_32768, lds + (me/16)*128, me%16, b);
	
	__syncthreads();

	
	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*4096] = R0;
	}
}


template<int dir>
__global__
void fft_256_128_brc_pk(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[2048];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	iOffset = (batch/16)*32768 + (batch%16)*2048;
	oOffset = (batch/16)*32768 + (batch%16)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{
		lfft_256<dir, 0>(twiddles_256, 0, lds + t*1024 + (me/64)*256, me%64, 0);

		__syncthreads();

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*128 + t*4096] = R0;
	}

}


template<int dir>
__global__
void fft_256_256_bcc_pk(hipLaunchParm lp, float2 *twiddles_256, float2 *twiddles_65536, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[2048];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/32)*65536 + (batch%32)*8;
	oOffset = (batch/32)*65536 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%8) + (me/8)*256 + t*8192];
		lds[t*32 + (me%8)*256 + (me/8)] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{

		b = (batch%32)*8 + t*4 + (me/64);

		lfft_256<dir, 2>(twiddles_256, twiddles_65536, lds + t*1024 + (me/64)*256, me%64, b);

		__syncthreads();

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
	}
}


template<int dir>
__global__
void fft_256_256_brc_pk(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[2048];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	iOffset = (batch/32)*65536 + (batch%32)*2048;
	oOffset = (batch/32)*65536 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{
		lfft_256<dir, 0>(twiddles_256, 0, lds + t*1024 + (me/64)*256, me%64, 0);

		__syncthreads();

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
	}

}


template<int dir>
__global__
void fft_64_2048_bcc_pk(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_131072, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/128)*131072 + (batch%128)*16;
	oOffset = (batch/128)*131072 + (batch%128)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%16) + (me/16)*2048 + t*16384];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{

		b = (batch%128)*16 + t*8 + (me/16);

		lfft_64<dir, 3>(twiddles_64, twiddles_131072, lds + t*512 + (me/16)*64, me%16, b);

		__syncthreads();

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*2048 + t*16384] = R0;
	}
}



template<int dir>
__global__
void fft_64_4096_bcc_pk(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_262144, float2 * gbIn, float2 * gbOut)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float2 lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	float2 R0;

	uint b = 0;

	iOffset = (batch/256)*262144 + (batch%256)*16;
	oOffset = (batch/256)*262144 + (batch%256)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(uint t=0; t<8; t++)
	{
		R0 = lwbIn[(me%16) + (me/16)*4096 + t*32768];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(uint t=0; t<2; t++)
	{

		b = (batch%256)*16 + t*8 + (me/16);

		lfft_64<dir, 3>(twiddles_64, twiddles_262144, lds + t*512 + (me/16)*64, me%16, b);

		__syncthreads();

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*4096 + t*32768] = R0;
	}
}





// Local structure to embody/capture tile dimensions
typedef struct tag_Tile
{
   size_t x;
   size_t y;
} Tile;

enum TransTileDir
{
	TTD_IP_HOR,
	TTD_IP_VER,
};

template<int dir, int twl, TransTileDir ttd>
__global__
void
transpose_var1( hipLaunchParm lp, float2 *twiddles_large, float2* pmComplexIn, float2* pmComplexOut,
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
   __shared__ float2 lds[ 64 ][ 64 ];

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
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
		 
	 if(twl == 3)
	 {
		 if(dir == -1)
		 {
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
		 }
		 else
		 {
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
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
   
   float2* tileOut = pmComplexOut + oOffset;

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


#endif // FFT_POW2_LARGE_ENTRY_HIP_H

