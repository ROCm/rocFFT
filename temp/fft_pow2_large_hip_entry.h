
template<int dir>
__global__
void fft_8192_1(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_8192, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_8192_2(hipLaunchParm lp, float2 *twiddles_128, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_16384_1(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_16384, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_16384_2(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_32768_1(hipLaunchParm lp, float2 *twiddles_128, float2 *twiddles_32768, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_32768_2(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_65536_1(hipLaunchParm lp, float2 *twiddles_256, float2 *twiddles_65536, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_65536_2(hipLaunchParm lp, float2 *twiddles_256, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_131072_1(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_131072, float2 * gbIn, float2 * gbOut, const uint count)
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
void fft_131072_2(hipLaunchParm lp, float2 *twiddles_2048, float2 * gb, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;
	
	__shared__ float lds[2048];
	uint ioOffset;
	float2 *lwb;

	ioOffset = (batch/64)*131072 + (batch%64)*2048;
	lwb = gb + ioOffset;
	
	fft_2048<SB_UNIT, dir>(twiddles_2048, lwb, lwb, lds, me, 1, 1);

}


// Local structure to embody/capture tile dimensions
typedef struct tag_Tile
{
   size_t x;
   size_t y;
} Tile;

__global__
void
transpose_131072( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 1;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*131072;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 2048;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 2048;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*131072;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 64;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 64;
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


template<int dir>
__global__
void fft_262144_1(hipLaunchParm lp, float2 *twiddles_64, float2 *twiddles_262144, float2 * gbIn, float2 * gbOut, const uint count)
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


template<int dir>
__global__
void fft_262144_2(hipLaunchParm lp, float2 *twiddles_4096, float2 * gb, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;
	
	__shared__ float lds[4096];
	uint ioOffset;
	float2 *lwb;

	ioOffset = (batch/64)*262144 + (batch%64)*4096;
	lwb = gb + ioOffset;
	
	fft_4096<SB_UNIT, dir>(twiddles_4096, lwb, lwb, lds, me, 1, 1);

}


__global__
void
transpose_262144( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 1;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*262144;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 4096;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 4096;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*262144;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 64;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 64;
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



__global__
void
transpose_524288_1( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 512;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 512;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*557056;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 1088;
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


template<int dir>
__global__
void fft_524288_1(hipLaunchParm lp, float2 *twiddles_1024, float2 * gbIn, float2 * gbOut, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	iOffset = (batch/512)*557056 + (batch%512)*1088;
	oOffset = (batch/512)*524288 + (batch%512)*1024;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_1024<SB_UNIT, dir>(twiddles_1024, lwbIn, lwbOut, lds, me, 1, 1);
}


template<int dir>
__global__
void
transpose_524288_2( hipLaunchParm lp, float2 *twiddles_524288, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 8;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 1024;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
		 
		 if(dir == -1)
		 {
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_524288, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
		 }
		 else
		 {
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_524288, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
		 }
		 
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*589824;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 576;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 576;
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


template<int dir>
__global__
void fft_524288_2(hipLaunchParm lp, float2 *twiddles_512, float2 * gb, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint ioOffset;
	float2 *lwb;

	ioOffset = (batch/1024)*589824 + (batch%1024)*576;
	lwb = gb + ioOffset;

	fft_512<SB_UNIT, dir>(twiddles_512, lwb, lwb, lds, me, 1, 1);
}


__global__
void
transpose_524288_3( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 8;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*589824;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 576;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
   iOffset += currDimIndex * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 576;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
   oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 1024;
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


__global__
void
transpose_1048576_1( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 1024;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 1088;
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


template<int dir>
__global__
void fft_1048576_1(hipLaunchParm lp, float2 *twiddles_1024, float2 * gbIn, float2 * gbOut, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];

	uint iOffset;
	uint oOffset;
	float2 *lwbIn;
	float2 *lwbOut;

	iOffset = (batch/1024)*1114112 + (batch%1024)*1088;
	oOffset = (batch/1024)*1048576 + (batch%1024)*1024;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	tfft_1024(twiddles_1024, me, lds, lwbIn, lwbOut);
}


template<int dir>
__global__
void
transpose_1048576_2( hipLaunchParm lp, float2 *twiddles_1048576, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 1024;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
		 
         // Transpose of Tile data happens here
		 if(dir == -1)
		 {
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_1048576, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)		 
		 }
		 else
		 {
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_1048576, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)		 
		 }
		 
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 1088;
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


__global__
void fft_1048576_2(hipLaunchParm lp, float2 *twiddles_1024, float2 * gb, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];

	uint ioOffset;
	float2 *lwb;

	ioOffset = (batch/1024)*1114112 + (batch%1024)*1088;
	lwb = gb + ioOffset;
	
	fft_1024<SB_UNIT, dir>(twiddles_1024, lwb, lwb, lds, me, 1, 1);
}

__global__
void
transpose_1048576_3( hipLaunchParm lp, float2* pmComplexIn, float2* pmComplexOut, const uint count )
{
   const Tile localIndex = { (size_t)hipThreadIdx_x, (size_t)hipThreadIdx_y }; 
   const Tile localExtent = { (size_t)hipBlockDim_x, (size_t)hipBlockDim_y }; 
   const Tile groupIndex = { (size_t)hipBlockIdx_x, (size_t)hipBlockIdx_y };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   __shared__ float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
   iOffset += currDimIndex * wgTileExtent.x;
   
   float2* tileIn = pmComplexIn + iOffset;
   float2 tmp;
   rowSizeinUnits = 1088;
   

      for( uint t=0; t < wgUnroll; t++ )
      {
         size_t xInd = localIndex.x + localExtent.x * ( localIndex.y % wgTileExtent.y ); 
         size_t yInd = localIndex.y/wgTileExtent.y + t * wgTileExtent.y; 
         size_t gInd = xInd + rowSizeinUnits * yInd;
         tmp = tileIn[ gInd ];
         // Transpose of Tile data happens here
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   __syncthreads();
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
   oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
   
   float2* tileOut = pmComplexOut + oOffset;

   rowSizeinUnits = 1024;
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


