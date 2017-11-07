/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_LARGE_ENTRY_HIP_H
#define POW2_LARGE_ENTRY_HIP_H

#include "common.h"
#include "pow2_large.h"


template<typename T,  int dir>
__global__
void fft_64_128_bcc_d1_pk(hipLaunchParm lp, T *twiddles_64, T *twiddles_8192, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/8)*8192 + (batch%8)*16;
	oOffset = (batch/8)*8192 + (batch%8)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_128_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_8192, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_128_64_brc_d1_pk(hipLaunchParm lp, T *twiddles_128, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/8)*8192 + (batch%8)*1024;
	oOffset = (batch/8)*8192 + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_128_64_brc<T, SB_UNIT, dir>(twiddles_128, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_256_bcc_d1_pk(hipLaunchParm lp, T *twiddles_64, T *twiddles_16384, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/16)*16384 + (batch%16)*16;
	oOffset = (batch/16)*16384 + (batch%16)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_256_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_16384, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_64_brc_d1_pk(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/8)*16384 + (batch%8)*2048;
	oOffset = (batch/8)*16384 + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;
	
	fft_256_64_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_128_256_bcc_d1_pk(hipLaunchParm lp, T *twiddles_128, T *twiddles_32768, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/32)*32768 + (batch%32)*8;
	oOffset = (batch/32)*32768 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_128_256_bcc<T, SB_UNIT, dir>(twiddles_128, twiddles_32768, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_128_brc_d1_pk(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/16)*32768 + (batch%16)*2048;
	oOffset = (batch/16)*32768 + (batch%16)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_128_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_256_bcc_d1_pk(hipLaunchParm lp, T *twiddles_256, T *twiddles_65536, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/32)*65536 + (batch%32)*8;
	oOffset = (batch/32)*65536 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_256_bcc<T, SB_UNIT, dir>(twiddles_256, twiddles_65536, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_256_brc_d1_pk(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/32)*65536 + (batch%32)*2048;
	oOffset = (batch/32)*65536 + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_256_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_2048_bcc_d1_pk(hipLaunchParm lp, T *twiddles_64, T *twiddles_131072, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/128)*131072 + (batch%128)*16;
	oOffset = (batch/128)*131072 + (batch%128)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_2048_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_131072, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_4096_bcc_d1_pk(hipLaunchParm lp, T *twiddles_64, T *twiddles_262144, T * gbIn, T * gbOut)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/256)*262144 + (batch%256)*16;
	oOffset = (batch/256)*262144 + (batch%256)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_4096_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_262144, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_128_bcc_d2_s1(hipLaunchParm lp, T *twiddles_64, T *twiddles_8192, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(8*len))*dist_i;
	oOffset = (batch/(8*len))*dist_o;
	batch = batch%(8*len);

	iOffset += (batch/8)*stride_i + (batch%8)*16;
	oOffset += (batch/8)*stride_o + (batch%8)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_128_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_8192, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_128_64_brc_d2_s1(hipLaunchParm lp, T *twiddles_128, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(8*len))*dist_i;
	oOffset = (batch/(8*len))*dist_o;
	batch = batch%(8*len);

	iOffset += (batch/8)*stride_i + (batch%8)*1024;
	oOffset += (batch/8)*stride_o + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_128_64_brc<T, SB_UNIT, dir>(twiddles_128, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_256_bcc_d2_s1(hipLaunchParm lp, T *twiddles_64, T *twiddles_16384, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(16*len))*dist_i;
	oOffset = (batch/(16*len))*dist_o;
	batch = batch%(16*len);

	iOffset += (batch/16)*stride_i + (batch%16)*16;
	oOffset += (batch/16)*stride_o + (batch%16)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_256_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_16384, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_64_brc_d2_s1(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(8*len))*dist_i;
	oOffset = (batch/(8*len))*dist_o;
	batch = batch%(8*len);

	iOffset += (batch/8)*stride_i + (batch%8)*2048;
	oOffset += (batch/8)*stride_o + (batch%8)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;
	
	fft_256_64_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_128_256_bcc_d2_s1(hipLaunchParm lp, T *twiddles_128, T *twiddles_32768, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(32*len))*dist_i;
	oOffset = (batch/(32*len))*dist_o;
	batch = batch%(32*len);

	iOffset += (batch/32)*stride_i + (batch%32)*8;
	oOffset += (batch/32)*stride_o + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_128_256_bcc<T, SB_UNIT, dir>(twiddles_128, twiddles_32768, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_128_brc_d2_s1(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(16*len))*dist_i;
	oOffset = (batch/(16*len))*dist_o;
	batch = batch%(16*len);

	iOffset += (batch/16)*stride_i + (batch%16)*2048;
	oOffset += (batch/16)*stride_o + (batch%16)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_128_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_256_bcc_d2_s1(hipLaunchParm lp, T *twiddles_256, T *twiddles_65536, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(32*len))*dist_i;
	oOffset = (batch/(32*len))*dist_o;
	batch = batch%(32*len);

	iOffset += (batch/32)*stride_i + (batch%32)*8;
	oOffset += (batch/32)*stride_o + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_256_bcc<T, SB_UNIT, dir>(twiddles_256, twiddles_65536, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_256_256_brc_d2_s1(hipLaunchParm lp, T *twiddles_256, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(32*len))*dist_i;
	oOffset = (batch/(32*len))*dist_o;
	batch = batch%(32*len);

	iOffset += (batch/32)*stride_i + (batch%32)*2048;
	oOffset += (batch/32)*stride_o + (batch%32)*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_256_256_brc<T, SB_UNIT, dir>(twiddles_256, lwbIn, lwbOut, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_2048_bcc_d2_s1(hipLaunchParm lp, T *twiddles_64, T *twiddles_131072, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(128*len))*dist_i;
	oOffset = (batch/(128*len))*dist_o;
	batch = batch%(128*len);

	iOffset += (batch/128)*stride_i + (batch%128)*16;
	oOffset += (batch/128)*stride_o + (batch%128)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_2048_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_131072, lwbIn, lwbOut, batch, 1, 1);
}


template<typename T,  int dir>
__global__
void fft_64_4096_bcc_d2_s1(hipLaunchParm lp, T *twiddles_64, T *twiddles_262144, T * gbIn, T * gbOut, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint batch = hipBlockIdx_x;

	uint iOffset;
	uint oOffset;
	T *lwbIn;
	T *lwbOut;

	iOffset = (batch/(256*len))*dist_i;
	oOffset = (batch/(256*len))*dist_o;
	batch = batch%(256*len);

	iOffset += (batch/256)*stride_i + (batch%256)*16;
	oOffset += (batch/256)*stride_o + (batch%256)*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_64_4096_bcc<T, SB_UNIT, dir>(twiddles_64, twiddles_262144, lwbIn, lwbOut, batch, 1, 1);
}

//////////////////////////////////////////////////////////////////////////




#endif // POW2_LARGE_ENTRY_HIP_H

