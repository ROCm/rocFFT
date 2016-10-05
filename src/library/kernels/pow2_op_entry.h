/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_OP_ENTRY_HIP_H
#define POW2_OP_ENTRY_HIP_H

#include "pow2.h"

__global__
void fft_1_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*64 + me)*1;
	float2 *lwb_o = buffer_o + (batch*64 + me)*1;
	
	fft_1(lwb_i, lwb_o, rw);
}


__global__
void fft_2_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*64 + me)*2;
	float2 *lwb_o = buffer_o + (batch*64 + me)*2;
	
	fft_2<SB_UNIT>(twiddles, lwb_i, lwb_o, rw, 1, 1);
}	


template <int dir>
__global__
void fft_4_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*32 + (me/2))*4;
	float2 *lwb_o = buffer_o + (batch*32 + (me/2))*4;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_8_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*32 + (me/2))*8;
	float2 *lwb_o = buffer_o + (batch*32 + (me/2))*8;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_16_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*16 + (me/4))*16;
	float2 *lwb_o = buffer_o + (batch*16 + (me/4))*16;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}



template <int dir>
__global__
void fft_32_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*16 + (me/4))*32;
	float2 *lwb_o = buffer_o + (batch*16 + (me/4))*32;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}


template <int dir>
__global__
void fft_64_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*4 + (me/16))*64;
	float2 *lwb_o = buffer_o + (batch*4 + (me/16))*64;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);	
}


template <int dir>
__global__
void fft_128_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*4 + (me/16))*128;
	float2 *lwb_o = buffer_o + (batch*4 + (me/16))*128;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);
}


template <int dir>
__global__
void fft_256_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb_i = buffer_i + batch*256;
	float2 *lwb_o = buffer_o + batch*256;
	
	fft_256<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


template <int dir>
__global__
void fft_512_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb_i = buffer_i + batch*512;
	float2 *lwb_o = buffer_o + batch*512;
	
	fft_512<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}
	

template <int dir>
__global__
void fft_1024_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb_i = buffer_i + batch*1024;
	float2 *lwb_o = buffer_o + batch*1024;
	
	fft_1024<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


	
template <int dir>
__global__
void fft_2048_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb_i = buffer_i + batch*2048;
	float2 *lwb_o = buffer_o + batch*2048;
	
	fft_2048<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

template <int dir>
__global__
void fft_4096_op_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb_i = buffer_i + batch*4096;
	float2 *lwb_o = buffer_o + batch*4096;
	
	fft_4096<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


/////////////////////////////////////////////////////

__global__
void fft_1_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*64 + me)*dist_i;
	float2 *lwb_o = buffer_o + (batch*64 + me)*dist_o;
	
	fft_1(lwb_i, lwb_o, rw);
}


__global__
void fft_2_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*64 + me)*dist_i;
	float2 *lwb_o = buffer_o + (batch*64 + me)*dist_o;
	
	fft_2<SB_NONUNIT>(twiddles, lwb_i, lwb_o, rw, stride_i, stride_o);
}	


template <int dir>
__global__
void fft_4_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*32 + (me/2))*dist_i;
	float2 *lwb_o = buffer_o + (batch*32 + (me/2))*dist_o;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, stride_i, stride_o);
}



template <int dir>
__global__
void fft_8_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*32 + (me/2))*dist_i;
	float2 *lwb_o = buffer_o + (batch*32 + (me/2))*dist_o;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, stride_i, stride_o);
}



template <int dir>
__global__
void fft_16_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*16 + (me/4))*dist_i;
	float2 *lwb_o = buffer_o + (batch*16 + (me/4))*dist_o;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, stride_i, stride_o);	
}



template <int dir>
__global__
void fft_32_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*16 + (me/4))*dist_i;
	float2 *lwb_o = buffer_o + (batch*16 + (me/4))*dist_o;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, stride_i, stride_o);	
}


template <int dir>
__global__
void fft_64_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*4 + (me/16))*dist_i;
	float2 *lwb_o = buffer_o + (batch*4 + (me/16))*dist_o;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, stride_i, stride_o);	
}


template <int dir>
__global__
void fft_128_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + (batch*4 + (me/16))*dist_i;
	float2 *lwb_o = buffer_o + (batch*4 + (me/16))*dist_o;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, stride_i, stride_o);
}


template <int dir>
__global__
void fft_256_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb_i = buffer_i + batch*dist_i;
	float2 *lwb_o = buffer_o + batch*dist_o;
	
	fft_256<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


template <int dir>
__global__
void fft_512_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb_i = buffer_i + batch*dist_i;
	float2 *lwb_o = buffer_o + batch*dist_o;
	
	fft_512<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}
	

template <int dir>
__global__
void fft_1024_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb_i = buffer_i + batch*dist_i;
	float2 *lwb_o = buffer_o + batch*dist_o;
	
	fft_1024<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


	
template <int dir>
__global__
void fft_2048_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb_i = buffer_i + batch*dist_i;
	float2 *lwb_o = buffer_o + batch*dist_o;
	
	fft_2048<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}

template <int dir>
__global__
void fft_4096_op_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb_i = buffer_i + batch*dist_i;
	float2 *lwb_o = buffer_o + batch*dist_o;
	
	fft_4096<SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


/////////////////////////////////////////////////////////////


__global__
void fft_1_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*64 + me)/len)*dist_i + ((batch*64 + me)%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*64 + me)/len)*dist_o + ((batch*64 + me)%len)*stride_o;
	
	fft_1(lwb_i, lwb_o, rw);
}


__global__
void fft_2_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*64 + me)/len)*dist_i + ((batch*64 + me)%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*64 + me)/len)*dist_o + ((batch*64 + me)%len)*stride_o;
	
	fft_2<SB_UNIT>(twiddles, lwb_i, lwb_o, rw, 1, 1);
}	


template <int dir>
__global__
void fft_4_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*32 + (me/2))/len)*dist_i + ((batch*32 + (me/2))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*32 + (me/2))/len)*dist_o + ((batch*32 + (me/2))%len)*stride_o;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_8_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*32 + (me/2))/len)*dist_i + ((batch*32 + (me/2))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*32 + (me/2))/len)*dist_o + ((batch*32 + (me/2))%len)*stride_o;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_16_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*16 + (me/4))/len)*dist_i + ((batch*16 + (me/4))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*16 + (me/4))/len)*dist_o + ((batch*16 + (me/4))%len)*stride_o;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}



template <int dir>
__global__
void fft_32_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*16 + (me/4))/len)*dist_i + ((batch*16 + (me/4))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*16 + (me/4))/len)*dist_o + ((batch*16 + (me/4))%len)*stride_o;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}


template <int dir>
__global__
void fft_64_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*4 + (me/16))/len)*dist_i + ((batch*4 + (me/16))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*4 + (me/16))/len)*dist_o + ((batch*4 + (me/16))%len)*stride_o;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);	
}


template <int dir>
__global__
void fft_128_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	float2 *lwb_i = buffer_i + ((batch*4 + (me/16))/len)*dist_i + ((batch*4 + (me/16))%len)*stride_i;
	float2 *lwb_o = buffer_o + ((batch*4 + (me/16))/len)*dist_o + ((batch*4 + (me/16))%len)*stride_o;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);
}


template <int dir>
__global__
void fft_256_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	float2 *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_256<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


template <int dir>
__global__
void fft_512_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	float2 *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_512<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}
	

template <int dir>
__global__
void fft_1024_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	float2 *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_1024<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


	
template <int dir>
__global__
void fft_2048_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	float2 *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_2048<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

template <int dir>
__global__
void fft_4096_op_d2_s1(hipLaunchParm lp, float2 *twiddles, float2 *buffer_i, float2 *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	float2 *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_4096<SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

#endif // POW2_OP_ENTRY_HIP_H

