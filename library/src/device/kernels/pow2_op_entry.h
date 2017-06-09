/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_OP_ENTRY_HIP_H
#define POW2_OP_ENTRY_HIP_H

#include "common.h"
#include "pow2.h"

template <typename T>
__global__
void fft_1_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*64 + me)*1;
	T *lwb_o = buffer_o + (batch*64 + me)*1;
	
	fft_1<T>(lwb_i, lwb_o, rw);
}

template <typename T>
__global__
void fft_2_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*64 + me)*2;
	T *lwb_o = buffer_o + (batch*64 + me)*2;
	
	fft_2<T, SB_UNIT>(twiddles, lwb_i, lwb_o, rw, 1, 1);
}	


template <typename T,  int dir>
__global__
void fft_4_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*32 + (me/2))*4;
	T *lwb_o = buffer_o + (batch*32 + (me/2))*4;
	real_type_t<T>  *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <typename T,  int dir>
__global__
void fft_8_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*32 + (me/2))*8;
	T *lwb_o = buffer_o + (batch*32 + (me/2))*8;
	real_type_t<T>  *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <typename T,  int dir>
__global__
void fft_16_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*16 + (me/4))*16;
	T *lwb_o = buffer_o + (batch*16 + (me/4))*16;
	real_type_t<T>  *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}



template <typename T,  int dir>
__global__
void fft_32_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*16 + (me/4))*32;
	T *lwb_o = buffer_o + (batch*16 + (me/4))*32;
	real_type_t<T>  *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}


template <typename T,  int dir>
__global__
void fft_64_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*4 + (me/16))*64;
	T *lwb_o = buffer_o + (batch*4 + (me/16))*64;
	real_type_t<T>  *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);	
}


template <typename T,  int dir>
__global__
void fft_128_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*4 + (me/16))*128;
	T *lwb_o = buffer_o + (batch*4 + (me/16))*128;
	real_type_t<T>  *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);
}


template <typename T,  int dir>
__global__
void fft_256_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	T *lwb_i = buffer_i + batch*256;
	T *lwb_o = buffer_o + batch*256;
	
	fft_256<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


template <typename T,  int dir>
__global__
void fft_512_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];
	
	T *lwb_i = buffer_i + batch*512;
	T *lwb_o = buffer_o + batch*512;
	
	fft_512<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}
	

template <typename T,  int dir>
__global__
void fft_1024_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[1024];
	
	T *lwb_i = buffer_i + batch*1024;
	T *lwb_o = buffer_o + batch*1024;
	
	fft_1024<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


	
template <typename T,  int dir>
__global__
void fft_2048_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[2048];
	
	T *lwb_i = buffer_i + batch*2048;
	T *lwb_o = buffer_o + batch*2048;
	
	fft_2048<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

template <typename T,  int dir>
__global__
void fft_4096_op_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[4096];
	
	T *lwb_i = buffer_i + batch*4096;
	T *lwb_o = buffer_o + batch*4096;
	
	fft_4096<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


/////////////////////////////////////////////////////
template <typename T>
__global__
void fft_1_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*64 + me)*dist_i;
	T *lwb_o = buffer_o + (batch*64 + me)*dist_o;
	
	fft_1<T>(lwb_i, lwb_o, rw);
}

template <typename T>
__global__
void fft_2_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*64 + me)*dist_i;
	T *lwb_o = buffer_o + (batch*64 + me)*dist_o;
	
	fft_2<T, SB_NONUNIT>(twiddles, lwb_i, lwb_o, rw, stride_i, stride_o);
}	


template <typename T,  int dir>
__global__
void fft_4_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*32 + (me/2))*dist_i;
	T *lwb_o = buffer_o + (batch*32 + (me/2))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, stride_i, stride_o);
}



template <typename T,  int dir>
__global__
void fft_8_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*32 + (me/2))*dist_i;
	T *lwb_o = buffer_o + (batch*32 + (me/2))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, stride_i, stride_o);
}



template <typename T,  int dir>
__global__
void fft_16_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*16 + (me/4))*dist_i;
	T *lwb_o = buffer_o + (batch*16 + (me/4))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, stride_i, stride_o);	
}



template <typename T,  int dir>
__global__
void fft_32_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*16 + (me/4))*dist_i;
	T *lwb_o = buffer_o + (batch*16 + (me/4))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, stride_i, stride_o);	
}


template <typename T,  int dir>
__global__
void fft_64_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*4 + (me/16))*dist_i;
	T *lwb_o = buffer_o + (batch*4 + (me/16))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, stride_i, stride_o);	
}


template <typename T,  int dir>
__global__
void fft_128_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + (batch*4 + (me/16))*dist_i;
	T *lwb_o = buffer_o + (batch*4 + (me/16))*dist_o;
	real_type_t<T>  *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, stride_i, stride_o);
}


template <typename T,  int dir>
__global__
void fft_256_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	T *lwb_i = buffer_i + batch*dist_i;
	T *lwb_o = buffer_o + batch*dist_o;
	
	fft_256<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


template <typename T,  int dir>
__global__
void fft_512_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];
	
	T *lwb_i = buffer_i + batch*dist_i;
	T *lwb_o = buffer_o + batch*dist_o;
	
	fft_512<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}
	

template <typename T,  int dir>
__global__
void fft_1024_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[1024];
	
	T *lwb_i = buffer_i + batch*dist_i;
	T *lwb_o = buffer_o + batch*dist_o;
	
	fft_1024<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


	
template <typename T,  int dir>
__global__
void fft_2048_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[2048];
	
	T *lwb_i = buffer_i + batch*dist_i;
	T *lwb_o = buffer_o + batch*dist_o;
	
	fft_2048<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}

template <typename T,  int dir>
__global__
void fft_4096_op_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[4096];
	
	T *lwb_i = buffer_i + batch*dist_i;
	T *lwb_o = buffer_o + batch*dist_o;
	
	fft_4096<T, SB_NONUNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, stride_i, stride_o);
}


/////////////////////////////////////////////////////////////

template <typename T>
__global__
void fft_1_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*64 + me)/len)*dist_i + ((batch*64 + me)%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*64 + me)/len)*dist_o + ((batch*64 + me)%len)*stride_o;
	
	fft_1<T>(lwb_i, lwb_o, rw);
}

template <typename T>
__global__
void fft_2_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*64 + me)/len)*dist_i + ((batch*64 + me)%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*64 + me)/len)*dist_o + ((batch*64 + me)%len)*stride_o;
	
	fft_2<T, SB_UNIT>(twiddles, lwb_i, lwb_o, rw, 1, 1);
}	


template <typename T,  int dir>
__global__
void fft_4_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[128];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*32 + (me/2))/len)*dist_i + ((batch*32 + (me/2))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*32 + (me/2))/len)*dist_o + ((batch*32 + (me/2))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <typename T,  int dir>
__global__
void fft_8_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*32 + (me/2))/len)*dist_i + ((batch*32 + (me/2))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*32 + (me/2))/len)*dist_o + ((batch*32 + (me/2))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%2, rw, 1, 1);
}



template <typename T,  int dir>
__global__
void fft_16_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*16 + (me/4))/len)*dist_i + ((batch*16 + (me/4))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*16 + (me/4))/len)*dist_o + ((batch*16 + (me/4))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}



template <typename T,  int dir>
__global__
void fft_32_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*16 + (me/4))/len)*dist_i + ((batch*16 + (me/4))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*16 + (me/4))/len)*dist_o + ((batch*16 + (me/4))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%4, rw, 1, 1);	
}


template <typename T,  int dir>
__global__
void fft_64_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*4 + (me/16))/len)*dist_i + ((batch*4 + (me/16))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*4 + (me/16))/len)*dist_o + ((batch*4 + (me/16))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);	
}


template <typename T,  int dir>
__global__
void fft_128_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const uint count, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	T *lwb_i = buffer_i + ((batch*4 + (me/16))/len)*dist_i + ((batch*4 + (me/16))%len)*stride_i;
	T *lwb_o = buffer_o + ((batch*4 + (me/16))/len)*dist_o + ((batch*4 + (me/16))%len)*stride_o;
	real_type_t<T>  *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, ldsp, me%16, rw, 1, 1);
}


template <typename T,  int dir>
__global__
void fft_256_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[256];

	T *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	T *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_256<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


template <typename T,  int dir>
__global__
void fft_512_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[512];
	
	T *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	T *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_512<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}
	

template <typename T,  int dir>
__global__
void fft_1024_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[1024];
	
	T *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	T *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_1024<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}


	
template <typename T,  int dir>
__global__
void fft_2048_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[2048];
	
	T *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	T *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_2048<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

template <typename T,  int dir>
__global__
void fft_4096_op_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer_i, T *buffer_o, const ulong len, const ulong stride_i, const ulong stride_o, const ulong dist_i, const ulong dist_o)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T>  lds[4096];
	
	T *lwb_i = buffer_i + (batch/len)*dist_i + (batch%len)*stride_i;
	T *lwb_o = buffer_o + (batch/len)*dist_o + (batch%len)*stride_o;
	
	fft_4096<T, SB_UNIT, dir>(twiddles, lwb_i, lwb_o, lds, me, 1, 1);
}

#endif // POW2_OP_ENTRY_HIP_H

