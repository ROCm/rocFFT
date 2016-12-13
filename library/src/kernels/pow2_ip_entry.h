/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_IP_ENTRY_HIP_H
#define POW2_IP_ENTRY_HIP_H

#include "common.h"
#include "pow2.h"

template<typename T>
__global__
void fft_1_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb = buffer + (batch*64 + me)*1;
	
	fft_1<T>(lwb, lwb, rw);
}

template<typename T>
__global__
void fft_2_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb = buffer + (batch*64 + me)*2;
	
	fft_2<T, SB_UNIT>(twiddles, lwb, lwb, rw, 1, 1);
}	


template <typename T, int dir>
__global__
void fft_4_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + (batch*32 + (me/2))*4;
	real_type_t<T> *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <typename T, int dir>
__global__
void fft_8_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + (batch*32 + (me/2))*8;
	real_type_t<T> *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <typename T, int dir>
__global__
void fft_16_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + (batch*16 + (me/4))*16;
	real_type_t<T> *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}



template <typename T, int dir>
__global__
void fft_32_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + (batch*16 + (me/4))*32;
	real_type_t<T> *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}


template <typename T, int dir>
__global__
void fft_64_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + (batch*4 + (me/16))*64;
	real_type_t<T> *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);	
}


template <typename T, int dir>
__global__
void fft_128_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + (batch*4 + (me/16))*128;
	real_type_t<T> *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);
}


template <typename T, int dir>
__global__
void fft_256_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	T *lwb = buffer + batch*256;
	
	fft_256<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


template <typename T, int dir>
__global__
void fft_512_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];
	
	T *lwb = buffer + batch*512;
	
	fft_512<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}
	

template <typename T, int dir>
__global__
void fft_1024_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[1024];
	
	T *lwb = buffer + batch*1024;
	
	fft_1024<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


	
template <typename T, int dir>
__global__
void fft_2048_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[2048];
	
	T *lwb = buffer + batch*2048;
	
	fft_2048<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

template <typename T, int dir>
__global__
void fft_4096_ip_d1_pk(hipLaunchParm lp, T *twiddles, T *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[4096];
	
	T *lwb = buffer + batch*4096;
	
	fft_4096<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

//////////////////////////////////////////////
template<typename T>
__global__
void fft_1_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb = buffer + (batch*64 + me)*dist;
	
	fft_1(lwb, lwb, rw);
}

template<typename T>
__global__
void fft_2_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	T *lwb = buffer + (batch*64 + me)*dist;
	
	fft_2<T, SB_NONUNIT>(twiddles, lwb, lwb, rw, stride, stride);
}	


template <typename T, int dir>
__global__
void fft_4_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + (batch*32 + (me/2))*dist;
	real_type_t<T> *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, stride, stride);
}



template <typename T, int dir>
__global__
void fft_8_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + (batch*32 + (me/2))*dist;
	real_type_t<T> *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, stride, stride);
}



template <typename T, int dir>
__global__
void fft_16_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + (batch*16 + (me/4))*dist;
	real_type_t<T> *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, stride, stride);	
}



template <typename T, int dir>
__global__
void fft_32_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + (batch*16 + (me/4))*dist;
	real_type_t<T> *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, stride, stride);	
}


template <typename T, int dir>
__global__
void fft_64_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + (batch*4 + (me/16))*dist;
	real_type_t<T> *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, stride, stride);	
}


template <typename T, int dir>
__global__
void fft_128_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + (batch*4 + (me/16))*dist;
	real_type_t<T> *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, stride, stride);
}


template <typename T, int dir>
__global__
void fft_256_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	T *lwb = buffer + batch*dist;
	
	fft_256<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}


template <typename T, int dir>
__global__
void fft_512_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];
	
	T *lwb = buffer + batch*dist;
	
	fft_512<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}
	

template <typename T, int dir>
__global__
void fft_1024_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[1024];
	
	T *lwb = buffer + batch*dist;
	
	fft_1024<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}


	
template <typename T, int dir>
__global__
void fft_2048_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[2048];
	
	T *lwb = buffer + batch*dist;
	
	fft_2048<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}

template <typename T, int dir>
__global__
void fft_4096_ip_d1_gn(hipLaunchParm lp, T *twiddles, T *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[4096];
	
	T *lwb = buffer + batch*dist;
	
	fft_4096<T, SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}


/////////////////////////////////////////////////////////

template<typename T>
__global__
void fft_1_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	T *lwb = buffer + ((batch*64 + me)/len)*dist + ((batch*64 + me)%len)*stride;
	
	fft_1(lwb, lwb, rw);
}

template<typename T>
__global__
void fft_2_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count*len - batch*64)) ? 1 : 0;

	T *lwb = buffer + ((batch*64 + me)/len)*dist + ((batch*64 + me)%len)*stride;
	
	fft_2<T, SB_UNIT>(twiddles, lwb, lwb, rw, 1, 1);
}	


template <typename T, int dir>
__global__
void fft_4_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[128];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + ((batch*32 + (me/2))/len)*dist + ((batch*32 + (me/2))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/2)*4;
	
	fft_4<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <typename T, int dir>
__global__
void fft_8_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count*len - batch*32)*2) ? 1 : 0;

	T *lwb = buffer + ((batch*32 + (me/2))/len)*dist + ((batch*32 + (me/2))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/2)*8;
	
	fft_8<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <typename T, int dir>
__global__
void fft_16_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + ((batch*16 + (me/4))/len)*dist + ((batch*16 + (me/4))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/4)*16;
	
	fft_16<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}



template <typename T, int dir>
__global__
void fft_32_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count*len - batch*16)*4) ? 1 : 0;

	T *lwb = buffer + ((batch*16 + (me/4))/len)*dist + ((batch*16 + (me/4))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/4)*32;
	
	fft_32<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}


template <typename T, int dir>
__global__
void fft_64_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + ((batch*4 + (me/16))/len)*dist + ((batch*4 + (me/16))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/16)*64;
	
	fft_64<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);	
}


template <typename T, int dir>
__global__
void fft_128_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const uint count, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];

	uint rw = (me < (count*len - batch*4)*16) ? 1 : 0;

	T *lwb = buffer + ((batch*4 + (me/16))/len)*dist + ((batch*4 + (me/16))%len)*stride;
	real_type_t<T> *ldsp = lds + (me/16)*128;
	
	fft_128<T, SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);
}


template <typename T, int dir>
__global__
void fft_256_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[256];

	T *lwb = buffer + (batch/len)*dist + (batch%len)*stride;
	
	fft_256<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


template <typename T, int dir>
__global__
void fft_512_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[512];
	
	T *lwb = buffer + (batch/len)*dist + (batch%len)*stride;
	
	fft_512<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}
	

template <typename T, int dir>
__global__
void fft_1024_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[1024];
	
	T *lwb = buffer + (batch/len)*dist + (batch%len)*stride;
	
	fft_1024<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


	
template <typename T, int dir>
__global__
void fft_2048_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[2048];
	
	T *lwb = buffer + (batch/len)*dist + (batch%len)*stride;
	
	fft_2048<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

template <typename T, int dir>
__global__
void fft_4096_ip_d2_s1(hipLaunchParm lp, T *twiddles, T *buffer, const ulong len, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ real_type_t<T> lds[4096];
	
	T *lwb = buffer + (batch/len)*dist + (batch%len)*stride;
	
	fft_4096<T, SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

#endif // POW2_IP_ENTRY_HIP_H


