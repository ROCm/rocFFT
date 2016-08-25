#ifndef FFT_POW2_IP_ENTRY_HIP_H
#define FFT_POW2_IP_ENTRY_HIP_H

#include "fft_pow2_hip.h"

__global__
void fft_1_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*1;
	
	fft_1(lwb, lwb, rw);
}


__global__
void fft_2_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*2;
	
	fft_2<SB_UNIT>(twiddles, lwb, lwb, rw, 1, 1);
}	


template <int dir>
__global__
void fft_4_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*4;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_8_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*8;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, 1, 1);
}



template <int dir>
__global__
void fft_16_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*16;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}



template <int dir>
__global__
void fft_32_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*32;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, 1, 1);	
}


template <int dir>
__global__
void fft_64_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*64;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);	
}


template <int dir>
__global__
void fft_128_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*128;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_UNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, 1, 1);
}


template <int dir>
__global__
void fft_256_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb = buffer + batch*256;
	
	fft_256<SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


template <int dir>
__global__
void fft_512_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb = buffer + batch*512;
	
	fft_512<SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}
	

template <int dir>
__global__
void fft_1024_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb = buffer + batch*1024;
	
	fft_1024<SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}


	
template <int dir>
__global__
void fft_2048_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb = buffer + batch*2048;
	
	fft_2048<SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

template <int dir>
__global__
void fft_4096_ip_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb = buffer + batch*4096;
	
	fft_4096<SB_UNIT, dir>(twiddles, lwb, lwb, lds, me, 1, 1);
}

//////////////////////////////////////////////

__global__
void fft_1_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*dist;
	
	fft_1(lwb, lwb, rw);
}


__global__
void fft_2_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*dist;
	
	fft_2<SB_NONUNIT>(twiddles, lwb, lwb, rw, stride, stride);
}	


template <int dir>
__global__
void fft_4_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*dist;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, stride, stride);
}



template <int dir>
__global__
void fft_8_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*dist;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%2, rw, stride, stride);
}



template <int dir>
__global__
void fft_16_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*dist;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, stride, stride);	
}



template <int dir>
__global__
void fft_32_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*dist;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%4, rw, stride, stride);	
}


template <int dir>
__global__
void fft_64_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*dist;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, stride, stride);	
}


template <int dir>
__global__
void fft_128_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*dist;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_NONUNIT, dir>(twiddles, lwb, lwb, ldsp, me%16, rw, stride, stride);
}


template <int dir>
__global__
void fft_256_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb = buffer + batch*dist;
	
	fft_256<SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}


template <int dir>
__global__
void fft_512_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb = buffer + batch*dist;
	
	fft_512<SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}
	

template <int dir>
__global__
void fft_1024_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb = buffer + batch*dist;
	
	fft_1024<SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}


	
template <int dir>
__global__
void fft_2048_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb = buffer + batch*dist;
	
	fft_2048<SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}

template <int dir>
__global__
void fft_4096_ip_d1_gn(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const ulong stride, const ulong dist)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb = buffer + batch*dist;
	
	fft_4096<SB_NONUNIT, dir>(twiddles, lwb, lwb, lds, me, stride, stride);
}

#endif // FFT_POW2_IP_ENTRY_HIP_H

