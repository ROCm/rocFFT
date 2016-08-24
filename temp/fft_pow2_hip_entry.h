
#include "fft_pow2_hip.h"

__global__
void fft_1_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*1;
	
	fft_1(lwb, rw);
}


template <int dir>
__global__
void fft_2_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*2;
	
	fft_2<SB_UNIT>(twiddles, lwb, rw, 1);
}	


template <int dir>
__global__
void fft_4_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*4;
	float *ldsp = lds + (me/2)*4;
	
	fft_4<SB_UNIT, dir>(twiddles, lwb, ldsp, me%2, rw, 1);
}



template <int dir>
__global__
void fft_8_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*8;
	float *ldsp = lds + (me/2)*8;
	
	fft_8<SB_UNIT, dir>(twiddles, lwb, ldsp, me%2, rw, 1);
}



template <int dir>
__global__
void fft_16_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*16;
	float *ldsp = lds + (me/4)*16;
	
	fft_16<SB_UNIT, dir>(twiddles, lwb, ldsp, me%4, rw, 1);	
}



template <int dir>
__global__
void fft_32_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*32;
	float *ldsp = lds + (me/4)*32;
	
	fft_32<SB_UNIT, dir>(twiddles, lwb, ldsp, me%4, rw, 1);	
}


template <int dir>
__global__
void fft_64_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*64;
	float *ldsp = lds + (me/16)*64;
	
	fft_64<SB_UNIT, dir>(twiddles, lwb, ldsp, me%16, rw, 1);	
}


template <int dir>
__global__
void fft_128_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*128;
	float *ldsp = lds + (me/16)*128;
	
	fft_128<SB_UNIT, dir>(twiddles, lwb, ldsp, me%16, rw, 1);
}


template <int dir>
__global__
void fft_256_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];

	float2 *lwb = buffer + batch*256;
	
	fft_256<SB_UNIT, dir>(twiddles, lwb, lds, me, 1);
}


template <int dir>
__global__
void fft_512_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	
	float2 *lwb = buffer + batch*512;
	
	fft_512<SB_UNIT, dir>(twiddles, lwb, lds, me, 1);
}
	

template <int dir>
__global__
void fft_1024_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[1024];
	
	float2 *lwb = buffer + batch*1024;
	
	fft_1024<SB_UNIT, dir>(twiddles, lwb, lds, me, 1);
}


	
template <int dir>
__global__
void fft_2048_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[2048];
	
	float2 *lwb = buffer + batch*2048;
	
	fft_2048<SB_UNIT, dir>(twiddles, lwb, lds, me, 1);
}

template <int dir>
__global__
void fft_4096_d1_pk(hipLaunchParm lp, float2 *twiddles, float2 *buffer)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[4096];
	
	float2 *lwb = buffer + batch*4096;
	
	fft_4096<SB_UNIT, dir>(twiddles, lwb, lds, me, 1);
}


