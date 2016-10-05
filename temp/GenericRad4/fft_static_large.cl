/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "twiddles_pow2.cl"
#include "twiddles_pow2_large.cl"
#include "butterfly.cl"

__attribute__((always_inline)) float2
TW3step_8192(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_8192[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_8192[1][j].x - result.y * twiddle_dee_8192[1][j].y),
		(result.y * twiddle_dee_8192[1][j].x + result.x * twiddle_dee_8192[1][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_16384(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_16384[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_16384[1][j].x - result.y * twiddle_dee_16384[1][j].y),
		(result.y * twiddle_dee_16384[1][j].x + result.x * twiddle_dee_16384[1][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_32768(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_32768[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_32768[1][j].x - result.y * twiddle_dee_32768[1][j].y),
		(result.y * twiddle_dee_32768[1][j].x + result.x * twiddle_dee_32768[1][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_65536(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_65536[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_65536[1][j].x - result.y * twiddle_dee_65536[1][j].y),
		(result.y * twiddle_dee_65536[1][j].x + result.x * twiddle_dee_65536[1][j].y));
	return result;
}

__attribute__((always_inline)) float2
TW3step_131072(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_131072[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_131072[1][j].x - result.y * twiddle_dee_131072[1][j].y),
		(result.y * twiddle_dee_131072[1][j].x + result.x * twiddle_dee_131072[1][j].y));
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_131072[2][j].x - result.y * twiddle_dee_131072[2][j].y),
		(result.y * twiddle_dee_131072[2][j].x + result.x * twiddle_dee_131072[2][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_262144(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_262144[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_262144[1][j].x - result.y * twiddle_dee_262144[1][j].y),
		(result.y * twiddle_dee_262144[1][j].x + result.x * twiddle_dee_262144[1][j].y));
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_262144[2][j].x - result.y * twiddle_dee_262144[2][j].y),
		(result.y * twiddle_dee_262144[2][j].x + result.x * twiddle_dee_262144[2][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_524288(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_524288[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_524288[1][j].x - result.y * twiddle_dee_524288[1][j].y),
		(result.y * twiddle_dee_524288[1][j].x + result.x * twiddle_dee_524288[1][j].y));
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_524288[2][j].x - result.y * twiddle_dee_524288[2][j].y),
		(result.y * twiddle_dee_524288[2][j].x + result.x * twiddle_dee_524288[2][j].y));
	return result;
}


__attribute__((always_inline)) float2
TW3step_1048576(size_t u)
{
	size_t j = u & 255;
	float2 result = twiddle_dee_1048576[0][j];
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_1048576[1][j].x - result.y * twiddle_dee_1048576[1][j].y),
		(result.y * twiddle_dee_1048576[1][j].x + result.x * twiddle_dee_1048576[1][j].y));
	u >>= 8;
	j = u & 255;
	result = (float2) ((result.x * twiddle_dee_1048576[2][j].x - result.y * twiddle_dee_1048576[2][j].y),
		(result.y * twiddle_dee_1048576[2][j].x + result.x * twiddle_dee_1048576[2][j].y));
	return result;
}



__attribute__((always_inline)) void
fft_64_8192(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;
	
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];		

	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_8192, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_8192, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_8192, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_8192, (me + 48)*b, X3)			
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_8192, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_INV(TW3step_8192, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_INV(TW3step_8192, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_INV(TW3step_8192, (me + 48)*b, X3)				
	}
	
	
	lds[me + 0]  = X0;
	lds[me + 16] = X1;	
	lds[me + 32] = X2;
	lds[me + 48] = X3;		

}


__attribute__((always_inline)) void
fft_128(uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	X0 = lds[me + 0];
	X1 = lds[me + 16];
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	X4 = lds[me + 64];
	X5 = lds[me + 80];
	X6 = lds[me + 96];
	X7 = lds[me + 112];

	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lds[me*8 + 0] = X0;
	lds[me*8 + 1] = X1;
	lds[me*8 + 2] = X2;
	lds[me*8 + 3] = X3;
	lds[me*8 + 4] = X4;
	lds[me*8 + 5] = X5;
	lds[me*8 + 6] = X6;
	lds[me*8 + 7] = X7;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);


	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 2, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 2, X7)		
	}
	

	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
	
	
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3;
	
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 2, X7)
	}	
	
	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
		
		
	{
		__local float4 *ldsv = lds;	
		ldsv[me +  0] = (float4)(X0,X4);
		ldsv[me + 16] = (float4)(X1,X5);	
		ldsv[me + 32] = (float4)(X2,X6);
		ldsv[me + 48] = (float4)(X3,X7);			
	}	
	
}


__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_8192_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint t=0; t<2; t++)
	{
		b = (batch%8)*16 + t*8 + (me/16);
		
		fft_64_8192(b, me%16, lds + t*512 + (me/16)*64, dir);
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*128 + t*1024] = R0;
	}

}



__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_8192_2(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);

	fft_128(me%16, lds + (me/16)*128, dir);

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*64 + t*1024] = R0;
	}

}


__attribute__((always_inline)) void
fft_64_16384(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;
	
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];		

	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_16384, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_16384, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_16384, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_16384, (me + 48)*b, X3)			
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_16384, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_INV(TW3step_16384, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_INV(TW3step_16384, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_INV(TW3step_16384, (me + 48)*b, X3)				
	}
	
	
	lds[me + 0]  = X0;
	lds[me + 16] = X1;	
	lds[me + 32] = X2;
	lds[me + 48] = X3;		

}


__attribute__((always_inline)) void
fft_256(uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;

	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];		
	

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 2, X3)		
	}

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		
		
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 2, X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	lds[(me/16)*64 + me%16 +  0] = X0;
	lds[(me/16)*64 + me%16 + 16] = X1;
	lds[(me/16)*64 + me%16 + 32] = X2;
	lds[(me/16)*64 + me%16 + 48] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 2, X3)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me +   0] = X0;
	lds[me +  64] = X1;	
	lds[me + 128] = X2;
	lds[me + 192] = X3;		

}


__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_16384_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint t=0; t<2; t++)
	{
		b = (batch%16)*16 + t*8 + (me/16);

		fft_64_16384(b, me%16, lds + t*512 + (me/16)*64, dir);

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*256 + t*2048] = R0;
	}

}

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_16384_2(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[2048];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{
		fft_256(me%64, lds + t*1024 + (me/64)*256, dir);

		barrier(CLK_LOCAL_MEM_FENCE);
	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*64 + t*2048] = R0;
	}

}


__attribute__((always_inline)) void
fft_128_32768(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	X0 = lds[me + 0];
	X1 = lds[me + 16];
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	X4 = lds[me + 64];
	X5 = lds[me + 80];
	X6 = lds[me + 96];
	X7 = lds[me + 112];

	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lds[me*8 + 0] = X0;
	lds[me*8 + 1] = X1;
	lds[me*8 + 2] = X2;
	lds[me*8 + 3] = X3;
	lds[me*8 + 4] = X4;
	lds[me*8 + 5] = X5;
	lds[me*8 + 6] = X6;
	lds[me*8 + 7] = X7;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);


	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_128, 7 + 3*((2*me + 1)%8) + 2, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_128, 7 + 3*((2*me + 1)%8) + 2, X7)		
	}
	

	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
	
	
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3;
	
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_128, 31 + 3*((2*me + 1)%32) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_128, 31 + 3*((2*me + 1)%32) + 2, X7)
	}	
	
	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
		
	
	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 0) +  0)*b, X0)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 0) + 32)*b, X1)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 0) + 64)*b, X2)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 0) + 96)*b, X3)	

		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 1) +  0)*b, X4)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 1) + 32)*b, X5)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 1) + 64)*b, X6)
		TWIDDLE_3STEP_MUL_FWD(TW3step_32768, ((2*me + 1) + 96)*b, X7)
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 0) +  0)*b, X0)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 0) + 32)*b, X1)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 0) + 64)*b, X2)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 0) + 96)*b, X3)	

		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 1) +  0)*b, X4)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 1) + 32)*b, X5)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 1) + 64)*b, X6)
		TWIDDLE_3STEP_MUL_INV(TW3step_32768, ((2*me + 1) + 96)*b, X7)
	}
	
	
	{
		__local float4 *ldsv = lds;	
		ldsv[me +  0] = (float4)(X0,X4);
		ldsv[me + 16] = (float4)(X1,X5);	
		ldsv[me + 32] = (float4)(X2,X6);
		ldsv[me + 48] = (float4)(X3,X7);			
	}	
	
}


__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_32768_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	b = (batch%32)*8 + (me/16);

	fft_128_32768(b, me%16, lds + (me/16)*128, dir);
	
	barrier(CLK_LOCAL_MEM_FENCE);

	
	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*4096] = R0;
	}
}


__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_32768_2(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[2048];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{
		fft_256(me%64, lds + t*1024 + (me/64)*256, dir);

		barrier(CLK_LOCAL_MEM_FENCE);

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*128 + t*4096] = R0;
	}

}


__attribute__((always_inline)) void
fft_256_65536(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;

	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];		
	

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 3 + 3*(me%4) + 2, X3)		
	}

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		
		
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 15 + 3*(me%16) + 2, X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 15 + 3*(me%16) + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	lds[(me/16)*64 + me%16 +  0] = X0;
	lds[(me/16)*64 + me%16 + 16] = X1;
	lds[(me/16)*64 + me%16 + 32] = X2;
	lds[(me/16)*64 + me%16 + 48] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_256, 63 + 3*me + 2, X3)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_256, 63 + 3*me + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_65536, (me +   0)*b, X0)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_65536, (me +  64)*b, X1)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_65536, (me + 128)*b, X2)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_65536, (me + 192)*b, X3)			
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_65536, (me +   0)*b, X0)	
		TWIDDLE_3STEP_MUL_INV(TW3step_65536, (me +  64)*b, X1)	
		TWIDDLE_3STEP_MUL_INV(TW3step_65536, (me + 128)*b, X2)	
		TWIDDLE_3STEP_MUL_INV(TW3step_65536, (me + 192)*b, X3)				
	}

	
	lds[me +   0] = X0;
	lds[me +  64] = X1;	
	lds[me + 128] = X2;
	lds[me + 192] = X3;		

}

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_65536_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[2048];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{

		b = (batch%32)*8 + t*4 + (me/64);

		fft_256_65536(b, me%64, lds + t*1024 + (me/64)*256, dir);

		barrier(CLK_LOCAL_MEM_FENCE);

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
	}
}

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_65536_2(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[2048];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{
		fft_256(me%64, lds + t*1024 + (me/64)*256, dir);

		barrier(CLK_LOCAL_MEM_FENCE);

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
	}

}

__attribute__((always_inline)) void
fft_64_131072(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;
	
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];		

	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_131072, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_131072, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_131072, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_131072, (me + 48)*b, X3)			
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_131072, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_INV(TW3step_131072, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_INV(TW3step_131072, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_INV(TW3step_131072, (me + 48)*b, X3)				
	}
	
	
	lds[me + 0]  = X0;
	lds[me + 16] = X1;	
	lds[me + 32] = X2;
	lds[me + 48] = X3;		

}


__attribute__((always_inline)) void
fft_2048(uint me, __local float *lds, __global float2 *lwb, const int dir)
{

	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	X0 = lwb[me +    0];
	X1 = lwb[me +  256];
	X2 = lwb[me +  512];
	X3 = lwb[me +  768];
	X4 = lwb[me + 1024];
	X5 = lwb[me + 1280];
	X6 = lwb[me + 1536];
	X7 = lwb[me + 1792];
					
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

	lds[me*8 + 0] = X0.x;
	lds[me*8 + 1] = X1.x;
	lds[me*8 + 2] = X2.x;
	lds[me*8 + 3] = X3.x;
	lds[me*8 + 4] = X4.x;
	lds[me*8 + 5] = X5.x;
	lds[me*8 + 6] = X6.x;
	lds[me*8 + 7] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[me +    0];
	X1.x = lds[me +  256];
	X2.x = lds[me +  512];
	X3.x = lds[me +  768];
	X4.x = lds[me + 1024];
	X5.x = lds[me + 1280];
	X6.x = lds[me + 1536];
	X7.x = lds[me + 1792];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[me +    0];
	X1.y = lds[me +  256];
	X2.y = lds[me +  512];
	X3.y = lds[me +  768];
	X4.y = lds[me + 1024];
	X5.y = lds[me + 1280];
	X6.y = lds[me + 1536];
	X7.y = lds[me + 1792];

		
	barrier(CLK_LOCAL_MEM_FENCE);


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_FWD(twiddles_2048, 7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_INV(twiddles_2048, 7 + 7*(me%8) + 6, X7)				

	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lds[(me/8)*64 + (me%8) +  0] = X0.x;
	lds[(me/8)*64 + (me%8) +  8] = X1.x;
	lds[(me/8)*64 + (me%8) + 16] = X2.x;
	lds[(me/8)*64 + (me%8) + 24] = X3.x;
	lds[(me/8)*64 + (me%8) + 32] = X4.x;
	lds[(me/8)*64 + (me%8) + 40] = X5.x;
	lds[(me/8)*64 + (me%8) + 48] = X6.x;
	lds[(me/8)*64 + (me%8) + 56] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[me +    0];
	X1.x = lds[me +  256];
	X2.x = lds[me +  512];
	X3.x = lds[me +  768];
	X4.x = lds[me + 1024];
	X5.x = lds[me + 1280];
	X6.x = lds[me + 1536];
	X7.x = lds[me + 1792];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[me +    0];
	X1.y = lds[me +  256];
	X2.y = lds[me +  512];
	X3.y = lds[me +  768];
	X4.y = lds[me + 1024];
	X5.y = lds[me + 1280];
	X6.y = lds[me + 1536];
	X7.y = lds[me + 1792];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_FWD(twiddles_2048, 63 + 7*(me%64) + 6, X7)			

	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_INV(twiddles_2048, 63 + 7*(me%64) + 6, X7)
	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

		
	lds[(me/64)*512 + (me%64) +   0] = X0.x;
	lds[(me/64)*512 + (me%64) +  64] = X1.x;
	lds[(me/64)*512 + (me%64) + 128] = X2.x;
	lds[(me/64)*512 + (me%64) + 192] = X3.x;
	lds[(me/64)*512 + (me%64) + 256] = X4.x;
	lds[(me/64)*512 + (me%64) + 320] = X5.x;
	lds[(me/64)*512 + (me%64) + 384] = X6.x;
	lds[(me/64)*512 + (me%64) + 448] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[(2*me + 0) +    0];
	X1.x = lds[(2*me + 0) +  512];
	X2.x = lds[(2*me + 0) + 1024];
	X3.x = lds[(2*me + 0) + 1536];

	X4.x = lds[(2*me + 1) +    0];
	X5.x = lds[(2*me + 1) +  512];
	X6.x = lds[(2*me + 1) + 1024];
	X7.x = lds[(2*me + 1) + 1536];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[(me/64)*512 + (me%64) +   0] = X0.y;
	lds[(me/64)*512 + (me%64) +  64] = X1.y;
	lds[(me/64)*512 + (me%64) + 128] = X2.y;
	lds[(me/64)*512 + (me%64) + 192] = X3.y;
	lds[(me/64)*512 + (me%64) + 256] = X4.y;
	lds[(me/64)*512 + (me%64) + 320] = X5.y;
	lds[(me/64)*512 + (me%64) + 384] = X6.y;
	lds[(me/64)*512 + (me%64) + 448] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[(2*me + 0) +    0];
	X1.y = lds[(2*me + 0) +  512]; 
	X2.y = lds[(2*me + 0) + 1024];
	X3.y = lds[(2*me + 0) + 1536];

	X4.y = lds[(2*me + 1) +    0];
	X5.y = lds[(2*me + 1) +  512];
	X6.y = lds[(2*me + 1) + 1024];
	X7.y = lds[(2*me + 1) + 1536];	

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_2048, 511 + 3*((2*me + 1)%512) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_2048, 511 + 3*((2*me + 1)%512) + 2, X7)	
	}	
	
	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
		
	{
		__global float4 *lwbv = lwb;	
		lwbv[me +   0] = (float4)(X0,X4);
		lwbv[me + 256] = (float4)(X1,X5);	
		lwbv[me + 512] = (float4)(X2,X6);
		lwbv[me + 768] = (float4)(X3,X7);			
	}
}

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_131072_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{

		b = (batch%128)*16 + t*8 + (me/16);

		fft_64_131072(b, me%16, lds + t*512 + (me/16)*64, dir);

		barrier(CLK_LOCAL_MEM_FENCE);

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*2048 + t*16384] = R0;
	}
}

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_131072_2(__global const float2 * restrict gb, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);
	
	__local float lds[2048];
	uint ioOffset;
	__global float2 *lwb;

	ioOffset = (batch/64)*131072 + (batch%64)*2048;
	lwb = gb + ioOffset;
	
	fft_2048(me, lds, lwb, dir);

}


// Local structure to embody/capture tile dimensions
typedef struct tag_Tile
{
   size_t x;
   size_t y;
} Tile;

__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_131072( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 1;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*131072;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 2048;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*131072;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 64;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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


__attribute__((always_inline)) void
fft_64_262144(uint b, uint me, __local float2 *lds, const int dir)
{
	float2 X0, X1, X2, X3;
	
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];		

	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0;
	lds[me*4 + 1] = X1;
	lds[me*4 + 2] = X2;
	lds[me*4 + 3] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	lds[(me/4)*16 + me%4 +  0] = X0;
	lds[(me/4)*16 + me%4 +  4] = X1;
	lds[(me/4)*16 + me%4 +  8] = X2;
	lds[(me/4)*16 + me%4 + 12] = X3;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_64, 15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_64, 15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	if(dir == -1)
	{
		TWIDDLE_3STEP_MUL_FWD(TW3step_262144, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_262144, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_262144, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_FWD(TW3step_262144, (me + 48)*b, X3)			
	}
	else
	{
		TWIDDLE_3STEP_MUL_INV(TW3step_262144, (me +  0)*b, X0)	
		TWIDDLE_3STEP_MUL_INV(TW3step_262144, (me + 16)*b, X1)	
		TWIDDLE_3STEP_MUL_INV(TW3step_262144, (me + 32)*b, X2)	
		TWIDDLE_3STEP_MUL_INV(TW3step_262144, (me + 48)*b, X3)				
	}
	
	
	lds[me + 0]  = X0;
	lds[me + 16] = X1;	
	lds[me + 32] = X2;
	lds[me + 48] = X3;		

}


__attribute__((always_inline)) void
fft_4096(uint me, __local float *lds, __global float2 *lwb, const int dir)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	float2 X8, X9, X10, X11, X12, X13, X14, X15;
	

	 X0 = lwb[me +    0];
	 X1 = lwb[me +  256];
	 X2 = lwb[me +  512];
	 X3 = lwb[me +  768];
	 X4 = lwb[me + 1024];
	 X5 = lwb[me + 1280];
	 X6 = lwb[me + 1536];
	 X7 = lwb[me + 1792];
	 X8 = lwb[me + 2048];
	 X9 = lwb[me + 2304];
	X10 = lwb[me + 2560];
	X11 = lwb[me + 2816];
	X12 = lwb[me + 3072];
	X13 = lwb[me + 3328];
	X14 = lwb[me + 3584];
	X15 = lwb[me + 3840];
	
	if(dir == -1)
	{
		FwdRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	
	

	lds[me*16 +  0] =  X0.x;
	lds[me*16 +  1] =  X1.x;
	lds[me*16 +  2] =  X2.x;
	lds[me*16 +  3] =  X3.x;
	lds[me*16 +  4] =  X4.x;
	lds[me*16 +  5] =  X5.x;
	lds[me*16 +  6] =  X6.x;
	lds[me*16 +  7] =  X7.x;
	lds[me*16 +  8] =  X8.x;
	lds[me*16 +  9] =  X9.x;
	lds[me*16 + 10] = X10.x;
	lds[me*16 + 11] = X11.x;
	lds[me*16 + 12] = X12.x;
	lds[me*16 + 13] = X13.x;
	lds[me*16 + 14] = X14.x;
	lds[me*16 + 15] = X15.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	 X0.x = lds[me +    0];
	 X1.x = lds[me +  256];
	 X2.x = lds[me +  512];
	 X3.x = lds[me +  768];
	 X4.x = lds[me + 1024];
	 X5.x = lds[me + 1280];
	 X6.x = lds[me + 1536];
	 X7.x = lds[me + 1792];
	 X8.x = lds[me + 2048];
	 X9.x = lds[me + 2304];
	X10.x = lds[me + 2560];
	X11.x = lds[me + 2816];
	X12.x = lds[me + 3072];
	X13.x = lds[me + 3328];
	X14.x = lds[me + 3584];
	X15.x = lds[me + 3840];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[me*16 +  0] =  X0.y;
	lds[me*16 +  1] =  X1.y;
	lds[me*16 +  2] =  X2.y;
	lds[me*16 +  3] =  X3.y;
	lds[me*16 +  4] =  X4.y;
	lds[me*16 +  5] =  X5.y;
	lds[me*16 +  6] =  X6.y;
	lds[me*16 +  7] =  X7.y;
	lds[me*16 +  8] =  X8.y;
	lds[me*16 +  9] =  X9.y;
	lds[me*16 + 10] = X10.y;
	lds[me*16 + 11] = X11.y;
	lds[me*16 + 12] = X12.y;
	lds[me*16 + 13] = X13.y;
	lds[me*16 + 14] = X14.y;
	lds[me*16 + 15] = X15.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	 X0.y = lds[me +    0];
	 X1.y = lds[me +  256];
	 X2.y = lds[me +  512];
	 X3.y = lds[me +  768];
	 X4.y = lds[me + 1024];
	 X5.y = lds[me + 1280];
	 X6.y = lds[me + 1536];
	 X7.y = lds[me + 1792];
	 X8.y = lds[me + 2048];
	 X9.y = lds[me + 2304];
	X10.y = lds[me + 2560];
	X11.y = lds[me + 2816];
	X12.y = lds[me + 3072];
	X13.y = lds[me + 3328];
	X14.y = lds[me + 3584];
	X15.y = lds[me + 3840];

		
	barrier(CLK_LOCAL_MEM_FENCE);


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_FWD(twiddles_4096, 15 + 15*(me%16) + 14, X15)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_INV(twiddles_4096, 15 + 15*(me%16) + 14, X15)
	}
	
	if(dir == -1)
	{
		FwdRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}


	lds[(me/16)*256 + (me%16) +   0] =  X0.x;
	lds[(me/16)*256 + (me%16) +  16] =  X1.x;
	lds[(me/16)*256 + (me%16) +  32] =  X2.x;
	lds[(me/16)*256 + (me%16) +  48] =  X3.x;
	lds[(me/16)*256 + (me%16) +  64] =  X4.x;
	lds[(me/16)*256 + (me%16) +  80] =  X5.x;
	lds[(me/16)*256 + (me%16) +  96] =  X6.x;
	lds[(me/16)*256 + (me%16) + 112] =  X7.x;
	lds[(me/16)*256 + (me%16) + 128] =  X8.x;
	lds[(me/16)*256 + (me%16) + 144] =  X9.x;
	lds[(me/16)*256 + (me%16) + 160] = X10.x;
	lds[(me/16)*256 + (me%16) + 176] = X11.x;
	lds[(me/16)*256 + (me%16) + 192] = X12.x;
	lds[(me/16)*256 + (me%16) + 208] = X13.x;
	lds[(me/16)*256 + (me%16) + 224] = X14.x;
	lds[(me/16)*256 + (me%16) + 240] = X15.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	 X0.x = lds[me +    0];
	 X1.x = lds[me +  256];
	 X2.x = lds[me +  512];
	 X3.x = lds[me +  768];
	 X4.x = lds[me + 1024];
	 X5.x = lds[me + 1280];
	 X6.x = lds[me + 1536];
	 X7.x = lds[me + 1792];
	 X8.x = lds[me + 2048];
	 X9.x = lds[me + 2304];
	X10.x = lds[me + 2560];
	X11.x = lds[me + 2816];
	X12.x = lds[me + 3072];
	X13.x = lds[me + 3328];
	X14.x = lds[me + 3584];
	X15.x = lds[me + 3840];
	
		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[(me/16)*256 + (me%16) +   0] =  X0.y;
	lds[(me/16)*256 + (me%16) +  16] =  X1.y;
	lds[(me/16)*256 + (me%16) +  32] =  X2.y;
	lds[(me/16)*256 + (me%16) +  48] =  X3.y;
	lds[(me/16)*256 + (me%16) +  64] =  X4.y;
	lds[(me/16)*256 + (me%16) +  80] =  X5.y;
	lds[(me/16)*256 + (me%16) +  96] =  X6.y;
	lds[(me/16)*256 + (me%16) + 112] =  X7.y;
	lds[(me/16)*256 + (me%16) + 128] =  X8.y;
	lds[(me/16)*256 + (me%16) + 144] =  X9.y;
	lds[(me/16)*256 + (me%16) + 160] = X10.y;
	lds[(me/16)*256 + (me%16) + 176] = X11.y;
	lds[(me/16)*256 + (me%16) + 192] = X12.y;
	lds[(me/16)*256 + (me%16) + 208] = X13.y;
	lds[(me/16)*256 + (me%16) + 224] = X14.y;
	lds[(me/16)*256 + (me%16) + 240] = X15.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	 X0.y = lds[me +    0];
	 X1.y = lds[me +  256];
	 X2.y = lds[me +  512];
	 X3.y = lds[me +  768];
	 X4.y = lds[me + 1024];
	 X5.y = lds[me + 1280];
	 X6.y = lds[me + 1536];
	 X7.y = lds[me + 1792];
	 X8.y = lds[me + 2048];
	 X9.y = lds[me + 2304];
	X10.y = lds[me + 2560];
	X11.y = lds[me + 2816];
	X12.y = lds[me + 3072];
	X13.y = lds[me + 3328];
	X14.y = lds[me + 3584];
	X15.y = lds[me + 3840];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_FWD(twiddles_4096, 255 + 15*(me%256) + 14, X15)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_INV(twiddles_4096, 255 + 15*(me%256) + 14, X15)
	}
	
	if(dir == -1)
	{
		FwdRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}

	lwb[me +    0] =  X0;
	lwb[me +  256] =  X1;
	lwb[me +  512] =  X2;
	lwb[me +  768] =  X3;
	lwb[me + 1024] =  X4;
	lwb[me + 1280] =  X5;
	lwb[me + 1536] =  X6;
	lwb[me + 1792] =  X7;
	lwb[me + 2048] =  X8;
	lwb[me + 2304] =  X9;
	lwb[me + 2560] = X10;
	lwb[me + 2816] = X11;
	lwb[me + 3072] = X12;
	lwb[me + 3328] = X13;
	lwb[me + 3584] = X14;
	lwb[me + 3840] = X15;	

}


__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_262144_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float2 lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

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

	barrier(CLK_LOCAL_MEM_FENCE);


	for(uint t=0; t<2; t++)
	{

		b = (batch%256)*16 + t*8 + (me/16);

		fft_64_262144(b, me%16, lds + t*512 + (me/16)*64, dir);

		barrier(CLK_LOCAL_MEM_FENCE);

	}


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*4096 + t*32768] = R0;
	}
}

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_262144_2(__global const float2 * restrict gb, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);
	
	__local float lds[4096];
	uint ioOffset;
	__global float2 *lwb;

	ioOffset = (batch/64)*262144 + (batch%64)*4096;
	lwb = gb + ioOffset;
	
	fft_4096(me, lds, lwb, dir);

}


__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_262144( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 1;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*262144;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 4096;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*262144;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 64;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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


__attribute__((always_inline)) void
fft_1024(uint me, __local float *lds, __global float2 *lwbIn, __global float2 *lwbOut, const int dir)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	
	X0 = lwbIn[me +   0];
	X1 = lwbIn[me + 128];
	X2 = lwbIn[me + 256];
	X3 = lwbIn[me + 384];
	X4 = lwbIn[me + 512];
	X5 = lwbIn[me + 640];
	X6 = lwbIn[me + 768];
	X7 = lwbIn[me + 896];
					
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

	lds[me*8 + 0] = X0.x;
	lds[me*8 + 1] = X1.x;
	lds[me*8 + 2] = X2.x;
	lds[me*8 + 3] = X3.x;
	lds[me*8 + 4] = X4.x;
	lds[me*8 + 5] = X5.x;
	lds[me*8 + 6] = X6.x;
	lds[me*8 + 7] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me + 128];
	X2.x = lds[me + 256];
	X3.x = lds[me + 384];
	X4.x = lds[me + 512];
	X5.x = lds[me + 640];
	X6.x = lds[me + 768];
	X7.x = lds[me + 896];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me + 128];
	X2.y = lds[me + 256];
	X3.y = lds[me + 384];
	X4.y = lds[me + 512];
	X5.y = lds[me + 640];
	X6.y = lds[me + 768];
	X7.y = lds[me + 896];

		
	barrier(CLK_LOCAL_MEM_FENCE);


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles_1024, 7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(twiddles_1024, 7 + 7*(me%8) + 6, X7)
	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lds[(me/8)*64 + (me%8) +  0] = X0.x;
	lds[(me/8)*64 + (me%8) +  8] = X1.x;
	lds[(me/8)*64 + (me%8) + 16] = X2.x;
	lds[(me/8)*64 + (me%8) + 24] = X3.x;
	lds[(me/8)*64 + (me%8) + 32] = X4.x;
	lds[(me/8)*64 + (me%8) + 40] = X5.x;
	lds[(me/8)*64 + (me%8) + 48] = X6.x;
	lds[(me/8)*64 + (me%8) + 56] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[(2*me + 0) +   0];
	X1.x = lds[(2*me + 0) + 256];
	X2.x = lds[(2*me + 0) + 512];
	X3.x = lds[(2*me + 0) + 768];

	X4.x = lds[(2*me + 1) +   0];
	X5.x = lds[(2*me + 1) + 256];
	X6.x = lds[(2*me + 1) + 512];
	X7.x = lds[(2*me + 1) + 768];	

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[(2*me + 0) +   0];
	X1.y = lds[(2*me + 0) + 256];
	X2.y = lds[(2*me + 0) + 512];
	X3.y = lds[(2*me + 0) + 768];

	X4.y = lds[(2*me + 1) +   0];
	X5.y = lds[(2*me + 1) + 256];
	X6.y = lds[(2*me + 1) + 512];
	X7.y = lds[(2*me + 1) + 768];	

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_1024, 63 + 3*((2*me + 1)%64) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_1024, 63 + 3*((2*me + 1)%64) + 2, X7)	
	}	
	
	
	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}

		
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +   0] = X0.x;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +  64] = X1.x;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 128] = X2.x;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 192] = X3.x;
	
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +   0] = X4.x;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +  64] = X5.x;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 128] = X6.x;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 192] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[(2*me + 0) +   0];
	X1.x = lds[(2*me + 0) + 256];
	X2.x = lds[(2*me + 0) + 512];
	X3.x = lds[(2*me + 0) + 768];

	X4.x = lds[(2*me + 1) +   0];
	X5.x = lds[(2*me + 1) + 256];
	X6.x = lds[(2*me + 1) + 512];
	X7.x = lds[(2*me + 1) + 768];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +   0] = X0.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +  64] = X1.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 128] = X2.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 192] = X3.y;
	
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +   0] = X4.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +  64] = X5.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 128] = X6.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 192] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[(2*me + 0) +   0];
	X1.y = lds[(2*me + 0) + 256];
	X2.y = lds[(2*me + 0) + 512];
	X3.y = lds[(2*me + 0) + 768];

	X4.y = lds[(2*me + 1) +   0];
	X5.y = lds[(2*me + 1) + 256];
	X6.y = lds[(2*me + 1) + 512];
	X7.y = lds[(2*me + 1) + 768];	

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_1024, 255 + 3*((2*me + 1)%256) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_1024, 255 + 3*((2*me + 1)%256) + 2, X7)
	}	
	
	if(dir == -1)
	{
		FwdRad4(&X0, &X1, &X2, &X3);
		FwdRad4(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4(&X0, &X1, &X2, &X3);
		InvRad4(&X4, &X5, &X6, &X7);
	}	
		
	{
		__global float4 *lwbv = lwbOut;	
		lwbv[me +   0] = (float4)(X0,X4);
		lwbv[me + 128] = (float4)(X1,X5);	
		lwbv[me + 256] = (float4)(X2,X6);
		lwbv[me + 384] = (float4)(X3,X7);			
	}	
}


__attribute__((always_inline)) void
fft_512(uint me, __local float *lds, __global float2 *lwb, const int dir)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	X0 = lwb[me +   0];
	X1 = lwb[me +  64];
	X2 = lwb[me + 128];
	X3 = lwb[me + 192];
	X4 = lwb[me + 256];
	X5 = lwb[me + 320];
	X6 = lwb[me + 384];
	X7 = lwb[me + 448];
					
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

	lds[me*8 + 0] = X0.x;
	lds[me*8 + 1] = X1.x;
	lds[me*8 + 2] = X2.x;
	lds[me*8 + 3] = X3.x;
	lds[me*8 + 4] = X4.x;
	lds[me*8 + 5] = X5.x;
	lds[me*8 + 6] = X6.x;
	lds[me*8 + 7] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	X4.x = lds[me + 256];
	X5.x = lds[me + 320];
	X6.x = lds[me + 384];
	X7.x = lds[me + 448];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	X4.y = lds[me + 256];
	X5.y = lds[me + 320];
	X6.y = lds[me + 384];
	X7.y = lds[me + 448];

		
	barrier(CLK_LOCAL_MEM_FENCE);


			
	if(dir == -1)
	{
		float2 W;
		float TR, TI;
		
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles_512, 7 + 7*(me%8) + 6, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(twiddles_512, 7 + 7*(me%8) + 6, X7)
	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lds[(me/8)*64 + (me%8) +  0] = X0.x;
	lds[(me/8)*64 + (me%8) +  8] = X1.x;
	lds[(me/8)*64 + (me%8) + 16] = X2.x;
	lds[(me/8)*64 + (me%8) + 24] = X3.x;
	lds[(me/8)*64 + (me%8) + 32] = X4.x;
	lds[(me/8)*64 + (me%8) + 40] = X5.x;
	lds[(me/8)*64 + (me%8) + 48] = X6.x;
	lds[(me/8)*64 + (me%8) + 56] = X7.x;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	X4.x = lds[me + 256];
	X5.x = lds[me + 320];
	X6.x = lds[me + 384];
	X7.x = lds[me + 448];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	barrier(CLK_LOCAL_MEM_FENCE);
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	X4.y = lds[me + 256];
	X5.y = lds[me + 320];
	X6.y = lds[me + 384];
	X7.y = lds[me + 448];

		
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles_512, 63 + 7*me + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 0, X1)			
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 1, X2)	
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 2, X3)	
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 3, X4)	
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 4, X5)	
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 5, X6)	
		TWIDDLE_MUL_INV(twiddles_512, 63 + 7*me + 6, X7)
	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	lwb[me +   0] = X0;
	lwb[me +  64] = X1;
	lwb[me + 128] = X2;
	lwb[me + 192] = X3;
	lwb[me + 256] = X4;
	lwb[me + 320] = X5;
	lwb[me + 384] = X6;
	lwb[me + 448] = X7;		
		
}



__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_524288_1( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 512;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*557056;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_524288_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

	iOffset = (batch/512)*557056 + (batch%512)*1088;
	oOffset = (batch/512)*524288 + (batch%512)*1024;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_1024(me, lds, lwbIn, lwbOut, dir);
}

__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_524288_2( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count, const int dir )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 8;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
			TWIDDLE_3STEP_MUL_FWD(TW3step_524288, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
		 }
		 else
		 {
			TWIDDLE_3STEP_MUL_INV(TW3step_524288, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)
		 }
		 
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*589824;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 576;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_524288_2(__global const float2 * restrict gb, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];

	uint ioOffset;
	__global float2 *lwb;

	ioOffset = (batch/1024)*589824 + (batch%1024)*576;
	lwb = gb + ioOffset;

	fft_512(me, lds, lwb, dir);

}


__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_524288_3( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 8;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*589824;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 576;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
   iOffset += currDimIndex * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*524288;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
   oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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


__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_1048576_1( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_1048576_1(__global const float2 * restrict gbIn, __global float2 * restrict gbOut, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[1024];

	uint iOffset;
	uint oOffset;
	__global float2 *lwbIn;
	__global float2 *lwbOut;

	iOffset = (batch/1024)*1114112 + (batch%1024)*1088;
	oOffset = (batch/1024)*1048576 + (batch%1024)*1024;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	fft_1024(me, lds, lwbIn, lwbOut, dir);
}

__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_1048576_2( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count, const int dir )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * currDimIndex;
   iOffset += groupIndex.x * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
			TWIDDLE_3STEP_MUL_FWD(TW3step_1048576, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)		 
		 }
		 else
		 {
			TWIDDLE_3STEP_MUL_INV(TW3step_1048576, (groupIndex.x * wgTileExtent.x + xInd) * (currDimIndex * wgTileExtent.y * wgUnroll + yInd), tmp)		 
		 }
		 
         lds[ xInd ][ yInd ] = tmp; 
      }
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   oOffset += rowSizeinUnits * wgTileExtent.x * groupIndex.x;
   oOffset += currDimIndex * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_1048576_2(__global const float2 * restrict gb, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[1024];

	uint ioOffset;
	__global float2 *lwb;

	ioOffset = (batch/1024)*1114112 + (batch%1024)*1088;
	lwb = gb + ioOffset;
	
	fft_1024(me, lds, lwb, lwb, dir);

}

__attribute__(( reqd_work_group_size( 16, 16, 1 ) ))
kernel void
transpose_1048576_3( global float2* restrict pmComplexIn, global float2* restrict pmComplexOut, const uint count )
{
   const Tile localIndex = { get_local_id( 0 ), get_local_id( 1 ) }; 
   const Tile localExtent = { get_local_size( 0 ), get_local_size( 1 ) }; 
   const Tile groupIndex = { get_group_id( 0 ), get_group_id( 1 ) };
   
   // Calculate the unit address (in terms of datatype) of the beginning of the Tile for the WG block
   // Transpose of input & output blocks happens with the Offset calculation
   const size_t reShapeFactor = 4;
   const size_t wgUnroll = 16;
   const Tile wgTileExtent = { localExtent.x * reShapeFactor, localExtent.y / reShapeFactor };
   const size_t numGroupsY_1 = 16;
   // LDS is always complex and allocated transposed: lds[ wgTileExtent.y * wgUnroll ][ wgTileExtent.x ];
   local float2 lds[ 64 ][ 64 ];

   size_t currDimIndex;
   size_t rowSizeinUnits;

   size_t iOffset = 0;
   currDimIndex = groupIndex.y;
   iOffset += (currDimIndex/numGroupsY_1)*1114112;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1088;
   iOffset += rowSizeinUnits * wgTileExtent.y * wgUnroll * groupIndex.x;
   iOffset += currDimIndex * wgTileExtent.x;
   
   global float2* tileIn = pmComplexIn + iOffset;
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
   
   barrier( CLK_LOCAL_MEM_FENCE );
   
   size_t oOffset = 0;
   currDimIndex = groupIndex.y;
   oOffset += (currDimIndex/numGroupsY_1)*1048576;
   currDimIndex = currDimIndex % numGroupsY_1;
   rowSizeinUnits = 1024;
   oOffset += rowSizeinUnits * wgTileExtent.x * currDimIndex;
   oOffset += groupIndex.x * wgTileExtent.y * wgUnroll;
   
   global float2* tileOut = pmComplexOut + oOffset;

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


