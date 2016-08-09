
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

