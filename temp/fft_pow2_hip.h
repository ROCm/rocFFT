
#include "butterfly.h"


__device__
void fft_1(float2 *lwb, const uint rw)
{
	float2 X0;

	if(rw)
	{
		X0 = lwb[0];
		lwb[0] = X0;
	}
}

template <StrideBin sb>
__device__
void fft_2(float2 *twiddles, float2 *lwb, const uint rw, const ulong stride)
{
	float2 X0, X1;

	if(rw)
	{
		if(sb == SB_UNIT)
		{
		X0 = lwb[0];
		X1 = lwb[1];
		}
		else
		{
		X0 = lwb[0*stride];
		X1 = lwb[1*stride];			
		}
	}
	
	Rad2(&X0, &X1);
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{
		lwb[0] = X0;
		lwb[1] = X1;
		}
		else
		{
		lwb[0*stride] = X0;
		lwb[1*stride] = X1;			
		}	
	}
}	



template <StrideBin sb, int dir>
__device__
void fft_4(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		X0 = lwb[me + 0];
		X1 = lwb[me + 2];	
		}
		else
		{
		X0 = lwb[(me + 0)*stride];
		X1 = lwb[(me + 2)*stride];			
		}
	}
	
	Rad2(&X0, &X1);

	lds[me*2 + 0] = X0.x;
	lds[me*2 + 1] = X1.x;
	
	__syncthreads();
	
	X0.x = lds[me + 0];
	X1.x = lds[me + 2];
	
	__syncthreads();
	
	lds[me*2 + 0] = X0.y;
	lds[me*2 + 1] = X1.y;
	
	__syncthreads();
	
	X0.y = lds[me + 0];
	X1.y = lds[me + 2];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 1 + (me%2), X1)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 1 + (me%2), X1)
	}
	
	Rad2(&X0, &X1);
	
	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		lwb[me + 0] = X0;
		lwb[me + 2] = X1;	
		}
		else
		{
		lwb[(me + 0)*stride] = X0;
		lwb[(me + 2)*stride] = X1;			
		}	
	}
}



template <StrideBin sb, int dir>
__device__
void fft_8(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1, X2, X3;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		X0 = lwb[me + 0];
		X1 = lwb[me + 2];	
		X2 = lwb[me + 4];
		X3 = lwb[me + 6];	
		}
		else		
		{
		X0 = lwb[(me + 0)*stride];
		X1 = lwb[(me + 2)*stride];	
		X2 = lwb[(me + 4)*stride];
		X3 = lwb[(me + 6)*stride];			
		}
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0.x;
	lds[me*4 + 1] = X1.x;
	lds[me*4 + 2] = X2.x;
	lds[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me*2 + 0 + 0];
	X1.x = lds[me*2 + 0 + 4];	
	X2.x = lds[me*2 + 1 + 0];
	X3.x = lds[me*2 + 1 + 4];
	
	__syncthreads();
	
	lds[me*4 + 0] = X0.y;
	lds[me*4 + 1] = X1.y;
	lds[me*4 + 2] = X2.y;
	lds[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me*2 + 0 + 0];
	X1.y = lds[me*2 + 0 + 4];	
	X2.y = lds[me*2 + 1 + 0];
	X3.y = lds[me*2 + 1 + 4];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_FWD(twiddles, 3 + ((2*me+1)%4), X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_INV(twiddles, 3 + ((2*me+1)%4), X3)			
	}
	
	Rad2(&X0, &X1);
	Rad2(&X2, &X3);
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{		
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me + 0] = float4(X0.x,X0.y,X2.x,X2.y);
		lwbv[me + 2] = float4(X1.x,X1.y,X3.x,X3.y);
		}		
		else		
		{
		lwb[(me + 0)*stride] = X0;
		lwb[(me + 2)*stride] = X1;	
		lwb[(me + 4)*stride] = X2;
		lwb[(me + 6)*stride] = X3;			
		}		
	}
}



template <StrideBin sb, int dir>
__device__
void fft_16(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1, X2, X3;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		X0 = lwb[me + 0];
		X1 = lwb[me + 4];	
		X2 = lwb[me + 8];
		X3 = lwb[me + 12];	
		}
		else
		{
		X0 = lwb[(me +  0)*stride];
		X1 = lwb[(me +  4)*stride];	
		X2 = lwb[(me +  8)*stride];
		X3 = lwb[(me + 12)*stride];				
		}
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0.x;
	lds[me*4 + 1] = X1.x;
	lds[me*4 + 2] = X2.x;
	lds[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me + 0];
	X1.x = lds[me + 4];	
	X2.x = lds[me + 8];
	X3.x = lds[me + 12];
	
	__syncthreads();
	
	lds[me*4 + 0] = X0.y;
	lds[me*4 + 1] = X1.y;
	lds[me*4 + 2] = X2.y;
	lds[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me + 0];
	X1.y = lds[me + 4];	
	X2.y = lds[me + 8];
	X3.y = lds[me + 12];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 2, X3)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 2, X3)	
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	if(rw)
	{	
		if(sb == SB_UNIT)
		{		
		lwb[me + 0]  = X0;
		lwb[me + 4]  = X1;	
		lwb[me + 8]  = X2;
		lwb[me + 12] = X3;		
		}
		else
		{
		lwb[(me +  0)*stride]  = X0;
		lwb[(me +  4)*stride]  = X1;	
		lwb[(me +  8)*stride]  = X2;
		lwb[(me + 12)*stride] = X3;			
		}
	}
}



template <StrideBin sb, int dir>
__device__
void fft_32(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		X0 = lwb[me +  0];
		X1 = lwb[me +  4];
		X2 = lwb[me +  8];
		X3 = lwb[me + 12];
		X4 = lwb[me + 16];
		X5 = lwb[me + 20];
		X6 = lwb[me + 24];
		X7 = lwb[me + 28];
		}
		else
		{
		X0 = lwb[(me +  0)*stride];
		X1 = lwb[(me +  4)*stride];
		X2 = lwb[(me +  8)*stride];
		X3 = lwb[(me + 12)*stride];
		X4 = lwb[(me + 16)*stride];
		X5 = lwb[(me + 20)*stride];
		X6 = lwb[(me + 24)*stride];
		X7 = lwb[(me + 28)*stride];			
		}
	}				
	
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
				
	__syncthreads();

	
	X0.x = lds[(2*me + 0) +  0];
	X1.x = lds[(2*me + 0) +  8];
	X2.x = lds[(2*me + 0) + 16];
	X3.x = lds[(2*me + 0) + 24];

	X4.x = lds[(2*me + 1) +  0];
	X5.x = lds[(2*me + 1) +  8];
	X6.x = lds[(2*me + 1) + 16];
	X7.x = lds[(2*me + 1) + 24];	

	__syncthreads();
				

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
				
	__syncthreads();

	
	X0.y = lds[(2*me + 0) +  0];
	X1.y = lds[(2*me + 0) +  8];
	X2.y = lds[(2*me + 0) + 16];
	X3.y = lds[(2*me + 0) + 24];

	X4.y = lds[(2*me + 1) +  0];
	X5.y = lds[(2*me + 1) +  8];
	X6.y = lds[(2*me + 1) + 16];
	X7.y = lds[(2*me + 1) + 24];	

	__syncthreads();
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 2, X7)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 2, X7)	
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
		
		
	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +  0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me +  4] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me +  8] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 12] = float4(X3.x,X3.y,X7.x,X7.y);			
		}
		else
		{
		lwb[(me +  0)*stride] = X0;
		lwb[(me +  4)*stride] = X1;
		lwb[(me +  8)*stride] = X2;
		lwb[(me + 12)*stride] = X3;
		lwb[(me + 16)*stride] = X4;
		lwb[(me + 20)*stride] = X5;
		lwb[(me + 24)*stride] = X6;
		lwb[(me + 28)*stride] = X7;				
		}
	}			
	
}


template <StrideBin sb, int dir>
__device__
void fft_64(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1, X2, X3;

	if(rw)
	{
		if(sb == SB_UNIT)
		{
		X0 = lwb[me + 0];
		X1 = lwb[me + 16];	
		X2 = lwb[me + 32];
		X3 = lwb[me + 48];	
		}
		else
		{
			
		}
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lds[me*4 + 0] = X0.x;
	lds[me*4 + 1] = X1.x;
	lds[me*4 + 2] = X2.x;
	lds[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me + 0];
	X1.x = lds[me + 16];	
	X2.x = lds[me + 32];
	X3.x = lds[me + 48];
	
	__syncthreads();
	
	lds[me*4 + 0] = X0.y;
	lds[me*4 + 1] = X1.y;
	lds[me*4 + 2] = X2.y;
	lds[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me + 0];
	X1.y = lds[me + 16];	
	X2.y = lds[me + 32];
	X3.y = lds[me + 48];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	lds[(me/4)*16 + me%4 +  0] = X0.x;
	lds[(me/4)*16 + me%4 +  4] = X1.x;
	lds[(me/4)*16 + me%4 +  8] = X2.x;
	lds[(me/4)*16 + me%4 + 12] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me + 0];
	X1.x = lds[me + 16];	
	X2.x = lds[me + 32];
	X3.x = lds[me + 48];
	
	__syncthreads();
	
	lds[(me/4)*16 + me%4 +  0] = X0.y;
	lds[(me/4)*16 + me%4 +  4] = X1.y;
	lds[(me/4)*16 + me%4 +  8] = X2.y;
	lds[(me/4)*16 + me%4 + 12] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me + 0];
	X1.y = lds[me + 16];	
	X2.y = lds[me + 32];
	X3.y = lds[me + 48];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);

		
	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		lwb[me + 0]  = X0;
		lwb[me + 16] = X1;	
		lwb[me + 32] = X2;
		lwb[me + 48] = X3;		
		}
		else
		{
		lwb[(me +  0)*stride]  = X0;
		lwb[(me + 16)*stride] = X1;	
		lwb[(me + 32)*stride] = X2;
		lwb[(me + 48)*stride] = X3;			
		}
	}
}


template <StrideBin sb, int dir>
__device__
void fft_128(float2 *twiddles, float2 *lwb, float *lds, const uint me, const uint rw, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		X0 = lwb[me + 0];
		X1 = lwb[me + 16];
		X2 = lwb[me + 32];
		X3 = lwb[me + 48];
		X4 = lwb[me + 64];
		X5 = lwb[me + 80];
		X6 = lwb[me + 96];
		X7 = lwb[me + 112];
		}
		else
		{
		X0 = lwb[(me +   0)*stride];
		X1 = lwb[(me +  16)*stride];
		X2 = lwb[(me +  32)*stride];
		X3 = lwb[(me +  48)*stride];
		X4 = lwb[(me +  64)*stride];
		X5 = lwb[(me +  80)*stride];
		X6 = lwb[(me +  96)*stride];
		X7 = lwb[(me + 112)*stride];			
		}
	}
	
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
	
	__syncthreads();
	
	X0.x = lds[(2*me + 0) +  0];
	X1.x = lds[(2*me + 0) + 32];
	X2.x = lds[(2*me + 0) + 64];
	X3.x = lds[(2*me + 0) + 96];

	X4.x = lds[(2*me + 1) +  0];
	X5.x = lds[(2*me + 1) + 32];
	X6.x = lds[(2*me + 1) + 64];
	X7.x = lds[(2*me + 1) + 96];	

	__syncthreads();

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	
	__syncthreads();
	
	X0.y = lds[(2*me + 0) +  0];
	X1.y = lds[(2*me + 0) + 32];
	X2.y = lds[(2*me + 0) + 64];
	X3.y = lds[(2*me + 0) + 96];

	X4.y = lds[(2*me + 1) +  0];
	X5.y = lds[(2*me + 1) + 32];
	X6.y = lds[(2*me + 1) + 64];
	X7.y = lds[(2*me + 1) + 96];	

	__syncthreads();	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 7 + 3*((2*me + 1)%8) + 2, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 7 + 3*((2*me + 1)%8) + 2, X7)		
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
	
	
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.x;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.x;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.x;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.x;
	
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.x;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.x;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.x;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.x;
	
	__syncthreads();
	
	X0.x = lds[(2*me + 0) +  0];
	X1.x = lds[(2*me + 0) + 32];
	X2.x = lds[(2*me + 0) + 64];
	X3.x = lds[(2*me + 0) + 96];

	X4.x = lds[(2*me + 1) +  0];
	X5.x = lds[(2*me + 1) + 32];
	X6.x = lds[(2*me + 1) + 64];
	X7.x = lds[(2*me + 1) + 96];	

	__syncthreads();

	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.y;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.y;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.y;
	lds[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.y;
	
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.y;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.y;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.y;
	lds[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.y;
	
	__syncthreads();
	
	X0.y = lds[(2*me + 0) +  0];
	X1.y = lds[(2*me + 0) + 32];
	X2.y = lds[(2*me + 0) + 64];
	X3.y = lds[(2*me + 0) + 96];

	X4.y = lds[(2*me + 1) +  0];
	X5.y = lds[(2*me + 1) + 32];
	X6.y = lds[(2*me + 1) + 64];
	X7.y = lds[(2*me + 1) + 96];	

	__syncthreads();	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 31 + 3*((2*me + 1)%32) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 31 + 3*((2*me + 1)%32) + 2, X7)
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
		
		
	if(rw)
	{
		if(sb == SB_UNIT)
		{
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +  0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 16] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 32] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 48] = float4(X3.x,X3.y,X7.x,X7.y);	
		}
		else
		{
		lwb[(me +   0)*stride] = X0;
		lwb[(me +  16)*stride] = X1;
		lwb[(me +  32)*stride] = X2;
		lwb[(me +  48)*stride] = X3;
		lwb[(me +  64)*stride] = X4;
		lwb[(me +  80)*stride] = X5;
		lwb[(me +  96)*stride] = X6;
		lwb[(me + 112)*stride] = X7;			
		}
	}	
	
}


template <StrideBin sb, int dir>
__device__
void fft_256(float2 *twiddles, float2 *lwb, float *lds, const uint me, const ulong stride)
{
	float2 X0, X1, X2, X3;

	if(sb == SB_UNIT)
	{
	X0 = lwb[me +   0];
	X1 = lwb[me +  64];	
	X2 = lwb[me + 128];
	X3 = lwb[me + 192];	
	}
	else	
	{
	X0 = lwb[(me +   0)*stride];
	X1 = lwb[(me +  64)*stride];	
	X2 = lwb[(me + 128)*stride];
	X3 = lwb[(me + 192)*stride];			
	}
	

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	lds[me*4 + 0] = X0.x;
	lds[me*4 + 1] = X1.x;
	lds[me*4 + 2] = X2.x;
	lds[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	__syncthreads();
	
	lds[me*4 + 0] = X0.y;
	lds[me*4 + 1] = X1.y;
	lds[me*4 + 2] = X2.y;
	lds[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
	__syncthreads();	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 3 + 3*(me%4) + 2, X3)		
	}

	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		
		
	lds[(me/4)*16 + me%4 +  0] = X0.x;
	lds[(me/4)*16 + me%4 +  4] = X1.x;
	lds[(me/4)*16 + me%4 +  8] = X2.x;
	lds[(me/4)*16 + me%4 + 12] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	__syncthreads();
	
	lds[(me/4)*16 + me%4 +  0] = X0.y;
	lds[(me/4)*16 + me%4 +  4] = X1.y;
	lds[(me/4)*16 + me%4 +  8] = X2.y;
	lds[(me/4)*16 + me%4 + 12] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
	__syncthreads();	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 15 + 3*(me%16) + 2, X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 15 + 3*(me%16) + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);


	lds[(me/16)*64 + me%16 +  0] = X0.x;
	lds[(me/16)*64 + me%16 + 16] = X1.x;
	lds[(me/16)*64 + me%16 + 32] = X2.x;
	lds[(me/16)*64 + me%16 + 48] = X3.x;
	
	__syncthreads();
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	__syncthreads();
	
	lds[(me/16)*64 + me%16 +  0] = X0.y;
	lds[(me/16)*64 + me%16 + 16] = X1.y;
	lds[(me/16)*64 + me%16 + 32] = X2.y;
	lds[(me/16)*64 + me%16 + 48] = X3.y;
	
	__syncthreads();
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
	__syncthreads();		
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*me + 2, X3)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 63 + 3*me + 0, X1)	
		TWIDDLE_MUL_INV(twiddles, 63 + 3*me + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 63 + 3*me + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		
	if(sb == SB_UNIT)
	{
	lwb[me +   0] = X0;
	lwb[me +  64] = X1;	
	lwb[me + 128] = X2;
	lwb[me + 192] = X3;		
	}
	else
	{
	lwb[(me +   0)*stride] = X0;
	lwb[(me +  64)*stride] = X1;	
	lwb[(me + 128)*stride] = X2;
	lwb[(me + 192)*stride] = X3;		
	}

}


template <StrideBin sb, int dir>
__device__
void fft_512(float2 *twiddles, float2 *lwb, float *lds, const uint me, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	X0 = lwb[me +   0];
	X1 = lwb[me +  64];
	X2 = lwb[me + 128];
	X3 = lwb[me + 192];
	X4 = lwb[me + 256];
	X5 = lwb[me + 320];
	X6 = lwb[me + 384];
	X7 = lwb[me + 448];
	}
	else
	{
	X0 = lwb[(me +   0)*stride];
	X1 = lwb[(me +  64)*stride];
	X2 = lwb[(me + 128)*stride];
	X3 = lwb[(me + 192)*stride];
	X4 = lwb[(me + 256)*stride];
	X5 = lwb[(me + 320)*stride];
	X6 = lwb[(me + 384)*stride];
	X7 = lwb[(me + 448)*stride];		
	}
					
	
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
	

	__syncthreads();
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	X4.x = lds[me + 256];
	X5.x = lds[me + 320];
	X6.x = lds[me + 384];
	X7.x = lds[me + 448];

		
	__syncthreads();
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	X4.y = lds[me + 256];
	X5.y = lds[me + 320];
	X6.y = lds[me + 384];
	X7.y = lds[me + 448];

		
	__syncthreads();


			
	if(dir == -1)
	{
		float2 W;
		float TR, TI;
		
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 6, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 6, X7)
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
	

	__syncthreads();
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	X4.x = lds[me + 256];
	X5.x = lds[me + 320];
	X6.x = lds[me + 384];
	X7.x = lds[me + 448];

		
	__syncthreads();
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	X4.y = lds[me + 256];
	X5.y = lds[me + 320];
	X6.y = lds[me + 384];
	X7.y = lds[me + 448];

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*me + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 0, X1)			
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 2, X3)	
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 3, X4)	
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 4, X5)	
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 5, X6)	
		TWIDDLE_MUL_INV(twiddles, 63 + 7*me + 6, X7)
	}
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

	if(sb == SB_UNIT)
	{
	lwb[me +   0] = X0;
	lwb[me +  64] = X1;
	lwb[me + 128] = X2;
	lwb[me + 192] = X3;
	lwb[me + 256] = X4;
	lwb[me + 320] = X5;
	lwb[me + 384] = X6;
	lwb[me + 448] = X7;		
	}
	else
	{
	lwb[(me +   0)*stride] = X0;
	lwb[(me +  64)*stride] = X1;
	lwb[(me + 128)*stride] = X2;
	lwb[(me + 192)*stride] = X3;
	lwb[(me + 256)*stride] = X4;
	lwb[(me + 320)*stride] = X5;
	lwb[(me + 384)*stride] = X6;
	lwb[(me + 448)*stride] = X7;			
	}
		
}
	

template <StrideBin sb, int dir>
__device__
void fft_1024(float2 *twiddles, float2 *lwb, float *lds, const uint me, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	X0 = lwb[me +   0];
	X1 = lwb[me + 128];
	X2 = lwb[me + 256];
	X3 = lwb[me + 384];
	X4 = lwb[me + 512];
	X5 = lwb[me + 640];
	X6 = lwb[me + 768];
	X7 = lwb[me + 896];
	}
	else
	{
	X0 = lwb[(me +   0)*stride];
	X1 = lwb[(me + 128)*stride];
	X2 = lwb[(me + 256)*stride];
	X3 = lwb[(me + 384)*stride];
	X4 = lwb[(me + 512)*stride];
	X5 = lwb[(me + 640)*stride];
	X6 = lwb[(me + 768)*stride];
	X7 = lwb[(me + 896)*stride];		
	}
					
	
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
	

	__syncthreads();
			
		
	X0.x = lds[me +   0];
	X1.x = lds[me + 128];
	X2.x = lds[me + 256];
	X3.x = lds[me + 384];
	X4.x = lds[me + 512];
	X5.x = lds[me + 640];
	X6.x = lds[me + 768];
	X7.x = lds[me + 896];

		
	__syncthreads();
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[me +   0];
	X1.y = lds[me + 128];
	X2.y = lds[me + 256];
	X3.y = lds[me + 384];
	X4.y = lds[me + 512];
	X5.y = lds[me + 640];
	X6.y = lds[me + 768];
	X7.y = lds[me + 896];

		
	__syncthreads();


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 6, X7)
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
	

	__syncthreads();
			
		
	X0.x = lds[(2*me + 0) +   0];
	X1.x = lds[(2*me + 0) + 256];
	X2.x = lds[(2*me + 0) + 512];
	X3.x = lds[(2*me + 0) + 768];

	X4.x = lds[(2*me + 1) +   0];
	X5.x = lds[(2*me + 1) + 256];
	X6.x = lds[(2*me + 1) + 512];
	X7.x = lds[(2*me + 1) + 768];	

		
	__syncthreads();
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[(2*me + 0) +   0];
	X1.y = lds[(2*me + 0) + 256];
	X2.y = lds[(2*me + 0) + 512];
	X3.y = lds[(2*me + 0) + 768];

	X4.y = lds[(2*me + 1) +   0];
	X5.y = lds[(2*me + 1) + 256];
	X6.y = lds[(2*me + 1) + 512];
	X7.y = lds[(2*me + 1) + 768];	

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 63 + 3*((2*me + 1)%64) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 63 + 3*((2*me + 1)%64) + 2, X7)	
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
	

	__syncthreads();
			
		
	X0.x = lds[(2*me + 0) +   0];
	X1.x = lds[(2*me + 0) + 256];
	X2.x = lds[(2*me + 0) + 512];
	X3.x = lds[(2*me + 0) + 768];

	X4.x = lds[(2*me + 1) +   0];
	X5.x = lds[(2*me + 1) + 256];
	X6.x = lds[(2*me + 1) + 512];
	X7.x = lds[(2*me + 1) + 768];

		
	__syncthreads();
	

	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +   0] = X0.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 +  64] = X1.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 128] = X2.y;
	lds[((2*me + 0)/64)*256 + (2*me + 0)%64 + 192] = X3.y;
	
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +   0] = X4.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 +  64] = X5.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 128] = X6.y;
	lds[((2*me + 1)/64)*256 + (2*me + 1)%64 + 192] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[(2*me + 0) +   0];
	X1.y = lds[(2*me + 0) + 256];
	X2.y = lds[(2*me + 0) + 512];
	X3.y = lds[(2*me + 0) + 768];

	X4.y = lds[(2*me + 1) +   0];
	X5.y = lds[(2*me + 1) + 256];
	X6.y = lds[(2*me + 1) + 512];
	X7.y = lds[(2*me + 1) + 768];	

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 255 + 3*((2*me + 1)%256) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 255 + 3*((2*me + 1)%256) + 2, X7)
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
	
	if(sb == SB_UNIT)	
	{
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +   0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 128] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 256] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 384] = float4(X3.x,X3.y,X7.x,X7.y);			
	}	
	else
	{
		lwb[(me +   0)*stride] = X0;
		lwb[(me + 128)*stride] = X1;
		lwb[(me + 256)*stride] = X2;
		lwb[(me + 384)*stride] = X3;
		lwb[(me + 512)*stride] = X4;
		lwb[(me + 640)*stride] = X5;
		lwb[(me + 768)*stride] = X6;
		lwb[(me + 896)*stride] = X7;		
	}
}


	
template <StrideBin sb, int dir>
__device__
void fft_2048(float2 *twiddles, float2 *lwb, float *lds, const uint me, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	X0 = lwb[me +    0];
	X1 = lwb[me +  256];
	X2 = lwb[me +  512];
	X3 = lwb[me +  768];
	X4 = lwb[me + 1024];
	X5 = lwb[me + 1280];
	X6 = lwb[me + 1536];
	X7 = lwb[me + 1792];
	}
	else
	{
	X0 = lwb[(me +    0)*stride];
	X1 = lwb[(me +  256)*stride];
	X2 = lwb[(me +  512)*stride];
	X3 = lwb[(me +  768)*stride];
	X4 = lwb[(me + 1024)*stride];
	X5 = lwb[(me + 1280)*stride];
	X6 = lwb[(me + 1536)*stride];
	X7 = lwb[(me + 1792)*stride];		
	}
					
	
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
	

	__syncthreads();
			
		
	X0.x = lds[me +    0];
	X1.x = lds[me +  256];
	X2.x = lds[me +  512];
	X3.x = lds[me +  768];
	X4.x = lds[me + 1024];
	X5.x = lds[me + 1280];
	X6.x = lds[me + 1536];
	X7.x = lds[me + 1792];

		
	__syncthreads();
	

	lds[me*8 + 0] = X0.y;
	lds[me*8 + 1] = X1.y;
	lds[me*8 + 2] = X2.y;
	lds[me*8 + 3] = X3.y;
	lds[me*8 + 4] = X4.y;
	lds[me*8 + 5] = X5.y;
	lds[me*8 + 6] = X6.y;
	lds[me*8 + 7] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[me +    0];
	X1.y = lds[me +  256];
	X2.y = lds[me +  512];
	X3.y = lds[me +  768];
	X4.y = lds[me + 1024];
	X5.y = lds[me + 1280];
	X6.y = lds[me + 1536];
	X7.y = lds[me + 1792];

		
	__syncthreads();


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_FWD(twiddles, 7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_INV(twiddles, 7 + 7*(me%8) + 6, X7)				

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
	

	__syncthreads();
			
		
	X0.x = lds[me +    0];
	X1.x = lds[me +  256];
	X2.x = lds[me +  512];
	X3.x = lds[me +  768];
	X4.x = lds[me + 1024];
	X5.x = lds[me + 1280];
	X6.x = lds[me + 1536];
	X7.x = lds[me + 1792];

		
	__syncthreads();
	

	lds[(me/8)*64 + (me%8) +  0] = X0.y;
	lds[(me/8)*64 + (me%8) +  8] = X1.y;
	lds[(me/8)*64 + (me%8) + 16] = X2.y;
	lds[(me/8)*64 + (me%8) + 24] = X3.y;
	lds[(me/8)*64 + (me%8) + 32] = X4.y;
	lds[(me/8)*64 + (me%8) + 40] = X5.y;
	lds[(me/8)*64 + (me%8) + 48] = X6.y;
	lds[(me/8)*64 + (me%8) + 56] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[me +    0];
	X1.y = lds[me +  256];
	X2.y = lds[me +  512];
	X3.y = lds[me +  768];
	X4.y = lds[me + 1024];
	X5.y = lds[me + 1280];
	X6.y = lds[me + 1536];
	X7.y = lds[me + 1792];

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_FWD(twiddles, 63 + 7*(me%64) + 6, X7)			

	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_INV(twiddles, 63 + 7*(me%64) + 6, X7)
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
	

	__syncthreads();
			
		
	X0.x = lds[(2*me + 0) +    0];
	X1.x = lds[(2*me + 0) +  512];
	X2.x = lds[(2*me + 0) + 1024];
	X3.x = lds[(2*me + 0) + 1536];

	X4.x = lds[(2*me + 1) +    0];
	X5.x = lds[(2*me + 1) +  512];
	X6.x = lds[(2*me + 1) + 1024];
	X7.x = lds[(2*me + 1) + 1536];

		
	__syncthreads();
	

	lds[(me/64)*512 + (me%64) +   0] = X0.y;
	lds[(me/64)*512 + (me%64) +  64] = X1.y;
	lds[(me/64)*512 + (me%64) + 128] = X2.y;
	lds[(me/64)*512 + (me%64) + 192] = X3.y;
	lds[(me/64)*512 + (me%64) + 256] = X4.y;
	lds[(me/64)*512 + (me%64) + 320] = X5.y;
	lds[(me/64)*512 + (me%64) + 384] = X6.y;
	lds[(me/64)*512 + (me%64) + 448] = X7.y;
	

	__syncthreads();
			
		
	X0.y = lds[(2*me + 0) +    0];
	X1.y = lds[(2*me + 0) +  512]; 
	X2.y = lds[(2*me + 0) + 1024];
	X3.y = lds[(2*me + 0) + 1536];

	X4.y = lds[(2*me + 1) +    0];
	X5.y = lds[(2*me + 1) +  512];
	X6.y = lds[(2*me + 1) + 1024];
	X7.y = lds[(2*me + 1) + 1536];	

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles, 511 + 3*((2*me + 1)%512) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_INV(twiddles, 511 + 3*((2*me + 1)%512) + 2, X7)	
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
		
	if(sb == SB_UNIT)		
	{
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +   0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 256] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 512] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 768] = float4(X3.x,X3.y,X7.x,X7.y);			
	}
	else
	{
		lwb[(me +    0)*stride] = X0;
		lwb[(me +  256)*stride] = X1;
		lwb[(me +  512)*stride] = X2;
		lwb[(me +  768)*stride] = X3;
		lwb[(me + 1024)*stride] = X4;
		lwb[(me + 1280)*stride] = X5;
		lwb[(me + 1536)*stride] = X6;
		lwb[(me + 1792)*stride] = X7;		
	}
}

template <StrideBin sb, int dir>
__device__
void fft_4096(float2 *twiddles, float2 *lwb, float *lds, const uint me, const ulong stride)
{
	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	float2 X8, X9, X10, X11, X12, X13, X14, X15;
	
	if(sb == SB_UNIT)
	{
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
	}
	else
	{
	 X0 = lwb[(me +    0)*stride];
	 X1 = lwb[(me +  256)*stride];
	 X2 = lwb[(me +  512)*stride];
	 X3 = lwb[(me +  768)*stride];
	 X4 = lwb[(me + 1024)*stride];
	 X5 = lwb[(me + 1280)*stride];
	 X6 = lwb[(me + 1536)*stride];
	 X7 = lwb[(me + 1792)*stride];
	 X8 = lwb[(me + 2048)*stride];
	 X9 = lwb[(me + 2304)*stride];
	X10 = lwb[(me + 2560)*stride];
	X11 = lwb[(me + 2816)*stride];
	X12 = lwb[(me + 3072)*stride];
	X13 = lwb[(me + 3328)*stride];
	X14 = lwb[(me + 3584)*stride];
	X15 = lwb[(me + 3840)*stride];		
	}
	
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
	
	__syncthreads();
			
		
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

		
	__syncthreads();
	

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
	

	__syncthreads();
			
		
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

		
	__syncthreads();


			
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_FWD(twiddles, 15 + 15*(me%16) + 14, X15)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_INV(twiddles, 15 + 15*(me%16) + 14, X15)
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
	
	__syncthreads();
			
		
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
	
		
	__syncthreads();
	

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
	

	__syncthreads();
			
		
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

		
	__syncthreads();
	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_FWD(twiddles, 255 + 15*(me%256) + 14, X15)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_INV(twiddles, 255 + 15*(me%256) + 14, X15)
	}
	
	if(dir == -1)
	{
		FwdRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}

	if(sb == SB_UNIT)
	{		
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
	else
	{
	lwb[(me +    0)*stride] =  X0;
	lwb[(me +  256)*stride] =  X1;
	lwb[(me +  512)*stride] =  X2;
	lwb[(me +  768)*stride] =  X3;
	lwb[(me + 1024)*stride] =  X4;
	lwb[(me + 1280)*stride] =  X5;
	lwb[(me + 1536)*stride] =  X6;
	lwb[(me + 1792)*stride] =  X7;
	lwb[(me + 2048)*stride] =  X8;
	lwb[(me + 2304)*stride] =  X9;
	lwb[(me + 2560)*stride] = X10;
	lwb[(me + 2816)*stride] = X11;
	lwb[(me + 3072)*stride] = X12;
	lwb[(me + 3328)*stride] = X13;
	lwb[(me + 3584)*stride] = X14;
	lwb[(me + 3840)*stride] = X15;		
	}

}


