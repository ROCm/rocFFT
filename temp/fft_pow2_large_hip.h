#ifndef FFT_POW2_LARGE_HIP_H
#define FFT_POW2_LARGE_HIP_H

__device__ float2
TWLstep2(float2 *twiddles, size_t u)
{
	size_t j = u & 255;
	float2 result = twiddles[j];
	u >>= 8;
	j = u & 255;
	result = float2((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
		(result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
	return result;
}


__device__ float2
TWLstep3(float2 *twiddles, size_t u)
{
	size_t j = u & 255;
	float2 result = twiddles[j];
	u >>= 8;
	j = u & 255;
	result = float2((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
		(result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
	u >>= 8;
	j = u & 255;
	result = float2((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
		(result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
	return result;
}




template<int dir, int twl>
__device__ void
lfft_64(float2 *twiddles_64, float2 *twiddles_large, float2 *lds, uint me, uint b)
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
	
	__syncthreads();
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	__syncthreads();
	

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
	
	__syncthreads();
	
	X0 = lds[me + 0];
	X1 = lds[me + 16];	
	X2 = lds[me + 32];
	X3 = lds[me + 48];
	
	__syncthreads();
	

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


	if(twl == 2)
	{
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me +  0)*b, X0)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me + 16)*b, X1)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me + 32)*b, X2)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me + 48)*b, X3)			
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me +  0)*b, X0)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me + 16)*b, X1)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me + 32)*b, X2)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me + 48)*b, X3)				
		}
	}
	else if(twl == 3)
	{
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me +  0)*b, X0)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me + 16)*b, X1)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me + 32)*b, X2)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me + 48)*b, X3)			
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me +  0)*b, X0)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me + 16)*b, X1)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me + 32)*b, X2)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me + 48)*b, X3)				
		}		
	}
	
	lds[me + 0]  = X0;
	lds[me + 16] = X1;	
	lds[me + 32] = X2;
	lds[me + 48] = X3;		

}


template<int dir, int twl>
__device__ void
lfft_128(float2 *twiddles_128, float2 *twiddles_large, float2 *lds, uint me, uint b)
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
	
	__syncthreads();
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	__syncthreads();


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
	
	__syncthreads();
	
	X0 = lds[(2*me + 0) +  0];
	X1 = lds[(2*me + 0) + 32];
	X2 = lds[(2*me + 0) + 64];
	X3 = lds[(2*me + 0) + 96];

	X4 = lds[(2*me + 1) +  0];
	X5 = lds[(2*me + 1) + 32];
	X6 = lds[(2*me + 1) + 64];
	X7 = lds[(2*me + 1) + 96];	

	__syncthreads();

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
		
	
	if(twl == 2)
	{	
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 0) +  0)*b, X0)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 0) + 32)*b, X1)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 0) + 64)*b, X2)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 0) + 96)*b, X3)	

			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 1) +  0)*b, X4)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 1) + 32)*b, X5)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 1) + 64)*b, X6)
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, ((2*me + 1) + 96)*b, X7)
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 0) +  0)*b, X0)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 0) + 32)*b, X1)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 0) + 64)*b, X2)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 0) + 96)*b, X3)	

			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 1) +  0)*b, X4)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 1) + 32)*b, X5)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 1) + 64)*b, X6)
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, ((2*me + 1) + 96)*b, X7)
		}
	}
	else if(twl == 3)
	{
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 0) +  0)*b, X0)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 0) + 32)*b, X1)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 0) + 64)*b, X2)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 0) + 96)*b, X3)	

			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 1) +  0)*b, X4)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 1) + 32)*b, X5)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 1) + 64)*b, X6)
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, ((2*me + 1) + 96)*b, X7)
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 0) +  0)*b, X0)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 0) + 32)*b, X1)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 0) + 64)*b, X2)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 0) + 96)*b, X3)	

			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 1) +  0)*b, X4)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 1) + 32)*b, X5)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 1) + 64)*b, X6)
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, ((2*me + 1) + 96)*b, X7)
		}		
	}
	
	
	{
		float4 *ldsv = (float4 *)lds;	
		ldsv[me +  0] = float4(X0.x,X0.y,X4.x,X4.y);
		ldsv[me + 16] = float4(X1.x,X1.y,X5.x,X5.y);	
		ldsv[me + 32] = float4(X2.x,X2.y,X6.x,X6.y);
		ldsv[me + 48] = float4(X3.x,X3.y,X7.x,X7.y);			
	}	
	
}


template<int dir, int twl>
__device__ void
lfft_256(float2 *twiddles_256, float2 *twiddles_large, float2 *lds, uint me, uint b)
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
	
	__syncthreads();
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	__syncthreads();
	
	
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
	
	__syncthreads();
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	__syncthreads();
	

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
	
	__syncthreads();
	
	X0 = lds[me +   0];
	X1 = lds[me +  64];	
	X2 = lds[me + 128];
	X3 = lds[me + 192];
	
	__syncthreads();
	
	
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
		
	if(twl == 2)
	{
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me +   0)*b, X0)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me +  64)*b, X1)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me + 128)*b, X2)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep2, twiddles_large, (me + 192)*b, X3)			
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me +   0)*b, X0)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me +  64)*b, X1)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me + 128)*b, X2)	
			TWIDDLE_3STEP_MUL_INV(TWLstep2, twiddles_large, (me + 192)*b, X3)				
		}
	}
	else if(twl == 3)
	{
		if(dir == -1)
		{
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me +   0)*b, X0)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me +  64)*b, X1)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me + 128)*b, X2)	
			TWIDDLE_3STEP_MUL_FWD(TWLstep3, twiddles_large, (me + 192)*b, X3)			
		}
		else
		{
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me +   0)*b, X0)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me +  64)*b, X1)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me + 128)*b, X2)	
			TWIDDLE_3STEP_MUL_INV(TWLstep3, twiddles_large, (me + 192)*b, X3)				
		}		
	}
	
	lds[me +   0] = X0;
	lds[me +  64] = X1;	
	lds[me + 128] = X2;
	lds[me + 192] = X3;		

}


//////////////////////////////////////////

template<StrideBin sb, int dir>
__device__
void fft_64_128_bcc(float2 *twiddles_64, float2 *twiddles_8192, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	uint b = 0;

	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%16) + (me/16)*128 + t*1024];
		else
			R0 = lwbIn[((me%16) + (me/16)*128 + t*1024)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%16) + (me/16)*128 + t*1024] = R0;
		else
			lwbOut[((me%16) + (me/16)*128 + t*1024)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_128_64_brc(float2 *twiddles_128, float2 * lwbIn, float2 * lwbOut, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	
	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[me + t*128];
		else
			R0 = lwbIn[(me + t*128)*stride_i];

		lds[t*128 + me] = R0;
	}

	__syncthreads();

	lfft_128<dir, 0>(twiddles_128, 0, lds + (me/16)*128, me%16, 0);

	__syncthreads();


	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*64 + t*1024] = R0;
		else
			lwbOut[((me%8) + (me/8)*64 + t*1024)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_64_256_bcc(float2 *twiddles_64, float2 *twiddles_16384, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	uint b = 0;

	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%16) + (me/16)*256 + t*2048];
		else
			R0 = lwbIn[((me%16) + (me/16)*256 + t*2048)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%16) + (me/16)*256 + t*2048] = R0;
		else
			lwbOut[((me%16) + (me/16)*256 + t*2048)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_256_64_brc(float2 *twiddles_256, float2 * lwbIn, float2 * lwbOut, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[2048];

	float2 R0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[me + t*256];
		else
			R0 = lwbIn[(me + t*256)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*64 + t*2048] = R0;
		else
			lwbOut[((me%8) + (me/8)*64 + t*2048)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_128_256_bcc(float2 *twiddles_128, float2 *twiddles_32768, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	uint b = 0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%8) + (me/8)*256 + t*4096];
		else
			R0 = lwbIn[((me%8) + (me/8)*256 + t*4096)*stride_i];

		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	b = (batch%32)*8 + (me/16);

	lfft_128<dir, 2>(twiddles_128, twiddles_32768, lds + (me/16)*128, me%16, b);
	
	__syncthreads();

	
	for(uint t=0; t<8; t++)
	{
		R0 = lds[t*16 + (me%8)*128 + (me/8)];
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*256 + t*4096] = R0;
		else
			lwbOut[((me%8) + (me/8)*256 + t*4096)*stride_o] = R0;

	}
}


template<StrideBin sb, int dir>
__device__
void fft_256_128_brc(float2 *twiddles_256, float2 * lwbIn, float2 * lwbOut, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[2048];

	float2 R0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[me + t*256];
		else
			R0 = lwbIn[(me + t*256)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*128 + t*4096] = R0;
		else
			lwbOut[((me%8) + (me/8)*128 + t*4096)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_256_256_bcc(float2 *twiddles_256, float2 *twiddles_65536, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[2048];

	float2 R0;

	uint b = 0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%8) + (me/8)*256 + t*8192];
		else
			R0 = lwbIn[((me%8) + (me/8)*256 + t*8192)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
		else
			lwbOut[((me%8) + (me/8)*256 + t*8192)*stride_o] = R0;

	}
}


template<StrideBin sb, int dir>
__device__
void fft_256_256_brc(float2 *twiddles_256, float2 * lwbIn, float2 * lwbOut, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[2048];

	float2 R0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[me + t*256];
		else
			R0 = lwbIn[(me + t*256)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%8) + (me/8)*256 + t*8192] = R0;
		else
			lwbOut[((me%8) + (me/8)*256 + t*8192)*stride_o] = R0;

	}

}


template<StrideBin sb, int dir>
__device__
void fft_64_2048_bcc(float2 *twiddles_64, float2 *twiddles_131072, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	uint b = 0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%16) + (me/16)*2048 + t*16384];
		else
			R0 = lwbIn[((me%16) + (me/16)*2048 + t*16384)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%16) + (me/16)*2048 + t*16384] = R0;
		else
			lwbOut[((me%16) + (me/16)*2048 + t*16384)*stride_o] = R0;

	}
}


template<StrideBin sb, int dir>
__device__
void fft_64_4096_bcc(float2 *twiddles_64, float2 *twiddles_262144, float2 * lwbIn, float2 * lwbOut, const uint batch, const ulong stride_i, const ulong stride_o)
{
	uint me = hipThreadIdx_x;

	__shared__ float2 lds[1024];

	float2 R0;

	uint b = 0;


	for(uint t=0; t<8; t++)
	{
		if(sb == SB_UNIT)
			R0 = lwbIn[(me%16) + (me/16)*4096 + t*32768];
		else
			R0 = lwbIn[((me%16) + (me/16)*4096 + t*32768)*stride_i];

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
		
		if(sb == SB_UNIT)
			lwbOut[(me%16) + (me/16)*4096 + t*32768] = R0;
		else
			lwbOut[((me%16) + (me/16)*4096 + t*32768)*stride_o] = R0;

	}
}


#endif // FFT_POW2_LARGE_HIP_H

