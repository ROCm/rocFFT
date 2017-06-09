/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef POW2_HIP_H
#define POW2_HIP_H

#include "common.h"
#include "butterfly.h"

template <typename T>
__device__
void fft_1(T *lwb_in, T *lwb_out, const uint rw)
{
	T X0;

	if(rw)
	{
		X0 = lwb_in[0];
		lwb_out[0] = X0;
	}
}

template <typename T, StrideBin sb>
__device__
void fft_2(T *twiddles, T *lwb_in, T *lwb_out, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1;

	if(rw)
	{
		if(sb == SB_UNIT)
		{
		    X0 = lwb_in[0];
		    X1 = lwb_in[1];
		}
		else
		{
		    X0 = lwb_in[0*stride_in];
		    X1 = lwb_in[1*stride_in];			
		}
	}
	
	Rad2<T>(&X0, &X1);
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{
		    lwb_out[0] = X0;
		    lwb_out[1] = X1;
		}
		else
		{
		    lwb_out[0*stride_out] = X0;
		    lwb_out[1*stride_out] = X1;			
		}	
	}
}	



template <typename T, StrideBin sb, int dir>
__device__
void fft_4(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		    X0 = lwb_in[me + 0];
		    X1 = lwb_in[me + 2];	
		}
		else
		{
		    X0 = lwb_in[(me + 0)*stride_in];
		    X1 = lwb_in[(me + 2)*stride_in];			
		}
	}
	
	Rad2<T>(&X0, &X1);

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
	
	Rad2<T>(&X0, &X1);
	
	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		    lwb_out[me + 0] = X0;
		    lwb_out[me + 2] = X1;	
		}
		else
		{
		    lwb_out[(me + 0)*stride_out] = X0;
		    lwb_out[(me + 2)*stride_out] = X1;			
		}	
	}
}



template <typename T, StrideBin sb, int dir>
__device__
void fft_8(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		    X0 = lwb_in[me + 0];
		    X1 = lwb_in[me + 2];	
		    X2 = lwb_in[me + 4];
		    X3 = lwb_in[me + 6];	
		}
		else		
		{
		    X0 = lwb_in[(me + 0)*stride_in];
		    X1 = lwb_in[(me + 2)*stride_in];	
		    X2 = lwb_in[(me + 4)*stride_in];
		    X3 = lwb_in[(me + 6)*stride_in];			
		}
	}
	
	if(dir == -1)
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
		

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
	
	Rad2<T>(&X0, &X1);
	Rad2<T>(&X2, &X3);
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{		
		    vector4_type_t<T>  *lwbv = (vector4_type_t<T>  *)lwb_out;	
		    lwbv[me + 0] = lib_make_vector4< vector4_type_t<T> >(  X0.x,X0.y,X2.x,X2.y);
		    lwbv[me + 2] = lib_make_vector4< vector4_type_t<T> >(  X1.x,X1.y,X3.x,X3.y);
		}		
		else		
		{
		    lwb_out[(me + 0)*stride_out] = X0;
		    lwb_out[(me + 2)*stride_out] = X1;	
		    lwb_out[(me + 4)*stride_out] = X2;
		    lwb_out[(me + 6)*stride_out] = X3;			
		}		
	}
}



template <typename T, StrideBin sb, int dir>
__device__
void fft_16(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3;

	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		    X0 = lwb_in[me + 0];
		    X1 = lwb_in[me + 4];	
		    X2 = lwb_in[me + 8];
		    X3 = lwb_in[me + 12];	
		}
		else
		{
		    X0 = lwb_in[(me +  0)*stride_in];
		    X1 = lwb_in[(me +  4)*stride_in];	
		    X2 = lwb_in[(me +  8)*stride_in];
		    X3 = lwb_in[(me + 12)*stride_in];				
		}
	}
	
	if(dir == -1)
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
		

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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
	
	if(rw)
	{	
		if(sb == SB_UNIT)
		{		
		    lwb_out[me + 0]  = X0;
		    lwb_out[me + 4]  = X1;	
		    lwb_out[me + 8]  = X2;
		    lwb_out[me + 12] = X3;		
		}
		else
		{
		    lwb_out[(me +  0)*stride_out] = X0;
		    lwb_out[(me +  4)*stride_out] = X1;	
		    lwb_out[(me +  8)*stride_out] = X2;
		    lwb_out[(me + 12)*stride_out] = X3;			
		}
	}
}



template <typename T,  StrideBin sb, int dir>
__device__
void fft_32(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;

	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		    X0 = lwb_in[me +  0];
		    X1 = lwb_in[me +  4];
		    X2 = lwb_in[me +  8];
		    X3 = lwb_in[me + 12];
		    X4 = lwb_in[me + 16];
		    X5 = lwb_in[me + 20];
		    X6 = lwb_in[me + 24];
		    X7 = lwb_in[me + 28];
		}
		else
		{
		    X0 = lwb_in[(me +  0)*stride_in];
		    X1 = lwb_in[(me +  4)*stride_in];
		    X2 = lwb_in[(me +  8)*stride_in];
		    X3 = lwb_in[(me + 12)*stride_in];
		    X4 = lwb_in[(me + 16)*stride_in];
		    X5 = lwb_in[(me + 20)*stride_in];
		    X6 = lwb_in[(me + 24)*stride_in];
		    X7 = lwb_in[(me + 28)*stride_in];			
		}
	}				
	
	if(dir == -1)
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
	}	
		
		
	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		    vector4_type_t<T>  *lwbv = (vector4_type_t<T>  *)lwb_out;	
		    lwbv[me +  0] = lib_make_vector4< vector4_type_t<T> >(  X0.x,X0.y,X4.x,X4.y);
		    lwbv[me +  4] = lib_make_vector4< vector4_type_t<T> >(  X1.x,X1.y,X5.x,X5.y);	
		    lwbv[me +  8] = lib_make_vector4< vector4_type_t<T> >(  X2.x,X2.y,X6.x,X6.y);
		    lwbv[me + 12] = lib_make_vector4< vector4_type_t<T> >(  X3.x,X3.y,X7.x,X7.y);			
		}
		else
		{
		    lwb_out[(me +  0)*stride_out] = X0;
		    lwb_out[(me +  4)*stride_out] = X1;
		    lwb_out[(me +  8)*stride_out] = X2;
		    lwb_out[(me + 12)*stride_out] = X3;
		    lwb_out[(me + 16)*stride_out] = X4;
		    lwb_out[(me + 20)*stride_out] = X5;
		    lwb_out[(me + 24)*stride_out] = X6;
		    lwb_out[(me + 28)*stride_out] = X7;				
		}
	}			
	
}


template <typename T,  StrideBin sb, int dir>
__device__
void fft_64(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3;

	if(rw)
	{
		if(sb == SB_UNIT)
		{
		    X0 = lwb_in[me + 0];
		    X1 = lwb_in[me + 16];	
		    X2 = lwb_in[me + 32];
		    X3 = lwb_in[me + 48];	
		}
		else
		{
			
		}
	}
	
	if(dir == -1)
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
		

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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
	
	
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);

		
	if(rw)
	{	
		if(sb == SB_UNIT)
		{
		    lwb_out[me + 0]  = X0;
		    lwb_out[me + 16] = X1;	
		    lwb_out[me + 32] = X2;
		    lwb_out[me + 48] = X3;		
		}
		else
		{
		    lwb_out[(me +  0)*stride_out] = X0;
		    lwb_out[(me + 16)*stride_out] = X1;	
		    lwb_out[(me + 32)*stride_out] = X2;
		    lwb_out[(me + 48)*stride_out] = X3;			
		}
	}
}


template <typename T,  StrideBin sb, int dir>
__device__
void fft_128(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const uint rw, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(rw)
	{
		if(sb == SB_UNIT)
		{			
		    X0 = lwb_in[me + 0];
		    X1 = lwb_in[me + 16];
		    X2 = lwb_in[me + 32];
		    X3 = lwb_in[me + 48];
		    X4 = lwb_in[me + 64];
		    X5 = lwb_in[me + 80];
		    X6 = lwb_in[me + 96];
		    X7 = lwb_in[me + 112];
		}
		else
		{
		    X0 = lwb_in[(me +   0)*stride_in];
		    X1 = lwb_in[(me +  16)*stride_in];
		    X2 = lwb_in[(me +  32)*stride_in];
		    X3 = lwb_in[(me +  48)*stride_in];
		    X4 = lwb_in[(me +  64)*stride_in];
		    X5 = lwb_in[(me +  80)*stride_in];
		    X6 = lwb_in[(me +  96)*stride_in];
		    X7 = lwb_in[(me + 112)*stride_in];			
		}
	}
	
	if(dir == -1)
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
	}	
		
		
	if(rw)
	{
		if(sb == SB_UNIT)
		{
		    vector4_type_t<T>  *lwbv = (vector4_type_t<T>  *)lwb_out;	
		    lwbv[me +  0] = lib_make_vector4< vector4_type_t<T> >(  X0.x,X0.y,X4.x,X4.y);
		    lwbv[me + 16] = lib_make_vector4< vector4_type_t<T> >(  X1.x,X1.y,X5.x,X5.y);	
		    lwbv[me + 32] = lib_make_vector4< vector4_type_t<T> >(  X2.x,X2.y,X6.x,X6.y);
		    lwbv[me + 48] = lib_make_vector4< vector4_type_t<T> >(  X3.x,X3.y,X7.x,X7.y);	
		}
		else
		{
		    lwb_out[(me +   0)*stride_out] = X0;
		    lwb_out[(me +  16)*stride_out] = X1;
		    lwb_out[(me +  32)*stride_out] = X2;
		    lwb_out[(me +  48)*stride_out] = X3;
		    lwb_out[(me +  64)*stride_out] = X4;
		    lwb_out[(me +  80)*stride_out] = X5;
		    lwb_out[(me +  96)*stride_out] = X6;
		    lwb_out[(me + 112)*stride_out] = X7;			
		}
	}	
	
}


template <typename T,  StrideBin sb, int dir>
__device__
void fft_256(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3;

	if(sb == SB_UNIT)
	{
	    X0 = lwb_in[me +   0];
	    X1 = lwb_in[me +  64];	
	    X2 = lwb_in[me + 128];
	    X3 = lwb_in[me + 192];	
	}
	else	
	{
	    X0 = lwb_in[(me +   0)*stride_in];
	    X1 = lwb_in[(me +  64)*stride_in];	
	    X2 = lwb_in[(me + 128)*stride_in];
	    X3 = lwb_in[(me + 192)*stride_in];			
	}
	

	if(dir == -1)
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
	
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
		
		
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);


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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
	else
		InvRad4<T>(&X0, &X1, &X2, &X3);
		
	if(sb == SB_UNIT)
	{
	    lwb_out[me +   0] = X0;
	    lwb_out[me +  64] = X1;	
	    lwb_out[me + 128] = X2;
	    lwb_out[me + 192] = X3;		
	}
	else
	{
	    lwb_out[(me +   0)*stride_out] = X0;
	    lwb_out[(me +  64)*stride_out] = X1;	
	    lwb_out[(me + 128)*stride_out] = X2;
	    lwb_out[(me + 192)*stride_out] = X3;		
	}

}


template <typename T,  StrideBin sb, int dir>
__device__
void fft_512(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	X0 = lwb_in[me +   0];
	X1 = lwb_in[me +  64];
	X2 = lwb_in[me + 128];
	X3 = lwb_in[me + 192];
	X4 = lwb_in[me + 256];
	X5 = lwb_in[me + 320];
	X6 = lwb_in[me + 384];
	X7 = lwb_in[me + 448];
	}
	else
	{
	X0 = lwb_in[(me +   0)*stride_in];
	X1 = lwb_in[(me +  64)*stride_in];
	X2 = lwb_in[(me + 128)*stride_in];
	X3 = lwb_in[(me + 192)*stride_in];
	X4 = lwb_in[(me + 256)*stride_in];
	X5 = lwb_in[(me + 320)*stride_in];
	X6 = lwb_in[(me + 384)*stride_in];
	X7 = lwb_in[(me + 448)*stride_in];		
	}
					
	
	if(dir == -1)
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

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
		T W;
		real_type_t<T> TR, TI;
		
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
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


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
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

	if(sb == SB_UNIT)
	{
	    lwb_out[me +   0] = X0;
	    lwb_out[me +  64] = X1;
	    lwb_out[me + 128] = X2;
	    lwb_out[me + 192] = X3;
	    lwb_out[me + 256] = X4;
	    lwb_out[me + 320] = X5;
	    lwb_out[me + 384] = X6;
	    lwb_out[me + 448] = X7;		
	}
	else
	{
	    lwb_out[(me +   0)*stride_out] = X0;
	    lwb_out[(me +  64)*stride_out] = X1;
	    lwb_out[(me + 128)*stride_out] = X2;
	    lwb_out[(me + 192)*stride_out] = X3;
	    lwb_out[(me + 256)*stride_out] = X4;
	    lwb_out[(me + 320)*stride_out] = X5;
	    lwb_out[(me + 384)*stride_out] = X6;
	    lwb_out[(me + 448)*stride_out] = X7;			
	}
		
}
	

template <typename T,  StrideBin sb, int dir>
__device__
void fft_1024(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	    X0 = lwb_in[me +   0];
	    X1 = lwb_in[me + 128];
	    X2 = lwb_in[me + 256];
	    X3 = lwb_in[me + 384];
	    X4 = lwb_in[me + 512];
	    X5 = lwb_in[me + 640];
	    X6 = lwb_in[me + 768];
	    X7 = lwb_in[me + 896];
	}
	else
	{
	    X0 = lwb_in[(me +   0)*stride_in];
	    X1 = lwb_in[(me + 128)*stride_in];
	    X2 = lwb_in[(me + 256)*stride_in];
	    X3 = lwb_in[(me + 384)*stride_in];
	    X4 = lwb_in[(me + 512)*stride_in];
	    X5 = lwb_in[(me + 640)*stride_in];
	    X6 = lwb_in[(me + 768)*stride_in];
	    X7 = lwb_in[(me + 896)*stride_in];		
	}
					
	
	if(dir == -1)
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

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
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
	}	
	
	if(sb == SB_UNIT)	
	{
		vector4_type_t<T>  *lwbv = (vector4_type_t<T>  *)lwb_out;	
		lwbv[me +   0] = lib_make_vector4< vector4_type_t<T> >(  X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 128] = lib_make_vector4< vector4_type_t<T> >(  X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 256] = lib_make_vector4< vector4_type_t<T> >(  X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 384] = lib_make_vector4< vector4_type_t<T> >(  X3.x,X3.y,X7.x,X7.y);			
	}	
	else
	{
		lwb_out[(me +   0)*stride_out] = X0;
		lwb_out[(me + 128)*stride_out] = X1;
		lwb_out[(me + 256)*stride_out] = X2;
		lwb_out[(me + 384)*stride_out] = X3;
		lwb_out[(me + 512)*stride_out] = X4;
		lwb_out[(me + 640)*stride_out] = X5;
		lwb_out[(me + 768)*stride_out] = X6;
		lwb_out[(me + 896)*stride_out] = X7;		
	}
}


	
template <typename T,  StrideBin sb, int dir>
__device__
void fft_2048(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;
	
	if(sb == SB_UNIT)
	{
	    X0 = lwb_in[me +    0];
	    X1 = lwb_in[me +  256];
	    X2 = lwb_in[me +  512];
	    X3 = lwb_in[me +  768];
	    X4 = lwb_in[me + 1024];
	    X5 = lwb_in[me + 1280];
	    X6 = lwb_in[me + 1536];
	    X7 = lwb_in[me + 1792];
	}
	else
	{
	    X0 = lwb_in[(me +    0)*stride_in];
	    X1 = lwb_in[(me +  256)*stride_in];
	    X2 = lwb_in[(me +  512)*stride_in];
	    X3 = lwb_in[(me +  768)*stride_in];
	    X4 = lwb_in[(me + 1024)*stride_in];
	    X5 = lwb_in[(me + 1280)*stride_in];
	    X6 = lwb_in[(me + 1536)*stride_in];
	    X7 = lwb_in[(me + 1792)*stride_in];		
	}
					
	
	if(dir == -1)
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	
	

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
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


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
		FwdRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);

		
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
		FwdRad4<T>(&X0, &X1, &X2, &X3);
		FwdRad4<T>(&X4, &X5, &X6, &X7);
	}
	else	
	{
		InvRad4<T>(&X0, &X1, &X2, &X3);
		InvRad4<T>(&X4, &X5, &X6, &X7);
	}	
		
	if(sb == SB_UNIT)		
	{
		vector4_type_t<T>  *lwbv = (vector4_type_t<T>  *)lwb_out;	
		lwbv[me +   0] = lib_make_vector4< vector4_type_t<T> >(  X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 256] = lib_make_vector4< vector4_type_t<T> >(  X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 512] = lib_make_vector4< vector4_type_t<T> >(  X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 768] = lib_make_vector4< vector4_type_t<T> >(  X3.x,X3.y,X7.x,X7.y);			
	}
	else
	{
		lwb_out[(me +    0)*stride_out] = X0;
		lwb_out[(me +  256)*stride_out] = X1;
		lwb_out[(me +  512)*stride_out] = X2;
		lwb_out[(me +  768)*stride_out] = X3;
		lwb_out[(me + 1024)*stride_out] = X4;
		lwb_out[(me + 1280)*stride_out] = X5;
		lwb_out[(me + 1536)*stride_out] = X6;
		lwb_out[(me + 1792)*stride_out] = X7;		
	}
}

template <typename T,  StrideBin sb, int dir>
__device__
void fft_4096(T *twiddles, T *lwb_in, T *lwb_out, real_type_t<T>   *lds, const uint me, const ulong stride_in, const ulong stride_out)
{
	T X0, X1, X2, X3, X4, X5, X6, X7;
	T X8, X9, X10, X11, X12, X13, X14, X15;
	
	if(sb == SB_UNIT)
	{
	     X0 = lwb_in[me +    0];
	     X1 = lwb_in[me +  256];
	     X2 = lwb_in[me +  512];
	     X3 = lwb_in[me +  768];
	     X4 = lwb_in[me + 1024];
	     X5 = lwb_in[me + 1280];
	     X6 = lwb_in[me + 1536];
	     X7 = lwb_in[me + 1792];
	     X8 = lwb_in[me + 2048];
	     X9 = lwb_in[me + 2304];
	    X10 = lwb_in[me + 2560];
	    X11 = lwb_in[me + 2816];
	    X12 = lwb_in[me + 3072];
	    X13 = lwb_in[me + 3328];
	    X14 = lwb_in[me + 3584];
	    X15 = lwb_in[me + 3840];
	}
	else
	{
	     X0 = lwb_in[(me +    0)*stride_in];
	     X1 = lwb_in[(me +  256)*stride_in];
	     X2 = lwb_in[(me +  512)*stride_in];
	     X3 = lwb_in[(me +  768)*stride_in];
	     X4 = lwb_in[(me + 1024)*stride_in];
	     X5 = lwb_in[(me + 1280)*stride_in];
	     X6 = lwb_in[(me + 1536)*stride_in];
	     X7 = lwb_in[(me + 1792)*stride_in];
	     X8 = lwb_in[(me + 2048)*stride_in];
	     X9 = lwb_in[(me + 2304)*stride_in];
	    X10 = lwb_in[(me + 2560)*stride_in];
	    X11 = lwb_in[(me + 2816)*stride_in];
	    X12 = lwb_in[(me + 3072)*stride_in];
	    X13 = lwb_in[(me + 3328)*stride_in];
	    X14 = lwb_in[(me + 3584)*stride_in];
	    X15 = lwb_in[(me + 3840)*stride_in];		
	}
	
	if(dir == -1)
	{
		FwdRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
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
		FwdRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
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
		FwdRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}
	else
	{
		InvRad16<T>(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7, &X8, &X9, &X10, &X11, &X12, &X13, &X14, &X15);
	}

	if(sb == SB_UNIT)
	{		
	    lwb_out[me +    0] =  X0;
	    lwb_out[me +  256] =  X1;
	    lwb_out[me +  512] =  X2;
	    lwb_out[me +  768] =  X3;
	    lwb_out[me + 1024] =  X4;
	    lwb_out[me + 1280] =  X5;
	    lwb_out[me + 1536] =  X6;
	    lwb_out[me + 1792] =  X7;
	    lwb_out[me + 2048] =  X8;
	    lwb_out[me + 2304] =  X9;
	    lwb_out[me + 2560] = X10;
	    lwb_out[me + 2816] = X11;
	    lwb_out[me + 3072] = X12;
	    lwb_out[me + 3328] = X13;
	    lwb_out[me + 3584] = X14;
	    lwb_out[me + 3840] = X15;	
	}
	else
	{
	    lwb_out[(me +    0)*stride_out] =  X0;
	    lwb_out[(me +  256)*stride_out] =  X1;
	    lwb_out[(me +  512)*stride_out] =  X2;
	    lwb_out[(me +  768)*stride_out] =  X3;
	    lwb_out[(me + 1024)*stride_out] =  X4;
	    lwb_out[(me + 1280)*stride_out] =  X5;
	    lwb_out[(me + 1536)*stride_out] =  X6;
	    lwb_out[(me + 1792)*stride_out] =  X7;
	    lwb_out[(me + 2048)*stride_out] =  X8;
	    lwb_out[(me + 2304)*stride_out] =  X9;
	    lwb_out[(me + 2560)*stride_out] = X10;
	    lwb_out[(me + 2816)*stride_out] = X11;
	    lwb_out[(me + 3072)*stride_out] = X12;
	    lwb_out[(me + 3328)*stride_out] = X13;
	    lwb_out[(me + 3584)*stride_out] = X14;
	    lwb_out[(me + 3840)*stride_out] = X15;		
	}

}

#endif // POW2_HIP_H

