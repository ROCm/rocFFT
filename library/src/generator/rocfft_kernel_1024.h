#include "../kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*stride_in];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 )*stride_in];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 256 )*stride_in];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 384 )*stride_in];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 512 )*stride_in];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 640 )*stride_in];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 768 )*stride_in];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 896 )*stride_in];
	}



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 384 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 640 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 896 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 384 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 640 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 896 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 3];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R4).x) - (W.y * (*R4).y);
		TI = (W.y * (*R4).x) + (W.x * (*R4).y);
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 4];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 5];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 6];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 8 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 16 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 24 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 32 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 40 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 48 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 56 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 8 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 16 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 24 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 32 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 40 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 48 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 56 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 64 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 128 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 192 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 0 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 64 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 128 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 192 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 64 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 128 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 192 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 0 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 64 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 128 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 192 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass3_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	 //Optimization: coalescing into float4/double4 write
	if(sb == SB_UNIT) {
	vector4_type_t<T> *buff4g = (vector4_type_t<T>*)bufOut;
	
	buff4g[ 1*me + 0 + 0 ] = vector4_type_t<T>((*R0).x, (*R0).y, (*R4).x, (*R4).y) ;
	buff4g[ 1*me + 0 + 128 ] = vector4_type_t<T>((*R1).x, (*R1).y, (*R5).x, (*R5).y) ;
	buff4g[ 1*me + 0 + 256 ] = vector4_type_t<T>((*R2).x, (*R2).y, (*R6).x, (*R6).y) ;
	buff4g[ 1*me + 0 + 384 ] = vector4_type_t<T>((*R3).x, (*R3).y, (*R7).x, (*R7).y) ;
	}
	else{ // such optimization is not possible 
	bufOut[outOffset + ( 2*me + 0 + 0 )*stride_out] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 )*stride_out] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 256 )*stride_out] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 256 )*stride_out] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 512 )*stride_out] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 512 )*stride_out] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 768 )*stride_out] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 768 )*stride_out] = (*R7);
	}
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*stride_in];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 )*stride_in];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 256 )*stride_in];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 384 )*stride_in];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 512 )*stride_in];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 640 )*stride_in];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 768 )*stride_in];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 896 )*stride_in];
	}



	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 384 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 640 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*1 + 0 + 896 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 256 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 384 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 512 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 640 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*1 + 0 + 896 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 3];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R4).x) + (W.y * (*R4).y);
		TI = -(W.y * (*R4).x) + (W.x * (*R4).y);
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 4];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 5];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 7*((1*me + 0)%8) + 6];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 8 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 16 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 24 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 32 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 40 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 48 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 56 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 8 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 16 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 24 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 32 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 40 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 48 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/8)*64 + (1*me + 0)%8 + 56 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, real_type_t<T> *bufOutRe, real_type_t<T> *bufOutIm, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[63 + 3*((2*me + 1)%64) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 0 ) ] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 64 ) ] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 128 ) ] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 192 ) ] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 0 ) ] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 64 ) ] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 128 ) ] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 192 ) ] = (*R7).x;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 0 ) ] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 64 ) ] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 128 ) ] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/64)*256 + (2*me + 0)%64 + 192 ) ] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 0 ) ] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 64 ) ] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 128 ) ] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/64)*256 + (2*me + 1)%64 + 192 ) ] = (*R7).y;
	}


	__syncthreads();

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 256 ) ];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 256 ) ];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 512 ) ];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 512 ) ];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 768 ) ];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 768 ) ];
	}


	__syncthreads();

}

template <typename T, StrideBin sb> 
__device__ void
InvPass3_len1024(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, real_type_t<T> *bufInRe, real_type_t<T> *bufInIm, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{




	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 0)%256) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[255 + 3*((2*me + 1)%256) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	 //Optimization: coalescing into float4/double4 write
	if(sb == SB_UNIT) {
	vector4_type_t<T> *buff4g = (vector4_type_t<T>*)bufOut;
	
	buff4g[ 1*me + 0 + 0 ] = vector4_type_t<T>((*R0).x, (*R0).y, (*R4).x, (*R4).y) ;
	buff4g[ 1*me + 0 + 128 ] = vector4_type_t<T>((*R1).x, (*R1).y, (*R5).x, (*R5).y) ;
	buff4g[ 1*me + 0 + 256 ] = vector4_type_t<T>((*R2).x, (*R2).y, (*R6).x, (*R6).y) ;
	buff4g[ 1*me + 0 + 384 ] = vector4_type_t<T>((*R3).x, (*R3).y, (*R7).x, (*R7).y) ;
	}
	else{ // such optimization is not possible 
	bufOut[outOffset + ( 2*me + 0 + 0 )*stride_out] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 )*stride_out] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 256 )*stride_out] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 256 )*stride_out] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 512 )*stride_out] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 512 )*stride_out] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 768 )*stride_out] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 768 )*stride_out] = (*R7);
	}
	}

}

//Configuration: number of threads per thread block: 128 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len1024( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t stride_in, const size_t stride_out, const int batch_count, T * __restrict__   gb)
{
	unsigned int me = hipThreadIdx_x;
	unsigned int batch = hipBlockIdx_x;

	__shared__  real_type_t<T> lds[1024];

	unsigned int ioOffset;
	T *lwb;

	T R0, R1, R2, R3, R4, R5, R6, R7;

	unsigned int rw = 1;

	unsigned int b = 0;

	ioOffset = batch*1024*stride_in;
	lwb = gb + ioOffset;

	FwdPass0_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lwb, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass1_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass2_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass3_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lwb, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

//Configuration: number of threads per thread block: 128 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len1024( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t stride_in, const size_t stride_out, const int batch_count, T * __restrict__   gb)
{
	unsigned int me = hipThreadIdx_x;
	unsigned int batch = hipBlockIdx_x;

	__shared__  real_type_t<T> lds[1024];

	unsigned int ioOffset;
	T *lwb;

	T R0, R1, R2, R3, R4, R5, R6, R7;

	unsigned int rw = 1;

	unsigned int b = 0;

	ioOffset = batch*1024*stride_in;
	lwb = gb + ioOffset;

	InvPass0_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lwb, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass1_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass2_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass3_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lwb, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

//Configuration: number of threads per thread block: 128 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len1024( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t stride_in, const size_t stride_out, const int batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = hipThreadIdx_x;
	unsigned int batch = hipBlockIdx_x;

	__shared__  real_type_t<T> lds[1024];

	unsigned int iOffset;
	unsigned int oOffset;
	T *lwbIn;
	T *lwbOut;

	T R0, R1, R2, R3, R4, R5, R6, R7;

	unsigned int rw = 1;

	unsigned int b = 0;

	iOffset = batch*1024*stride_in;
	oOffset = batch*1024*stride_out;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	FwdPass0_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass1_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass2_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass3_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

//Configuration: number of threads per thread block: 128 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len1024( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t stride_in, const size_t stride_out, const int batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = hipThreadIdx_x;
	unsigned int batch = hipBlockIdx_x;

	__shared__  real_type_t<T> lds[1024];

	unsigned int iOffset;
	unsigned int oOffset;
	T *lwbIn;
	T *lwbOut;

	T R0, R1, R2, R3, R4, R5, R6, R7;

	unsigned int rw = 1;

	unsigned int b = 0;

	iOffset = batch*1024*stride_in;
	oOffset = batch*1024*stride_out;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;

	InvPass0_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lwbIn, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass1_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass2_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass3_len1024<T, sb>(twiddles, stride_in, stride_out, 1, b, me, 0, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

