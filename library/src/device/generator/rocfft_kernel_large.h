#include "kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 0) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R0).x) - (W.y * (*R0).y);
		TI = (W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 16) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 32) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 48) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 16 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 32 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len64_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 0) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R0).x) + (W.y * (*R0).y);
		TI = -(W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 16) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 32) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%16 + 48) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 16 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 32 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void 
fwd_len64_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	FwdPass0_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass1_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass2_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

template <typename T, StrideBin sb> 
__device__ void 
back_len64_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	InvPass0_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass1_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass2_len64_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len64_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/16);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/16))*stride_in[2] + (counter_mod % (lengths[1]/16))*16;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/16))*16 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len64_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*512 + (me/16)*64, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwb[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len64_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/16);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/16))*stride_in[2] + (counter_mod % (lengths[1]/16))*16;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/16))*16 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len64_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*512 + (me/16)*64, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwb[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len64_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/16);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/16))*stride_in[2] + (counter_mod % (lengths[1]/16))*16;
	iOffset += (counter_mod/(lengths[1]/16))*stride_out[2] + (counter_mod % (lengths[1]/16))*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/16))*16 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len64_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*512 + (me/16)*64, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*stride_out[0] + t*stride_out[0]*8] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len64_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/16);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/16))*stride_in[2] + (counter_mod % (lengths[1]/16))*16;
	iOffset += (counter_mod/(lengths[1]/16))*stride_out[2] + (counter_mod % (lengths[1]/16))*16;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%16) + (me/16)*stride_in[0] + t*stride_in[0]*8];
		lds[t*8 + (me%16)*64 + (me/16)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/16))*16 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len64_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*512 + (me/16)*64, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*8 + (me%16)*64 + (me/16)];
		lwbOut[(me%16) + (me/16)*stride_out[0] + t*stride_out[0]*8] = R0;
	}

}

#include "kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 ) ];
	}



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 ) ] = (*R1);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 ) ] = (*R2);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 ) ] = (*R3);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 ) ] = (*R4);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 ) ] = (*R5);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 ) ] = (*R6);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 0) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R0).x) - (W.y * (*R0).y);
		TI = (W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 32) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 64) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 96) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 0) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R4).x) - (W.y * (*R4).y);
		TI = (W.y * (*R4).x) + (W.x * (*R4).y);
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 32) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 64) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 96) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 ) ] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 32 ) ] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 32 ) ] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 64 ) ] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 64 ) ] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 96 ) ] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 96 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 ) ];
	}



	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 ) ] = (*R1);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 ) ] = (*R2);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 ) ] = (*R3);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 ) ] = (*R4);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 ) ] = (*R5);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 ) ] = (*R6);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len128_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 0) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R0).x) + (W.y * (*R0).y);
		TI = -(W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 32) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 64) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 0)%32 + 96) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 0) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R4).x) + (W.y * (*R4).y);
		TI = -(W.y * (*R4).x) + (W.x * (*R4).y);
		(*R4).x = TR;
		(*R4).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 32) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 64) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((2*me + 1)%32 + 96) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 ) ] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 32 ) ] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 32 ) ] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 64 ) ] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 64 ) ] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 96 ) ] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 96 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void 
fwd_len128_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7;
	FwdPass0_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	FwdPass1_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	FwdPass2_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
}

template <typename T, StrideBin sb> 
__device__ void 
back_len128_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7;
	InvPass0_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	InvPass1_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	InvPass2_len128_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len128_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16];
		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len128_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len128_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16];
		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len128_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len128_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16];
		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len128_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len128_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16];
		lds[t*16 + (me%8)*128 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*8 + (me/16);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len128_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*16] = R0;
	}

}

#include "kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 16 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 32 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass3_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 0) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R0).x) - (W.y * (*R0).y);
		TI = (W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 64) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 128) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 192) * b );
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 64 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 128 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 192 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 16 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 32 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass3_len256_BCT_C2C(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 0) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R0).x) + (W.y * (*R0).y);
		TI = -(W.y * (*R0).x) + (W.x * (*R0).y);
		(*R0).x = TR;
		(*R0).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 64) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 128) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = TW3step(twiddles_large, ((1*me + 0)%64 + 192) * b );
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 64 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 128 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 192 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void 
fwd_len256_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	FwdPass0_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass1_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass2_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass3_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

template <typename T, StrideBin sb> 
__device__ void 
back_len256_BCT_C2C_device(const T *twiddles, const T *twiddles_large, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	InvPass0_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass1_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass2_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass3_len256_BCT_C2C<T, sb>(twiddles, twiddles_large, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len256_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32];
		lds[t*32 + (me%8)*256 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*4 + (me/64);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len256_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len256_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32];
		lds[t*32 + (me%8)*256 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*4 + (me/64);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len256_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len256_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32];
		lds[t*32 + (me%8)*256 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*4 + (me/64);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len256_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len256_BCT_C2C( hipLaunchParm lp, const T * __restrict__ twiddles, const T * __restrict__ twiddles_large, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8;
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32];
		lds[t*32 + (me%8)*256 + (me/8)] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		b = (batch % (lengths[1]/8))*8 + t*4 + (me/64);

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len256_BCT_C2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*32] = R0;
	}

}

#include "kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 ) ];
	}



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 ) ] = (*R1);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 ) ] = (*R2);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 ) ] = (*R3);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 ) ] = (*R4);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 ) ] = (*R5);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 ) ] = (*R6);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 ) ] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 32 ) ] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 32 ) ] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 64 ) ] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 64 ) ] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 96 ) ] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 96 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 ) ];
	}



	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 ) ] = (*R3);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 ) ] = (*R4);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 ) ] = (*R5);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 ) ] = (*R6);
	bufOut[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 ) ] = (*R1);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 ) ] = (*R2);
	bufOut[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 ) ] = (*R3);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 ) ] = (*R4);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 ) ] = (*R5);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 ) ] = (*R6);
	bufOut[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len128_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6, T *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*2 + 0 + 0 ) ];
	(*R4) = bufIn[inOffset + ( 0 + me*2 + 1 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*2 + 0 + 32 ) ];
	(*R5) = bufIn[inOffset + ( 0 + me*2 + 1 + 32 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*2 + 0 + 64 ) ];
	(*R6) = bufIn[inOffset + ( 0 + me*2 + 1 + 64 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*2 + 0 + 96 ) ];
	(*R7) = bufIn[inOffset + ( 0 + me*2 + 1 + 96 ) ];
	}



	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		T W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( 2*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 2*me + 1 + 0 ) ] = (*R4);
	bufOut[outOffset + ( 2*me + 0 + 32 ) ] = (*R1);
	bufOut[outOffset + ( 2*me + 1 + 32 ) ] = (*R5);
	bufOut[outOffset + ( 2*me + 0 + 64 ) ] = (*R2);
	bufOut[outOffset + ( 2*me + 1 + 64 ) ] = (*R6);
	bufOut[outOffset + ( 2*me + 0 + 96 ) ] = (*R3);
	bufOut[outOffset + ( 2*me + 1 + 96 ) ] = (*R7);
	}

}

template <typename T, StrideBin sb> 
__device__ void 
fwd_len128_BCT_R2C_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7;
	FwdPass0_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	FwdPass1_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	FwdPass2_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
}

template <typename T, StrideBin sb> 
__device__ void 
back_len128_BCT_R2C_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3, R4, R5, R6, R7;
	InvPass0_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	InvPass1_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
	InvPass2_len128_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	__syncthreads();
}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len128_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[me + t*128];
		lds[t*128 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len128_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len128_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[me + t*128];
		lds[t*128 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len128_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len128_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[me + t*128];
		lds[t*128 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len128_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*16] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 128 transforms: 4 Passes: 3
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len128_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[1024];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[me + t*128];
		lds[t*128 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<1; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len128_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  rw, b, me%16, t*1024 + (me/16)*128, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*16 + (me%8)*128 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*16] = R0;
	}

}

#include "kernels/common.h"
#include "rocfft_butterfly_template.h"

template <typename T, StrideBin sb> 
__device__ void
FwdPass0_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass1_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass2_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 16 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 32 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
FwdPass3_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 64 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 128 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 192 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass0_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 1 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 2 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/1)*4 + (1*me + 0)%1 + 3 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass1_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[3 + 3*((1*me + 0)%4) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 4 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 8 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/4)*16 + (1*me + 0)%4 + 12 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass2_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[15 + 3*((1*me + 0)%16) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 0 ) ] = (*R0);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 16 ) ] = (*R1);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 32 ) ] = (*R2);
	bufOut[outOffset + ( ((1*me + 0)/16)*64 + (1*me + 0)%16 + 48 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void
InvPass3_len256_BCT_R2C(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int inOffset, unsigned int outOffset, T *bufIn, T *bufOut, T *R0, T *R1, T *R2, T *R3)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 ) ];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 ) ];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 128 ) ];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 192 ) ];
	}



	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 0];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 1];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		T W = twiddles[63 + 3*((1*me + 0)%64) + 2];
		real_type_t<T> TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);

	__syncthreads();



	if(rw)
	{
	bufOut[outOffset + ( 1*me + 0 + 0 ) ] = (*R0);
	bufOut[outOffset + ( 1*me + 0 + 64 ) ] = (*R1);
	bufOut[outOffset + ( 1*me + 0 + 128 ) ] = (*R2);
	bufOut[outOffset + ( 1*me + 0 + 192 ) ] = (*R3);
	}

}

template <typename T, StrideBin sb> 
__device__ void 
fwd_len256_BCT_R2C_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	FwdPass0_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass1_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass2_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	FwdPass3_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

template <typename T, StrideBin sb> 
__device__ void 
back_len256_BCT_R2C_device(const T *twiddles, const size_t stride_in, const size_t stride_out, unsigned int rw, unsigned int b, unsigned int me, unsigned int ldsOffset, T *lwbIn, T *lwbOut, T *lds)
{
	T R0, R1, R2, R3;
	InvPass0_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lwbIn, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass1_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass2_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lds, &R0, &R1, &R2, &R3);
	__syncthreads();
	InvPass3_len256_BCT_R2C<T, sb>(twiddles, stride_in, stride_out, rw, b, me, ldsOffset, ldsOffset, lds, lwbOut, &R0, &R1, &R2, &R3);
	__syncthreads();
}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_ip_len256_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len256_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_ip_len256_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t batch_count, T * __restrict__   gb)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int ioOffset = 0;
	T *lwb;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		ioOffset += (counter_mod/currentLength)*stride_in[i];
		counter_mod = (counter_mod % currentLength); 
	}
	ioOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	lwb = gb + ioOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwb[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len256_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_in[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwb[(me%8) + (me/8)*stride_in[0] + t*stride_in[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_fwd_op_len256_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		fwd_len256_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*32] = R0;
	}

}

//Kernel configuration: number of threads per thread block: 256 transforms: 1 Passes: 4
template <typename T, StrideBin sb>
__global__ void 
fft_back_op_len256_BCT_R2C( hipLaunchParm lp, const T * __restrict__ twiddles, const size_t dim, const size_t *lengths, const size_t *stride_in, const size_t *stride_out, const size_t batch_count, T * __restrict__   gbIn, T * __restrict__   gbOut)
{
	unsigned int me = (unsigned int)hipThreadIdx_x;
	unsigned int batch = (unsigned int)hipBlockIdx_x;

	__shared__ T lds[2048];

	unsigned int iOffset = 0;
	unsigned int oOffset = 0;
	T *lwbIn;
	T *lwbOut;

	unsigned int rw = 1;

	//suppress warning
	#ifdef __NVCC__
		(void)(rw == rw);
	#else
		(void)rw;
	#endif
	unsigned int b = 0;

	size_t counter_mod = batch;
	for(int i = dim; i>2; i--){
		int currentLength = 1;
		for(int j=2; j<i; j++){
			currentLength *= lengths[j];
		}
		currentLength *= (lengths[1]/8);

		iOffset += (counter_mod/currentLength)*stride_in[i];
		oOffset += (counter_mod/currentLength)*stride_out[i];
		counter_mod = (counter_mod % currentLength); 
	}
	iOffset += (counter_mod/(lengths[1]/8))*stride_in[2] + (counter_mod % (lengths[1]/8))*8*lengths[0];
	iOffset += (counter_mod/(lengths[1]/8))*stride_out[2] + (counter_mod % (lengths[1]/8))*8;
	lwbIn = gbIn + iOffset;
	lwbOut = gbOut + oOffset;


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lwbIn[me + t*256];
		lds[t*256 + me] = R0;
	}

	__syncthreads();


	for(unsigned int t=0; t<2; t++)
	{

		// Perform FFT input: lwb(In) ; output: lwb(Out); working space: lds 
		// rw, b, me% control read/write; then ldsOffset, lwb, lds
		back_len256_BCT_R2C_device<T, sb>(twiddles, stride_in[0], stride_out[0],  1, b, me%64, t*1024 + (me/64)*256, lds, lds, lds);

	}

	__syncthreads();


	for(unsigned int t=0; t<8; t++)
	{
		T R0 = lds[t*32 + (me%8)*256 + (me/8)];
		lwbOut[(me%8) + (me/8)*stride_out[0] + t*stride_out[0]*32] = R0;
	}

}

