

#define fptype float

#define fvect2 float2



__attribute__((always_inline)) void
FwdPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*1];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 )*1];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 )*1];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 )*1];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 )*1];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*1];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 )*1];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 )*1];
	}



	FwdRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
FwdPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{




	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
FwdPass2(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{




	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		float TR, TI;
		TR = (W.x * (*R5).x) - (W.y * (*R5).y);
		TI = (W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		float TR, TI;
		TR = (W.x * (*R6).x) - (W.y * (*R6).y);
		TI = (W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		float TR, TI;
		TR = (W.x * (*R7).x) - (W.y * (*R7).y);
		TI = (W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);
	FwdRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	__global float4 *buff4g = bufOut;
	
	buff4g[ 1*me + 0 + 0 ] = (float4)((*R0).x, (*R0).y, (*R4).x, (*R4).y) ;
	buff4g[ 1*me + 0 + 16 ] = (float4)((*R1).x, (*R1).y, (*R5).x, (*R5).y) ;
	buff4g[ 1*me + 0 + 32 ] = (float4)((*R2).x, (*R2).y, (*R6).x, (*R6).y) ;
	buff4g[ 1*me + 0 + 48 ] = (float4)((*R3).x, (*R3).y, (*R7).x, (*R7).y) ;
	}

}

__attribute__((always_inline)) void
InvPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{


	if(rw)
	{
	(*R0) = bufIn[inOffset + ( 0 + me*1 + 0 + 0 )*1];
	(*R1) = bufIn[inOffset + ( 0 + me*1 + 0 + 16 )*1];
	(*R2) = bufIn[inOffset + ( 0 + me*1 + 0 + 32 )*1];
	(*R3) = bufIn[inOffset + ( 0 + me*1 + 0 + 48 )*1];
	(*R4) = bufIn[inOffset + ( 0 + me*1 + 0 + 64 )*1];
	(*R5) = bufIn[inOffset + ( 0 + me*1 + 0 + 80 )*1];
	(*R6) = bufIn[inOffset + ( 0 + me*1 + 0 + 96 )*1];
	(*R7) = bufIn[inOffset + ( 0 + me*1 + 0 + 112 )*1];
	}



	InvRad8B1(R0, R1, R2, R3, R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 1 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 2 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 3 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 4 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 5 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 6 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((1*me + 0)/1)*8 + (1*me + 0)%1 + 7 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
InvPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{




	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 0)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[7 + 3*((2*me + 1)%8) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).x;
	bufOutRe[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).x;
	bufOutRe[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).x;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).x = bufOutRe[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).x = bufOutRe[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 0 )*1] = (*R0).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 8 )*1] = (*R1).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 16 )*1] = (*R2).y;
	bufOutIm[outOffset + ( ((2*me + 0)/8)*32 + (2*me + 0)%8 + 24 )*1] = (*R3).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 0 )*1] = (*R4).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 8 )*1] = (*R5).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 16 )*1] = (*R6).y;
	bufOutIm[outOffset + ( ((2*me + 1)/8)*32 + (2*me + 1)%8 + 24 )*1] = (*R7).y;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	if(rw)
	{
	(*R0).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 0 )*1];
	(*R4).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 0 )*1];
	(*R1).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 32 )*1];
	(*R5).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 32 )*1];
	(*R2).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 64 )*1];
	(*R6).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 64 )*1];
	(*R3).y = bufOutIm[outOffset + ( 0 + me*2 + 0 + 96 )*1];
	(*R7).y = bufOutIm[outOffset + ( 0 + me*2 + 1 + 96 )*1];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
InvPass2(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3, float2 *R4, float2 *R5, float2 *R6, float2 *R7)
{




	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 0];
		float TR, TI;
		TR =  (W.x * (*R1).x) + (W.y * (*R1).y);
		TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 1];
		float TR, TI;
		TR =  (W.x * (*R2).x) + (W.y * (*R2).y);
		TI = -(W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 0)%32) + 2];
		float TR, TI;
		TR =  (W.x * (*R3).x) + (W.y * (*R3).y);
		TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 0];
		float TR, TI;
		TR =  (W.x * (*R5).x) + (W.y * (*R5).y);
		TI = -(W.y * (*R5).x) + (W.x * (*R5).y);
		(*R5).x = TR;
		(*R5).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 1];
		float TR, TI;
		TR =  (W.x * (*R6).x) + (W.y * (*R6).y);
		TI = -(W.y * (*R6).x) + (W.x * (*R6).y);
		(*R6).x = TR;
		(*R6).y = TI;
	}

	{
		float2 W = twiddles[31 + 3*((2*me + 1)%32) + 2];
		float TR, TI;
		TR =  (W.x * (*R7).x) + (W.y * (*R7).y);
		TI = -(W.y * (*R7).x) + (W.x * (*R7).y);
		(*R7).x = TR;
		(*R7).y = TI;
	}

	InvRad4B1(R0, R1, R2, R3);
	InvRad4B1(R4, R5, R6, R7);


	if(rw)
	{
	__global float4 *buff4g = bufOut;
	
	buff4g[ 1*me + 0 + 0 ] = (float4)((*R0).x, (*R0).y, (*R4).x, (*R4).y)  * 7.8125000000000000e-03f;
	buff4g[ 1*me + 0 + 16 ] = (float4)((*R1).x, (*R1).y, (*R5).x, (*R5).y)  * 7.8125000000000000e-03f;
	buff4g[ 1*me + 0 + 32 ] = (float4)((*R2).x, (*R2).y, (*R6).x, (*R6).y)  * 7.8125000000000000e-03f;
	buff4g[ 1*me + 0 + 48 ] = (float4)((*R3).x, (*R3).y, (*R7).x, (*R7).y)  * 7.8125000000000000e-03f;
	}

}

 typedef union  { uint u; int i; } cb_t;

__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_fwd(__constant cb_t *cb __attribute__((max_constant_size(32))), __global float2 * restrict gb)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];

	uint ioOffset;
	__global float2 *lwb;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	uint rw = (me < ((cb[0].u) - batch*4)*16) ? 1 : 0;

	uint b = 0;

	ioOffset = (batch*4 + (me/16))*128;
	lwb = gb + ioOffset;

	FwdPass0(rw, b, me%16, 0, (me/16)*128, lwb, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass1(rw, b, me%16, (me/16)*128, (me/16)*128, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	FwdPass2(rw, b, me%16, (me/16)*128, 0, lds, lds, lwb, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}

__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_back(__constant cb_t *cb __attribute__((max_constant_size(32))), __global float2 * restrict gb)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];

	uint ioOffset;
	__global float2 *lwb;

	float2 R0, R1, R2, R3, R4, R5, R6, R7;

	uint rw = (me < ((cb[0].u) - batch*4)*16) ? 1 : 0;

	uint b = 0;

	ioOffset = (batch*4 + (me/16))*128;
	lwb = gb + ioOffset;

	InvPass0(rw, b, me%16, 0, (me/16)*128, lwb, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass1(rw, b, me%16, (me/16)*128, (me/16)*128, lds, lds, lds, lds, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
	InvPass2(rw, b, me%16, (me/16)*128, 0, lds, lds, lwb, &R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7);
}


