
#define TWIDDLE_MUL_FWD(INDEX, REG) \
{ \
	float2 W = twiddles[INDEX]; \
	float TR, TI; \
	TR = (W.x * REG.x) - (W.y * REG.y); \
	TI = (W.y * REG.x) + (W.x * REG.y); \
	REG.x = TR; \
	REG.y = TI; \
}

#define TWIDDLE_MUL_INV(INDEX, REG) \
{ \
	float2 W = twiddles[INDEX]; \
	float TR, TI; \
	TR =  (W.x * REG.x) + (W.y * REG.y); \
	TI = -(W.y * REG.x) + (W.x * REG.y); \
	REG.x = TR; \
	REG.y = TI; \
}


__device__ void 
Rad2(float2 *R0, float2 *R1)
{

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	
}

__device__ void 
FwdRad4(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

__device__ void 
InvRad4(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	
	T = (*R1); (*R1) = (*R2); (*R2) = T;
	
}

#define C8Q  0.707106781186547573f

__device__ void 
FwdRad8(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + float2(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * float2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + float2(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * float2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

__device__ void 
InvRad8(float2 *R0, float2 *R4, float2 *R2, float2 *R6, float2 *R1, float2 *R5, float2 *R3, float2 *R7)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + float2((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * float2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + float2((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * float2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	
	T = (*R1); (*R1) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R6); (*R6) = T;
	
}

#define C16A 0.923879532511286738f
#define C16B 0.382683432365089837f

__device__ void 
FwdRad16(float2 *R0, float2 *R8, float2 *R4, float2 *R12, float2 *R2, float2 *R10, float2 *R6, float2 *R14, float2 *R1, float2 *R9, float2 *R5, float2 *R13, float2 *R3, float2 *R11, float2 *R7, float2 *R15)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	(*R9) = (*R8) - (*R9);
	(*R8) = 2.0f * (*R8) - (*R9);
	(*R11) = (*R10) - (*R11);
	(*R10) = 2.0f * (*R10) - (*R11);
	(*R13) = (*R12) - (*R13);
	(*R12) = 2.0f * (*R12) - (*R13);
	(*R15) = (*R14) - (*R15);
	(*R14) = 2.0f * (*R14) - (*R15);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + float2(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	(*R10) = (*R8) - (*R10);
	(*R8) = 2.0f * (*R8) - (*R10);
	(*R11) = (*R9) + float2(-(*R11).y, (*R11).x);
	(*R9) = 2.0f * (*R9) - (*R11);
	(*R14) = (*R12) - (*R14);
	(*R12) = 2.0f * (*R12) - (*R14);
	(*R15) = (*R13) + float2(-(*R15).y, (*R15).x);
	(*R13) = 2.0f * (*R13) - (*R15);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * float2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + float2(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * float2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	(*R12) = (*R8) - (*R12);
	(*R8) = 2.0f * (*R8) - (*R12);
	(*R13) = ((*R9) - C8Q * (*R13)) - C8Q * float2((*R13).y, -(*R13).x);
	(*R9) = 2.0f * (*R9) - (*R13);
	(*R14) = (*R10) + float2(-(*R14).y, (*R14).x);
	(*R10) = 2.0f * (*R10) - (*R14);
	(*R15) = ((*R11) + C8Q * (*R15)) - C8Q * float2((*R15).y, -(*R15).x);
	(*R11) = 2.0f * (*R11) - (*R15);
	
	(*R8) = (*R0) - (*R8);
	(*R0) = 2.0f * (*R0) - (*R8);
	(*R9) = ((*R1) - C16A * (*R9)) - C16B * float2((*R9).y, -(*R9).x);
	T = (*R8);
	(*R1) = 2.0f * (*R1) - (*R9);
	
	(*R10) = ((*R2) - C8Q * (*R10)) - C8Q * float2((*R10).y, -(*R10).x);
	(*R2) = 2.0f * (*R2) - (*R10);
	(*R11) = ((*R3) - C16B * (*R11)) - C16A * float2((*R11).y, -(*R11).x);
	(*R3) = 2.0f * (*R3) - (*R11);
	
	(*R12) = (*R4) + float2(-(*R12).y, (*R12).x);
	(*R4) = 2.0f * (*R4) - (*R12);
	(*R13) = ((*R5) + C16B * (*R13)) - C16A * float2((*R13).y, -(*R13).x);
	(*R5) = 2.0f * (*R5) - (*R13);
	
	(*R14) = ((*R6) + C8Q * (*R14)) - C8Q * float2((*R14).y, -(*R14).x);
	(*R6) = 2.0f * (*R6) - (*R14);
	(*R15) = ((*R7) + C16A * (*R15)) - C16B * float2((*R15).y, -(*R15).x);
	(*R7) = 2.0f * (*R7) - (*R15);
	
	T = (*R1); (*R1) = (*R8); (*R8) = T;
	T = (*R2); (*R2) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R12); (*R12) = T;
	T = (*R5); (*R5) = (*R10); (*R10) = T;
	T = (*R7); (*R7) = (*R14); (*R14) = T;
	T = (*R11); (*R11) = (*R13); (*R13) = T;
	
}

__device__ void 
InvRad16(float2 *R0, float2 *R8, float2 *R4, float2 *R12, float2 *R2, float2 *R10, float2 *R6, float2 *R14, float2 *R1, float2 *R9, float2 *R5, float2 *R13, float2 *R3, float2 *R11, float2 *R7, float2 *R15)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);
	(*R5) = (*R4) - (*R5);
	(*R4) = 2.0f * (*R4) - (*R5);
	(*R7) = (*R6) - (*R7);
	(*R6) = 2.0f * (*R6) - (*R7);
	(*R9) = (*R8) - (*R9);
	(*R8) = 2.0f * (*R8) - (*R9);
	(*R11) = (*R10) - (*R11);
	(*R10) = 2.0f * (*R10) - (*R11);
	(*R13) = (*R12) - (*R13);
	(*R12) = 2.0f * (*R12) - (*R13);
	(*R15) = (*R14) - (*R15);
	(*R14) = 2.0f * (*R14) - (*R15);
	
	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + float2((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	(*R10) = (*R8) - (*R10);
	(*R8) = 2.0f * (*R8) - (*R10);
	(*R11) = (*R9) + float2((*R11).y, -(*R11).x);
	(*R9) = 2.0f * (*R9) - (*R11);
	(*R14) = (*R12) - (*R14);
	(*R12) = 2.0f * (*R12) - (*R14);
	(*R15) = (*R13) + float2((*R15).y, -(*R15).x);
	(*R13) = 2.0f * (*R13) - (*R15);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * float2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + float2((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * float2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	(*R12) = (*R8) - (*R12);
	(*R8) = 2.0f * (*R8) - (*R12);
	(*R13) = ((*R9) - C8Q * (*R13)) + C8Q * float2((*R13).y, -(*R13).x);
	(*R9) = 2.0f * (*R9) - (*R13);
	(*R14) = (*R10) + float2((*R14).y, -(*R14).x);
	(*R10) = 2.0f * (*R10) - (*R14);
	(*R15) = ((*R11) + C8Q * (*R15)) + C8Q * float2((*R15).y, -(*R15).x);
	(*R11) = 2.0f * (*R11) - (*R15);
	
	(*R8) = (*R0) - (*R8);
	(*R0) = 2.0f * (*R0) - (*R8);
	(*R9) = ((*R1) - C16A * (*R9)) + C16B * float2((*R9).y, -(*R9).x);
	(*R1) = 2.0f * (*R1) - (*R9);
	(*R10) = ((*R2) - C8Q * (*R10)) + C8Q * float2((*R10).y, -(*R10).x);
	(*R2) = 2.0f * (*R2) - (*R10);
	(*R11) = ((*R3) - C16B * (*R11)) + C16A * float2((*R11).y, -(*R11).x);
	(*R3) = 2.0f * (*R3) - (*R11);
	(*R12) = (*R4) + float2((*R12).y, -(*R12).x);
	(*R4) = 2.0f * (*R4) - (*R12);
	(*R13) = ((*R5) + C16B * (*R13)) + C16A * float2((*R13).y, -(*R13).x);
	(*R5) = 2.0f * (*R5) - (*R13);
	(*R14) = ((*R6) + C8Q * (*R14)) + C8Q * float2((*R14).y, -(*R14).x);
	(*R6) = 2.0f * (*R6) - (*R14);
	(*R15) = ((*R7) + C16A * (*R15)) + C16B * float2((*R15).y, -(*R15).x);
	(*R7) = 2.0f * (*R7) - (*R15);
	
	T = (*R1); (*R1) = (*R8); (*R8) = T;
	T = (*R2); (*R2) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R12); (*R12) = T;
	T = (*R5); (*R5) = (*R10); (*R10) = T;
	T = (*R7); (*R7) = (*R14); (*R14) = T;
	T = (*R11); (*R11) = (*R13); (*R13) = T;
	
}

__global__
void fft_1(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*2;
	

	if(rw)
	{
		X0 = lwb[0];
		lwb[0] = X0;
	}
}


__global__
void fft_2(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0, X1;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	float2 *lwb = buffer + (batch*64 + me)*2;
	

	if(rw)
	{
		X0 = lwb[0];
		X1 = lwb[1];	
	}
	
	Rad2(&X0, &X1);
	
	if(rw)
	{
		lwb[0] = X0;
		lwb[1] = X1;	
	}
}	



__global__
void fft_4(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[128];
	float2 X0, X1;

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*4;
	float *ldsp = lds + (me/2)*4;
	
	me = me % 2;

	if(rw)
	{	
		X0 = lwb[me + 0];
		X1 = lwb[me + 2];	
	}
	
	Rad2(&X0, &X1);

	ldsp[me*2 + 0] = X0.x;
	ldsp[me*2 + 1] = X1.x;
	
	__syncthreads();
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 2];
	
	__syncthreads();
	
	ldsp[me*2 + 0] = X0.y;
	ldsp[me*2 + 1] = X1.y;
	
	__syncthreads();
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 2];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(1 + (me%2), X1)
	}
	else
	{
		TWIDDLE_MUL_INV(1 + (me%2), X1)
	}
	
	Rad2(&X0, &X1);
	
	if(rw)
	{	
		lwb[me + 0] = X0;
		lwb[me + 2] = X1;	
	}
}



__global__
void fft_8(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	float2 *lwb = buffer + (batch*32 + (me/2))*8;
	float *ldsp = lds + (me/2)*8;
	
	me = me % 2;

	if(rw)
	{	
		X0 = lwb[me + 0];
		X1 = lwb[me + 2];	
		X2 = lwb[me + 4];
		X3 = lwb[me + 6];		
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	ldsp[me*4 + 0] = X0.x;
	ldsp[me*4 + 1] = X1.x;
	ldsp[me*4 + 2] = X2.x;
	ldsp[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = ldsp[me*2 + 0 + 0];
	X1.x = ldsp[me*2 + 0 + 4];	
	X2.x = ldsp[me*2 + 1 + 0];
	X3.x = ldsp[me*2 + 1 + 4];
	
	__syncthreads();
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = ldsp[me*2 + 0 + 0];
	X1.y = ldsp[me*2 + 0 + 4];	
	X2.y = ldsp[me*2 + 1 + 0];
	X3.y = ldsp[me*2 + 1 + 4];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_FWD(3 + ((2*me+1)%4), X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_INV(3 + ((2*me+1)%4), X3)			
	}
	
	Rad2(&X0, &X1);
	Rad2(&X2, &X3);
	
	if(rw)
	{
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me + 0] = float4(X0.x,X0.y,X2.x,X2.y);
		lwbv[me + 2] = float4(X1.x,X1.y,X3.x,X3.y);	
	}
}



__global__
void fft_16(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*16;
	float *ldsp = lds + (me/4)*16;
	
	me = me % 4;

	if(rw)
	{	
		X0 = lwb[me + 0];
		X1 = lwb[me + 4];	
		X2 = lwb[me + 8];
		X3 = lwb[me + 12];		
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	ldsp[me*4 + 0] = X0.x;
	ldsp[me*4 + 1] = X1.x;
	ldsp[me*4 + 2] = X2.x;
	ldsp[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 4];	
	X2.x = ldsp[me + 8];
	X3.x = ldsp[me + 12];
	
	__syncthreads();
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 4];	
	X2.y = ldsp[me + 8];
	X3.y = ldsp[me + 12];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 2, X3)
	}
	else
	{
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 2, X3)	
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	if(rw)
	{	
		lwb[me + 0]  = X0;
		lwb[me + 4]  = X1;	
		lwb[me + 8]  = X2;
		lwb[me + 12] = X3;		
	}
}



__global__
void fft_32(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	float2 *lwb = buffer + (batch*16 + (me/4))*32;
	float *ldsp = lds + (me/4)*32;
	
	me = me % 4;

	if(rw)
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
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	ldsp[me*8 + 0] = X0.x;
	ldsp[me*8 + 1] = X1.x;
	ldsp[me*8 + 2] = X2.x;
	ldsp[me*8 + 3] = X3.x;
	ldsp[me*8 + 4] = X4.x;
	ldsp[me*8 + 5] = X5.x;
	ldsp[me*8 + 6] = X6.x;
	ldsp[me*8 + 7] = X7.x;
				
	__syncthreads();

	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) +  8];
	X2.x = ldsp[(2*me + 0) + 16];
	X3.x = ldsp[(2*me + 0) + 24];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) +  8];
	X6.x = ldsp[(2*me + 1) + 16];
	X7.x = ldsp[(2*me + 1) + 24];	

	__syncthreads();
				

	ldsp[me*8 + 0] = X0.y;
	ldsp[me*8 + 1] = X1.y;
	ldsp[me*8 + 2] = X2.y;
	ldsp[me*8 + 3] = X3.y;
	ldsp[me*8 + 4] = X4.y;
	ldsp[me*8 + 5] = X5.y;
	ldsp[me*8 + 6] = X6.y;
	ldsp[me*8 + 7] = X7.y;
				
	__syncthreads();

	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) +  8];
	X2.y = ldsp[(2*me + 0) + 16];
	X3.y = ldsp[(2*me + 0) + 24];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) +  8];
	X6.y = ldsp[(2*me + 1) + 16];
	X7.y = ldsp[(2*me + 1) + 24];	

	__syncthreads();
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 2, X7)		
	}
	else
	{
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 2, X7)	
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
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +  0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me +  4] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me +  8] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 12] = float4(X3.x,X3.y,X7.x,X7.y);			
	}			
	
}


__global__
void fft_64(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*64;
	float *ldsp = lds + (me/16)*64;
	
	me = me % 16;

	if(rw)
	{	
		X0 = lwb[me + 0];
		X1 = lwb[me + 16];	
		X2 = lwb[me + 32];
		X3 = lwb[me + 48];		
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	ldsp[me*4 + 0] = X0.x;
	ldsp[me*4 + 1] = X1.x;
	ldsp[me*4 + 2] = X2.x;
	ldsp[me*4 + 3] = X3.x;
	
	__syncthreads();
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 16];	
	X2.x = ldsp[me + 32];
	X3.x = ldsp[me + 48];
	
	__syncthreads();
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	__syncthreads();
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 16];	
	X2.y = ldsp[me + 32];
	X3.y = ldsp[me + 48];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
	
	
	ldsp[(me/4)*16 + me%4 +  0] = X0.x;
	ldsp[(me/4)*16 + me%4 +  4] = X1.x;
	ldsp[(me/4)*16 + me%4 +  8] = X2.x;
	ldsp[(me/4)*16 + me%4 + 12] = X3.x;
	
	__syncthreads();
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 16];	
	X2.x = ldsp[me + 32];
	X3.x = ldsp[me + 48];
	
	__syncthreads();
	
	ldsp[(me/4)*16 + me%4 +  0] = X0.y;
	ldsp[(me/4)*16 + me%4 +  4] = X1.y;
	ldsp[(me/4)*16 + me%4 +  8] = X2.y;
	ldsp[(me/4)*16 + me%4 + 12] = X3.y;
	
	__syncthreads();
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 16];	
	X2.y = ldsp[me + 32];
	X3.y = ldsp[me + 48];
	
	__syncthreads();

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 2, X3)				
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);

		
	if(rw)
	{	
		lwb[me + 0]  = X0;
		lwb[me + 16] = X1;	
		lwb[me + 32] = X2;
		lwb[me + 48] = X3;		
	}
}


__global__
void fft_128(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[512];
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	float2 *lwb = buffer + (batch*4 + (me/16))*128;
	float *ldsp = lds + (me/16)*128;
	
	me = me % 16;
	
	if(rw)
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
	
	if(dir == -1)
		FwdRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);
	else
		InvRad8(&X0, &X1, &X2, &X3, &X4, &X5, &X6, &X7);


	ldsp[me*8 + 0] = X0.x;
	ldsp[me*8 + 1] = X1.x;
	ldsp[me*8 + 2] = X2.x;
	ldsp[me*8 + 3] = X3.x;
	ldsp[me*8 + 4] = X4.x;
	ldsp[me*8 + 5] = X5.x;
	ldsp[me*8 + 6] = X6.x;
	ldsp[me*8 + 7] = X7.x;
	
	__syncthreads();
	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) + 32];
	X2.x = ldsp[(2*me + 0) + 64];
	X3.x = ldsp[(2*me + 0) + 96];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) + 32];
	X6.x = ldsp[(2*me + 1) + 64];
	X7.x = ldsp[(2*me + 1) + 96];	

	__syncthreads();

	ldsp[me*8 + 0] = X0.y;
	ldsp[me*8 + 1] = X1.y;
	ldsp[me*8 + 2] = X2.y;
	ldsp[me*8 + 3] = X3.y;
	ldsp[me*8 + 4] = X4.y;
	ldsp[me*8 + 5] = X5.y;
	ldsp[me*8 + 6] = X6.y;
	ldsp[me*8 + 7] = X7.y;
	
	__syncthreads();
	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) + 32];
	X2.y = ldsp[(2*me + 0) + 64];
	X3.y = ldsp[(2*me + 0) + 96];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) + 32];
	X6.y = ldsp[(2*me + 1) + 64];
	X7.y = ldsp[(2*me + 1) + 96];	

	__syncthreads();	

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(7 + 3*((2*me + 1)%8) + 2, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 0)%8) + 2, X3)	

		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(7 + 3*((2*me + 1)%8) + 2, X7)		
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
	
	
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.x;
	
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.x;
	
	__syncthreads();
	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) + 32];
	X2.x = ldsp[(2*me + 0) + 64];
	X3.x = ldsp[(2*me + 0) + 96];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) + 32];
	X6.x = ldsp[(2*me + 1) + 64];
	X7.x = ldsp[(2*me + 1) + 96];	

	__syncthreads();

	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.y;
	
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.y;
	
	__syncthreads();
	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) + 32];
	X2.y = ldsp[(2*me + 0) + 64];
	X3.y = ldsp[(2*me + 0) + 96];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) + 32];
	X6.y = ldsp[(2*me + 1) + 64];
	X7.y = ldsp[(2*me + 1) + 96];	

	__syncthreads();	
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_FWD(31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_FWD(31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_FWD(31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_FWD(31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_FWD(31 + 3*((2*me + 1)%32) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(31 + 3*((2*me + 0)%32) + 0, X1)
		TWIDDLE_MUL_INV(31 + 3*((2*me + 0)%32) + 1, X2)
		TWIDDLE_MUL_INV(31 + 3*((2*me + 0)%32) + 2, X3)	

		TWIDDLE_MUL_INV(31 + 3*((2*me + 1)%32) + 0, X5)
		TWIDDLE_MUL_INV(31 + 3*((2*me + 1)%32) + 1, X6)
		TWIDDLE_MUL_INV(31 + 3*((2*me + 1)%32) + 2, X7)
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
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +  0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 16] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 32] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 48] = float4(X3.x,X3.y,X7.x,X7.y);			
	}	
	
}


__global__
void fft_256(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	__shared__ float lds[256];
	float2 X0, X1, X2, X3;

	float2 *lwb = buffer + batch*256;
	

	X0 = lwb[me +   0];
	X1 = lwb[me +  64];	
	X2 = lwb[me + 128];
	X3 = lwb[me + 192];		
	

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
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_FWD(3 + 3*(me%4) + 2, X3)			
	}
	else
	{
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 0, X1)	
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 1, X2)	
		TWIDDLE_MUL_INV(3 + 3*(me%4) + 2, X3)		
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
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_FWD(15 + 3*(me%16) + 2, X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 0, X1)	
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 1, X2)	
		TWIDDLE_MUL_INV(15 + 3*(me%16) + 2, X3)			
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
		TWIDDLE_MUL_FWD(63 + 3*me + 0, X1)	
		TWIDDLE_MUL_FWD(63 + 3*me + 1, X2)	
		TWIDDLE_MUL_FWD(63 + 3*me + 2, X3)		
	}
	else
	{
		TWIDDLE_MUL_INV(63 + 3*me + 0, X1)	
		TWIDDLE_MUL_INV(63 + 3*me + 1, X2)	
		TWIDDLE_MUL_INV(63 + 3*me + 2, X3)			
	}
	
	if(dir == -1)
		FwdRad4(&X0, &X1, &X2, &X3);
	else
		InvRad4(&X0, &X1, &X2, &X3);
		

	lwb[me +   0] = X0;
	lwb[me +  64] = X1;	
	lwb[me + 128] = X2;
	lwb[me + 192] = X3;		

}


__global__
void fft_512(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__shared__ float lds[512];
	float2 *lwb = buffer + batch*512;
	

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
		
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 6, X7)			
	}
	else
	{
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 6, X7)
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
		TWIDDLE_MUL_FWD(63 + 7*me + 0, X1)			
		TWIDDLE_MUL_FWD(63 + 7*me + 1, X2)	
		TWIDDLE_MUL_FWD(63 + 7*me + 2, X3)	
		TWIDDLE_MUL_FWD(63 + 7*me + 3, X4)	
		TWIDDLE_MUL_FWD(63 + 7*me + 4, X5)	
		TWIDDLE_MUL_FWD(63 + 7*me + 5, X6)	
		TWIDDLE_MUL_FWD(63 + 7*me + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(63 + 7*me + 0, X1)			
		TWIDDLE_MUL_INV(63 + 7*me + 1, X2)	
		TWIDDLE_MUL_INV(63 + 7*me + 2, X3)	
		TWIDDLE_MUL_INV(63 + 7*me + 3, X4)	
		TWIDDLE_MUL_INV(63 + 7*me + 4, X5)	
		TWIDDLE_MUL_INV(63 + 7*me + 5, X6)	
		TWIDDLE_MUL_INV(63 + 7*me + 6, X7)
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
	

__global__
void fft_1024(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__shared__ float lds[1024];
	float2 *lwb = buffer + batch*1024;
	
	X0 = lwb[me +   0];
	X1 = lwb[me + 128];
	X2 = lwb[me + 256];
	X3 = lwb[me + 384];
	X4 = lwb[me + 512];
	X5 = lwb[me + 640];
	X6 = lwb[me + 768];
	X7 = lwb[me + 896];
					
	
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
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 0, X1)			
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 1, X2)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 2, X3)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 3, X4)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 4, X5)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 5, X6)	
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 6, X7)
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
		TWIDDLE_MUL_FWD(63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_FWD(63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_FWD(63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_FWD(63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_FWD(63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_FWD(63 + 3*((2*me + 1)%64) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(63 + 3*((2*me + 0)%64) + 0, X1)
		TWIDDLE_MUL_INV(63 + 3*((2*me + 0)%64) + 1, X2)
		TWIDDLE_MUL_INV(63 + 3*((2*me + 0)%64) + 2, X3)	

		TWIDDLE_MUL_INV(63 + 3*((2*me + 1)%64) + 0, X5)
		TWIDDLE_MUL_INV(63 + 3*((2*me + 1)%64) + 1, X6)
		TWIDDLE_MUL_INV(63 + 3*((2*me + 1)%64) + 2, X7)	
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
		TWIDDLE_MUL_FWD(255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_FWD(255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_FWD(255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_FWD(255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_FWD(255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_FWD(255 + 3*((2*me + 1)%256) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(255 + 3*((2*me + 0)%256) + 0, X1)
		TWIDDLE_MUL_INV(255 + 3*((2*me + 0)%256) + 1, X2)
		TWIDDLE_MUL_INV(255 + 3*((2*me + 0)%256) + 2, X3)	

		TWIDDLE_MUL_INV(255 + 3*((2*me + 1)%256) + 0, X5)
		TWIDDLE_MUL_INV(255 + 3*((2*me + 1)%256) + 1, X6)
		TWIDDLE_MUL_INV(255 + 3*((2*me + 1)%256) + 2, X7)
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
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +   0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 128] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 256] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 384] = float4(X3.x,X3.y,X7.x,X7.y);			
	}	
}


	
__global__
void fft_2048(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__shared__ float lds[2048];
	float2 *lwb = buffer + batch*2048;
	

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
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_FWD(7 + 7*(me%8) + 6, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 0, X1)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 1, X2)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 2, X3)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 3, X4)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 4, X5)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 5, X6)
		TWIDDLE_MUL_INV(7 + 7*(me%8) + 6, X7)				

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
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_FWD(63 + 7*(me%64) + 6, X7)			

	}
	else
	{
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 0, X1)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 1, X2)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 2, X3)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 3, X4)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 4, X5)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 5, X6)
		TWIDDLE_MUL_INV(63 + 7*(me%64) + 6, X7)
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
		TWIDDLE_MUL_FWD(511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_FWD(511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_FWD(511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_FWD(511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_FWD(511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_FWD(511 + 3*((2*me + 1)%512) + 2, X7)
	}
	else
	{
		TWIDDLE_MUL_INV(511 + 3*((2*me + 0)%512) + 0, X1)
		TWIDDLE_MUL_INV(511 + 3*((2*me + 0)%512) + 1, X2)
		TWIDDLE_MUL_INV(511 + 3*((2*me + 0)%512) + 2, X3)

		TWIDDLE_MUL_INV(511 + 3*((2*me + 1)%512) + 0, X5)
		TWIDDLE_MUL_INV(511 + 3*((2*me + 1)%512) + 1, X6)
		TWIDDLE_MUL_INV(511 + 3*((2*me + 1)%512) + 2, X7)	
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
		float4 *lwbv = (float4 *)lwb;	
		lwbv[me +   0] = float4(X0.x,X0.y,X4.x,X4.y);
		lwbv[me + 256] = float4(X1.x,X1.y,X5.x,X5.y);	
		lwbv[me + 512] = float4(X2.x,X2.y,X6.x,X6.y);
		lwbv[me + 768] = float4(X3.x,X3.y,X7.x,X7.y);			
	}
}

__global__
void fft_4096(hipLaunchParm lp, float2 *twiddles, float2 *buffer, const uint count, const int dir)
{
	uint me = hipThreadIdx_x;
	uint batch = hipBlockIdx_x;

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	float2 X8, X9, X10, X11, X12, X13, X14, X15;
	
	__shared__ float lds[4096];
	float2 *lwb = buffer + batch*4096;
	

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
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_FWD(15 + 15*(me%16) + 14, X15)		
	}
	else
	{
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  0,  X1)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  1,  X2)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  2,  X3)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  3,  X4)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  4,  X5)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  5,  X6)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  6,  X7)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  7,  X8)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  8,  X9)
		TWIDDLE_MUL_INV(15 + 15*(me%16) +  9, X10)
		TWIDDLE_MUL_INV(15 + 15*(me%16) + 10, X11)
		TWIDDLE_MUL_INV(15 + 15*(me%16) + 11, X12)
		TWIDDLE_MUL_INV(15 + 15*(me%16) + 12, X13)
		TWIDDLE_MUL_INV(15 + 15*(me%16) + 13, X14)
		TWIDDLE_MUL_INV(15 + 15*(me%16) + 14, X15)
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
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_FWD(255 + 15*(me%256) + 14, X15)
	}
	else
	{
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  0,  X1)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  1,  X2)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  2,  X3)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  3,  X4)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  4,  X5)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  5,  X6)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  6,  X7)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  7,  X8)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  8,  X9)
		TWIDDLE_MUL_INV(255 + 15*(me%256) +  9, X10)
		TWIDDLE_MUL_INV(255 + 15*(me%256) + 10, X11)
		TWIDDLE_MUL_INV(255 + 15*(me%256) + 11, X12)
		TWIDDLE_MUL_INV(255 + 15*(me%256) + 12, X13)
		TWIDDLE_MUL_INV(255 + 15*(me%256) + 13, X14)
		TWIDDLE_MUL_INV(255 + 15*(me%256) + 14, X15)
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


