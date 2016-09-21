#ifndef BUTTERFLY_H
#define BUTTERFLY_H

#ifdef __NVCC__

#include "vector_types.h"

__device__ float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x-b.x, a.y-b.y); }
__device__ float2 operator+(const float2 &a, const float2 &b) { return make_float2(a.x+b.x, a.y+b.y); }
__device__ float2 operator*(const float &a, const float2 &b) { return make_float2(a*b.x, a*b.y); }

#endif


#ifdef __NVCC__
#define MAKE_FLOAT2 make_float2
#define MAKE_FLOAT4 make_float4
#else
#define MAKE_FLOAT2 float2
#define MAKE_FLOAT4 float4
#endif

enum StrideBin
{
	SB_UNIT,
	SB_NONUNIT,
};

#define TWIDDLE_3STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
	{ \
		float2 W = TWFUNC(TWIDDLES, INDEX); \
		float TR, TI; \
		TR = (W.x * REG.x) - (W.y * REG.y); \
		TI = (W.y * REG.x) + (W.x * REG.y); \
		REG.x = TR; \
		REG.y = TI; \
	}

#define TWIDDLE_3STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
	{ \
		float2 W = TWFUNC(TWIDDLES, INDEX); \
		float TR, TI; \
		TR =  (W.x * REG.x) + (W.y * REG.y); \
		TI = -(W.y * REG.x) + (W.x * REG.y); \
		REG.x = TR; \
		REG.y = TI; \
	}
	
#define TWIDDLE_MUL_FWD(TWIDDLES, INDEX, REG) \
{ \
	float2 W = TWIDDLES[INDEX]; \
	float TR, TI; \
	TR = (W.x * REG.x) - (W.y * REG.y); \
	TI = (W.y * REG.x) + (W.x * REG.y); \
	REG.x = TR; \
	REG.y = TI; \
}

#define TWIDDLE_MUL_INV(TWIDDLES, INDEX, REG) \
{ \
	float2 W = TWIDDLES[INDEX]; \
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
	(*R3) = (*R1) + MAKE_FLOAT2(-(*R3).y, (*R3).x);
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
	(*R3) = (*R1) + MAKE_FLOAT2((*R3).y, -(*R3).x);
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
	(*R3) = (*R1) + MAKE_FLOAT2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + MAKE_FLOAT2(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * MAKE_FLOAT2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + MAKE_FLOAT2(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * MAKE_FLOAT2((*R7).y, -(*R7).x);
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
	(*R3) = (*R1) + MAKE_FLOAT2((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + MAKE_FLOAT2((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * MAKE_FLOAT2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + MAKE_FLOAT2((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * MAKE_FLOAT2((*R7).y, -(*R7).x);
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
	(*R3) = (*R1) + MAKE_FLOAT2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + MAKE_FLOAT2(-(*R7).y, (*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	(*R10) = (*R8) - (*R10);
	(*R8) = 2.0f * (*R8) - (*R10);
	(*R11) = (*R9) + MAKE_FLOAT2(-(*R11).y, (*R11).x);
	(*R9) = 2.0f * (*R9) - (*R11);
	(*R14) = (*R12) - (*R14);
	(*R12) = 2.0f * (*R12) - (*R14);
	(*R15) = (*R13) + MAKE_FLOAT2(-(*R15).y, (*R15).x);
	(*R13) = 2.0f * (*R13) - (*R15);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) - C8Q * MAKE_FLOAT2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + MAKE_FLOAT2(-(*R6).y, (*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) - C8Q * MAKE_FLOAT2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	(*R12) = (*R8) - (*R12);
	(*R8) = 2.0f * (*R8) - (*R12);
	(*R13) = ((*R9) - C8Q * (*R13)) - C8Q * MAKE_FLOAT2((*R13).y, -(*R13).x);
	(*R9) = 2.0f * (*R9) - (*R13);
	(*R14) = (*R10) + MAKE_FLOAT2(-(*R14).y, (*R14).x);
	(*R10) = 2.0f * (*R10) - (*R14);
	(*R15) = ((*R11) + C8Q * (*R15)) - C8Q * MAKE_FLOAT2((*R15).y, -(*R15).x);
	(*R11) = 2.0f * (*R11) - (*R15);
	
	(*R8) = (*R0) - (*R8);
	(*R0) = 2.0f * (*R0) - (*R8);
	(*R9) = ((*R1) - C16A * (*R9)) - C16B * MAKE_FLOAT2((*R9).y, -(*R9).x);
	T = (*R8);
	(*R1) = 2.0f * (*R1) - (*R9);
	
	(*R10) = ((*R2) - C8Q * (*R10)) - C8Q * MAKE_FLOAT2((*R10).y, -(*R10).x);
	(*R2) = 2.0f * (*R2) - (*R10);
	(*R11) = ((*R3) - C16B * (*R11)) - C16A * MAKE_FLOAT2((*R11).y, -(*R11).x);
	(*R3) = 2.0f * (*R3) - (*R11);
	
	(*R12) = (*R4) + MAKE_FLOAT2(-(*R12).y, (*R12).x);
	(*R4) = 2.0f * (*R4) - (*R12);
	(*R13) = ((*R5) + C16B * (*R13)) - C16A * MAKE_FLOAT2((*R13).y, -(*R13).x);
	(*R5) = 2.0f * (*R5) - (*R13);
	
	(*R14) = ((*R6) + C8Q * (*R14)) - C8Q * MAKE_FLOAT2((*R14).y, -(*R14).x);
	(*R6) = 2.0f * (*R6) - (*R14);
	(*R15) = ((*R7) + C16A * (*R15)) - C16B * MAKE_FLOAT2((*R15).y, -(*R15).x);
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
	(*R3) = (*R1) + MAKE_FLOAT2((*R3).y, -(*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);
	(*R6) = (*R4) - (*R6);
	(*R4) = 2.0f * (*R4) - (*R6);
	(*R7) = (*R5) + MAKE_FLOAT2((*R7).y, -(*R7).x);
	(*R5) = 2.0f * (*R5) - (*R7);
	(*R10) = (*R8) - (*R10);
	(*R8) = 2.0f * (*R8) - (*R10);
	(*R11) = (*R9) + MAKE_FLOAT2((*R11).y, -(*R11).x);
	(*R9) = 2.0f * (*R9) - (*R11);
	(*R14) = (*R12) - (*R14);
	(*R12) = 2.0f * (*R12) - (*R14);
	(*R15) = (*R13) + MAKE_FLOAT2((*R15).y, -(*R15).x);
	(*R13) = 2.0f * (*R13) - (*R15);
	
	(*R4) = (*R0) - (*R4);
	(*R0) = 2.0f * (*R0) - (*R4);
	(*R5) = ((*R1) - C8Q * (*R5)) + C8Q * MAKE_FLOAT2((*R5).y, -(*R5).x);
	(*R1) = 2.0f * (*R1) - (*R5);
	(*R6) = (*R2) + MAKE_FLOAT2((*R6).y, -(*R6).x);
	(*R2) = 2.0f * (*R2) - (*R6);
	(*R7) = ((*R3) + C8Q * (*R7)) + C8Q * MAKE_FLOAT2((*R7).y, -(*R7).x);
	(*R3) = 2.0f * (*R3) - (*R7);
	(*R12) = (*R8) - (*R12);
	(*R8) = 2.0f * (*R8) - (*R12);
	(*R13) = ((*R9) - C8Q * (*R13)) + C8Q * MAKE_FLOAT2((*R13).y, -(*R13).x);
	(*R9) = 2.0f * (*R9) - (*R13);
	(*R14) = (*R10) + MAKE_FLOAT2((*R14).y, -(*R14).x);
	(*R10) = 2.0f * (*R10) - (*R14);
	(*R15) = ((*R11) + C8Q * (*R15)) + C8Q * MAKE_FLOAT2((*R15).y, -(*R15).x);
	(*R11) = 2.0f * (*R11) - (*R15);
	
	(*R8) = (*R0) - (*R8);
	(*R0) = 2.0f * (*R0) - (*R8);
	(*R9) = ((*R1) - C16A * (*R9)) + C16B * MAKE_FLOAT2((*R9).y, -(*R9).x);
	(*R1) = 2.0f * (*R1) - (*R9);
	(*R10) = ((*R2) - C8Q * (*R10)) + C8Q * MAKE_FLOAT2((*R10).y, -(*R10).x);
	(*R2) = 2.0f * (*R2) - (*R10);
	(*R11) = ((*R3) - C16B * (*R11)) + C16A * MAKE_FLOAT2((*R11).y, -(*R11).x);
	(*R3) = 2.0f * (*R3) - (*R11);
	(*R12) = (*R4) + MAKE_FLOAT2((*R12).y, -(*R12).x);
	(*R4) = 2.0f * (*R4) - (*R12);
	(*R13) = ((*R5) + C16B * (*R13)) + C16A * MAKE_FLOAT2((*R13).y, -(*R13).x);
	(*R5) = 2.0f * (*R5) - (*R13);
	(*R14) = ((*R6) + C8Q * (*R14)) + C8Q * MAKE_FLOAT2((*R14).y, -(*R14).x);
	(*R6) = 2.0f * (*R6) - (*R14);
	(*R15) = ((*R7) + C16A * (*R15)) + C16B * MAKE_FLOAT2((*R15).y, -(*R15).x);
	(*R7) = 2.0f * (*R7) - (*R15);
	
	T = (*R1); (*R1) = (*R8); (*R8) = T;
	T = (*R2); (*R2) = (*R4); (*R4) = T;
	T = (*R3); (*R3) = (*R12); (*R12) = T;
	T = (*R5); (*R5) = (*R10); (*R10) = T;
	T = (*R7); (*R7) = (*R14); (*R14) = T;
	T = (*R11); (*R11) = (*R13); (*R13) = T;
	
}

#endif // BUTTERFLY_H

