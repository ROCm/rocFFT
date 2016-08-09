
#include "twiddles_pow2.cl"
#include "butterfly.cl"



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_1(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*64 + me)*2;
	

	if(rw)
	{
		X0 = lwb[0];
		lwb[0] = X0;
	}
}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_2(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0, X1;

	uint rw = (me < (count - batch*64)) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*64 + me)*2;
	

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



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_4(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[128];
	float2 X0, X1;

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*32 + (me/2))*4;
	__local  float *ldsp = lds + (me/2)*4;
	
	me = me % 2;

	if(rw)
	{	
		X0 = lwb[me + 0];
		X1 = lwb[me + 2];	
	}
	
	Rad2(&X0, &X1);

	ldsp[me*2 + 0] = X0.x;
	ldsp[me*2 + 1] = X1.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 2];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ldsp[me*2 + 0] = X0.y;
	ldsp[me*2 + 1] = X1.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 2];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_4, 1 + (me%2), X1)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_4, 1 + (me%2), X1)
	}
	
	Rad2(&X0, &X1);
	
	if(rw)
	{	
		lwb[me + 0] = X0;
		lwb[me + 2] = X1;	
	}
}



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_8(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*32)*2) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*32 + (me/2))*8;
	__local  float *ldsp = lds + (me/2)*8;
	
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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[me*2 + 0 + 0];
	X1.x = ldsp[me*2 + 0 + 4];	
	X2.x = ldsp[me*2 + 1 + 0];
	X3.x = ldsp[me*2 + 1 + 4];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[me*2 + 0 + 0];
	X1.y = ldsp[me*2 + 0 + 4];	
	X2.y = ldsp[me*2 + 1 + 0];
	X3.y = ldsp[me*2 + 1 + 4];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_8, 3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_FWD(twiddles_8, 3 + ((2*me+1)%4), X3)	
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_8, 3 + ((2*me+0)%4), X1)	
		TWIDDLE_MUL_INV(twiddles_8, 3 + ((2*me+1)%4), X3)			
	}
	
	Rad2(&X0, &X1);
	Rad2(&X2, &X3);
	
	if(rw)
	{
		__global float4 *lwbv = lwb;	
		lwbv[me + 0] = (float4)(X0,X2);
		lwbv[me + 2] = (float4)(X1,X3);	
	}
}



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_16(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*16 + (me/4))*16;
	__local  float *ldsp = lds + (me/4)*16;
	
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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 4];	
	X2.x = ldsp[me + 8];
	X3.x = ldsp[me + 12];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 4];	
	X2.y = ldsp[me + 8];
	X3.y = ldsp[me + 12];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_16, 3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_FWD(twiddles_16, 3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_16, 3 + 3*(me%4) + 2, X3)
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_16, 3 + 3*(me%4) + 0, X1)		
		TWIDDLE_MUL_INV(twiddles_16, 3 + 3*(me%4) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_16, 3 + 3*(me%4) + 2, X3)	
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



__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_32(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	uint rw = (me < (count - batch*16)*4) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*16 + (me/4))*32;
	__local  float *ldsp = lds + (me/4)*32;
	
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
				
	barrier(CLK_LOCAL_MEM_FENCE);

	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) +  8];
	X2.x = ldsp[(2*me + 0) + 16];
	X3.x = ldsp[(2*me + 0) + 24];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) +  8];
	X6.x = ldsp[(2*me + 1) + 16];
	X7.x = ldsp[(2*me + 1) + 24];	

	barrier(CLK_LOCAL_MEM_FENCE);
				

	ldsp[me*8 + 0] = X0.y;
	ldsp[me*8 + 1] = X1.y;
	ldsp[me*8 + 2] = X2.y;
	ldsp[me*8 + 3] = X3.y;
	ldsp[me*8 + 4] = X4.y;
	ldsp[me*8 + 5] = X5.y;
	ldsp[me*8 + 6] = X6.y;
	ldsp[me*8 + 7] = X7.y;
				
	barrier(CLK_LOCAL_MEM_FENCE);

	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) +  8];
	X2.y = ldsp[(2*me + 0) + 16];
	X3.y = ldsp[(2*me + 0) + 24];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) +  8];
	X6.y = ldsp[(2*me + 1) + 16];
	X7.y = ldsp[(2*me + 1) + 24];	

	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(dir == -1)
	{
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_FWD(twiddles_32, 7 + 3*((2*me + 1)%8) + 2, X7)		
	}
	else
	{
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 0)%8) + 0, X1)
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 0)%8) + 1, X2)
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 0)%8) + 2, X3)
		
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 1)%8) + 0, X5)
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 1)%8) + 1, X6)
		TWIDDLE_MUL_INV(twiddles_32, 7 + 3*((2*me + 1)%8) + 2, X7)	
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
		__global float4 *lwbv = lwb;	
		lwbv[me +  0] = (float4)(X0,X4);
		lwbv[me +  4] = (float4)(X1,X5);	
		lwbv[me +  8] = (float4)(X2,X6);
		lwbv[me + 12] = (float4)(X3,X7);			
	}			
	
}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_64(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[256];
	float2 X0, X1, X2, X3;

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*4 + (me/16))*64;
	__local  float *ldsp = lds + (me/16)*64;
	
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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 16];	
	X2.x = ldsp[me + 32];
	X3.x = ldsp[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ldsp[me*4 + 0] = X0.y;
	ldsp[me*4 + 1] = X1.y;
	ldsp[me*4 + 2] = X2.y;
	ldsp[me*4 + 3] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 16];	
	X2.y = ldsp[me + 32];
	X3.y = ldsp[me + 48];
	
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
	
	
	ldsp[(me/4)*16 + me%4 +  0] = X0.x;
	ldsp[(me/4)*16 + me%4 +  4] = X1.x;
	ldsp[(me/4)*16 + me%4 +  8] = X2.x;
	ldsp[(me/4)*16 + me%4 + 12] = X3.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[me + 0];
	X1.x = ldsp[me + 16];	
	X2.x = ldsp[me + 32];
	X3.x = ldsp[me + 48];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ldsp[(me/4)*16 + me%4 +  0] = X0.y;
	ldsp[(me/4)*16 + me%4 +  4] = X1.y;
	ldsp[(me/4)*16 + me%4 +  8] = X2.y;
	ldsp[(me/4)*16 + me%4 + 12] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[me + 0];
	X1.y = ldsp[me + 16];	
	X2.y = ldsp[me + 32];
	X3.y = ldsp[me + 48];
	
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

		
	if(rw)
	{	
		lwb[me + 0]  = X0;
		lwb[me + 16] = X1;	
		lwb[me + 32] = X2;
		lwb[me + 48] = X3;		
	}
}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_128(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[512];
	float2 X0, X1, X2, X3, X4, X5, X6, X7;

	uint rw = (me < (count - batch*4)*16) ? 1 : 0;

	__global float2 *lwb = buffer + (batch*4 + (me/16))*128;
	__local  float *ldsp = lds + (me/16)*128;
	
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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) + 32];
	X2.x = ldsp[(2*me + 0) + 64];
	X3.x = ldsp[(2*me + 0) + 96];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) + 32];
	X6.x = ldsp[(2*me + 1) + 64];
	X7.x = ldsp[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);

	ldsp[me*8 + 0] = X0.y;
	ldsp[me*8 + 1] = X1.y;
	ldsp[me*8 + 2] = X2.y;
	ldsp[me*8 + 3] = X3.y;
	ldsp[me*8 + 4] = X4.y;
	ldsp[me*8 + 5] = X5.y;
	ldsp[me*8 + 6] = X6.y;
	ldsp[me*8 + 7] = X7.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) + 32];
	X2.y = ldsp[(2*me + 0) + 64];
	X3.y = ldsp[(2*me + 0) + 96];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) + 32];
	X6.y = ldsp[(2*me + 1) + 64];
	X7.y = ldsp[(2*me + 1) + 96];	

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
	
	
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.x;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.x;
	
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.x;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = ldsp[(2*me + 0) +  0];
	X1.x = ldsp[(2*me + 0) + 32];
	X2.x = ldsp[(2*me + 0) + 64];
	X3.x = ldsp[(2*me + 0) + 96];

	X4.x = ldsp[(2*me + 1) +  0];
	X5.x = ldsp[(2*me + 1) + 32];
	X6.x = ldsp[(2*me + 1) + 64];
	X7.x = ldsp[(2*me + 1) + 96];	

	barrier(CLK_LOCAL_MEM_FENCE);

	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  0] = X0.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 +  8] = X1.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 16] = X2.y;
	ldsp[((2*me + 0)/8)*32 + (2*me + 0)%8 + 24] = X3.y;
	
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  0] = X4.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 +  8] = X5.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 16] = X6.y;
	ldsp[((2*me + 1)/8)*32 + (2*me + 1)%8 + 24] = X7.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = ldsp[(2*me + 0) +  0];
	X1.y = ldsp[(2*me + 0) + 32];
	X2.y = ldsp[(2*me + 0) + 64];
	X3.y = ldsp[(2*me + 0) + 96];

	X4.y = ldsp[(2*me + 1) +  0];
	X5.y = ldsp[(2*me + 1) + 32];
	X6.y = ldsp[(2*me + 1) + 64];
	X7.y = ldsp[(2*me + 1) + 96];	

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
		
		
	if(rw)
	{
		__global float4 *lwbv = lwb;	
		lwbv[me +  0] = (float4)(X0,X4);
		lwbv[me + 16] = (float4)(X1,X5);	
		lwbv[me + 32] = (float4)(X2,X6);
		lwbv[me + 48] = (float4)(X3,X7);			
	}	
	
}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_256(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	__local float lds[256];
	float2 X0, X1, X2, X3;

	__global float2 *lwb = buffer + batch*256;
	

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
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	lds[me*4 + 0] = X0.y;
	lds[me*4 + 1] = X1.y;
	lds[me*4 + 2] = X2.y;
	lds[me*4 + 3] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
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
		
		
	lds[(me/4)*16 + me%4 +  0] = X0.x;
	lds[(me/4)*16 + me%4 +  4] = X1.x;
	lds[(me/4)*16 + me%4 +  8] = X2.x;
	lds[(me/4)*16 + me%4 + 12] = X3.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	lds[(me/4)*16 + me%4 +  0] = X0.y;
	lds[(me/4)*16 + me%4 +  4] = X1.y;
	lds[(me/4)*16 + me%4 +  8] = X2.y;
	lds[(me/4)*16 + me%4 + 12] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
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


	lds[(me/16)*64 + me%16 +  0] = X0.x;
	lds[(me/16)*64 + me%16 + 16] = X1.x;
	lds[(me/16)*64 + me%16 + 32] = X2.x;
	lds[(me/16)*64 + me%16 + 48] = X3.x;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.x = lds[me +   0];
	X1.x = lds[me +  64];	
	X2.x = lds[me + 128];
	X3.x = lds[me + 192];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	lds[(me/16)*64 + me%16 +  0] = X0.y;
	lds[(me/16)*64 + me%16 + 16] = X1.y;
	lds[(me/16)*64 + me%16 + 32] = X2.y;
	lds[(me/16)*64 + me%16 + 48] = X3.y;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	X0.y = lds[me +   0];
	X1.y = lds[me +  64];	
	X2.y = lds[me + 128];
	X3.y = lds[me + 192];
	
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
		

	lwb[me +   0] = X0;
	lwb[me +  64] = X1;	
	lwb[me + 128] = X2;
	lwb[me + 192] = X3;		

}


__kernel __attribute__((reqd_work_group_size (64,1,1)))
void fft_512(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__local float lds[512];
	__global float2 *lwb = buffer + batch*512;
	

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
	

__kernel __attribute__((reqd_work_group_size (128,1,1)))
void fft_1024(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__local float lds[1024];
	__global float2 *lwb = buffer + batch*1024;
	
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
		__global float4 *lwbv = lwb;	
		lwbv[me +   0] = (float4)(X0,X4);
		lwbv[me + 128] = (float4)(X1,X5);	
		lwbv[me + 256] = (float4)(X2,X6);
		lwbv[me + 384] = (float4)(X3,X7);			
	}	
}


	
__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_2048(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	__local float lds[2048];
	__global float2 *lwb = buffer + batch*2048;
	

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

__kernel __attribute__((reqd_work_group_size (256,1,1)))
void fft_4096(__global float2 *buffer, const uint count, const int dir)
{
	uint me = get_local_id(0);
	uint batch = get_group_id(0);

	float2 X0, X1, X2, X3, X4, X5, X6, X7;
	float2 X8, X9, X10, X11, X12, X13, X14, X15;
	
	__local float lds[4096];
	__global float2 *lwb = buffer + batch*4096;
	

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


