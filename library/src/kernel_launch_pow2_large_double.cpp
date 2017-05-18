/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/



#include "kernel_launch.h"

#include <iostream>
#include "rocfft_hip.h"
#include "./kernels/common.h" 
#include "./kernels/pow2_ip_entry.h"
#include "./kernels/pow2_op_entry.h"
#include "./kernels/pow2_large_entry.h"

// precision, placement, iL, oL, scheme, dim, length **, iStrides **, oStrides **
void device_call_template(void *, void *);

#if 0

#define POW2_SMALL_IP_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], data->node->batch);\
}

#define POW2_SMALL_IP_B(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], data->node->batch);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], data->node->batch);\
}

#define POW2_SMALL_IP_C(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0]);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0]);\
}

#define POW2_SMALL_OP_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0],  (double2 *)data->bufOut[0], data->node->batch);\
}

#define POW2_SMALL_OP_B(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0], data->node->batch);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0], data->node->batch);\
}

#define POW2_SMALL_OP_C(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
}

/* INPLACE */
POW2_SMALL_IP_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_4096,fft_4096_ip_d1_pk)
POW2_SMALL_IP_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_2048,fft_2048_ip_d1_pk)
POW2_SMALL_IP_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_1024,fft_1024_ip_d1_pk)
POW2_SMALL_IP_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_512,fft_512_ip_d1_pk)
POW2_SMALL_IP_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_256,fft_256_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_128,fft_128_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_64,fft_64_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_32,fft_32_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_16,fft_16_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_8,fft_8_ip_d1_pk)
POW2_SMALL_IP_B(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_4,fft_4_ip_d1_pk)
POW2_SMALL_IP_A(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_2,fft_2_ip_d1_pk)
POW2_SMALL_IP_A(rocfft_internal_dfn_dp_ip_ci_ci_stoc_1_1,fft_1_ip_d1_pk)

 /* OUTPLACE */
POW2_SMALL_OP_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_4096,fft_4096_op_d1_pk)
POW2_SMALL_OP_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_2048,fft_2048_op_d1_pk)
POW2_SMALL_OP_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_1024,fft_1024_op_d1_pk)
POW2_SMALL_OP_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_512,fft_512_op_d1_pk)
POW2_SMALL_OP_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_256,fft_256_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_128,fft_128_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_64,fft_64_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_32,fft_32_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_16,fft_16_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_8,fft_8_op_d1_pk)
POW2_SMALL_OP_B(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_4,fft_4_op_d1_pk)
POW2_SMALL_OP_A(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_2,fft_2_op_d1_pk)
POW2_SMALL_OP_A(rocfft_internal_dfn_dp_op_ci_ci_stoc_1_1,fft_1_op_d1_pk)
#else 

#endif


#define POW2_LARGE_BCC_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->node->twiddles_large, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->node->twiddles_large, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
}

#define POW2_LARGE_BRC_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0]);\
}

POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_64_128,  fft_64_128_bcc_d1_pk)
POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_64_256,  fft_64_256_bcc_d1_pk)
POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_128_256, fft_128_256_bcc_d1_pk)
POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_256_256, fft_256_256_bcc_d1_pk)

POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_64_2048, fft_64_2048_bcc_d1_pk)
POW2_LARGE_BCC_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_2_64_4096, fft_64_4096_bcc_d1_pk)

POW2_LARGE_BRC_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_2_128_64,  fft_128_64_brc_d1_pk)
POW2_LARGE_BRC_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_2_256_64,  fft_256_64_brc_d1_pk)
POW2_LARGE_BRC_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_2_256_128, fft_256_128_brc_d1_pk)
POW2_LARGE_BRC_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_2_256_256, fft_256_256_brc_d1_pk)

#define TRANSPOSE_CALL(NUM_Y,DRN,TWL,TTD)	\
hipLaunchKernel(HIP_KERNEL_NAME( transpose_var1<double2, DRN,TWL,TTD> ),\
					dim3(data->gridParam.b_x, data->gridParam.b_y), dim3(data->gridParam.tpb_x, data->gridParam.tpb_x), 0, 0,\
					(double2 *)data->node->twiddles_large, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
					NUM_Y, data->node->inStride[1], data->node->outStride[1], data->node->iDist, data->node->oDist)

void rocfft_internal_transpose_var1_dp(void *data_p, void *back_p)
{
	DeviceCallIn *data = (DeviceCallIn *)data_p;
	DeviceCallOut *back = (DeviceCallOut *)back_p;

	if(data->node->transTileDir == TTD_IP_HOR)
	{
		if(data->node->large1D == 0)
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[1]/64), -1, 0, TTD_IP_HOR);
			else
			TRANSPOSE_CALL((data->node->length[1]/64),  1, 0, TTD_IP_HOR);
		}
		else if(data->node->large1D <= 16777216)
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[1]/64), -1, 3, TTD_IP_HOR);
			else
			TRANSPOSE_CALL((data->node->length[1]/64),  1, 3, TTD_IP_HOR);
		}
		else
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[1]/64), -1, 4, TTD_IP_HOR);
			else
			TRANSPOSE_CALL((data->node->length[1]/64),  1, 4, TTD_IP_HOR);
		}
	}
	else
	{
		if(data->node->large1D == 0)
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[0]/64), -1, 0, TTD_IP_VER);
			else
			TRANSPOSE_CALL((data->node->length[0]/64),  1, 0, TTD_IP_VER);
		}
		else if(data->node->large1D <= 16777216)
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[0]/64), -1, 3, TTD_IP_VER);
			else
			TRANSPOSE_CALL((data->node->length[0]/64),  1, 3, TTD_IP_VER);
		}
		else
		{
			if(data->node->direction == -1)
			TRANSPOSE_CALL((data->node->length[0]/64), -1, 4, TTD_IP_VER);
			else
			TRANSPOSE_CALL((data->node->length[0]/64),  1, 4, TTD_IP_VER);
		}
	}
}


#define POW2_SMALL_IP_2_C(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0],\
				data->node->length[1], data->node->inStride[1], data->node->iDist);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0],\
				data->node->length[1], data->node->inStride[1], data->node->iDist);\
}

#define POW2_SMALL_OP_2_C(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[1], data->node->inStride[1], data->node->outStride[1], data->node->iDist, data->node->oDist);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[1], data->node->inStride[1], data->node->outStride[1], data->node->iDist, data->node->oDist);\
}


POW2_SMALL_IP_2_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_2_4096, fft_4096_ip_d2_s1)
POW2_SMALL_IP_2_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_2_2048, fft_2048_ip_d2_s1)
POW2_SMALL_IP_2_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_2_1024, fft_1024_ip_d2_s1)
POW2_SMALL_IP_2_C(rocfft_internal_dfn_dp_ip_ci_ci_stoc_2_512,  fft_512_ip_d2_s1)

POW2_SMALL_OP_2_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_2_4096, fft_4096_op_d2_s1)
POW2_SMALL_OP_2_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_2_2048, fft_2048_op_d2_s1)
POW2_SMALL_OP_2_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_2_1024, fft_1024_op_d2_s1)
POW2_SMALL_OP_2_C(rocfft_internal_dfn_dp_op_ci_ci_stoc_2_512,  fft_512_op_d2_s1)


#define POW2_LARGE_BCC_3_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->node->twiddles_large, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[2], data->node->inStride[2], data->node->outStride[2], data->node->iDist, data->node->oDist);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->node->twiddles_large, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[2], data->node->inStride[2], data->node->outStride[2], data->node->iDist, data->node->oDist);\
}

#define POW2_LARGE_BRC_3_A(FUNCTION_NAME,KERN_NAME) \
void FUNCTION_NAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	if(data->node->direction == -1) \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[2], data->node->inStride[2], data->node->outStride[2], data->node->iDist, data->node->oDist);\
	else \
	hipLaunchKernel(HIP_KERNEL_NAME( KERN_NAME<double2,1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(double2 *)data->node->twiddles, (double2 *)data->bufIn[0], (double2 *)data->bufOut[0],\
				data->node->length[2], data->node->inStride[2], data->node->outStride[2], data->node->iDist, data->node->oDist);\
}

POW2_LARGE_BCC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_3_64_128,  fft_64_128_bcc_d2_s1)
POW2_LARGE_BCC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_3_64_256,  fft_64_256_bcc_d2_s1)
POW2_LARGE_BCC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_3_128_256, fft_128_256_bcc_d2_s1)
POW2_LARGE_BCC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbcc_3_256_256, fft_256_256_bcc_d2_s1)

POW2_LARGE_BRC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_3_128_64,  fft_128_64_brc_d2_s1)
POW2_LARGE_BRC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_3_256_64,  fft_256_64_brc_d2_s1)
POW2_LARGE_BRC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_3_256_128, fft_256_128_brc_d2_s1)
POW2_LARGE_BRC_3_A(rocfft_internal_dfn_dp_op_ci_ci_sbrc_3_256_256, fft_256_256_brc_d2_s1)
