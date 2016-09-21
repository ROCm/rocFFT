#include <vector>
#include <assert.h>
#include <iostream>

#include "./devicecall.h"

#define __HIPCC__

#include <iostream>
#include <hip_runtime.h>

#include "./kernels/pow2_ip_entry.h"
#include "./kernels/pow2_op_entry.h"
#include "./kernels/pow2_large_entry.h"

// precision, placement, iL, oL, scheme, dim, length **, iStrides **, oStrides **
void device_call_template(void *, void *);

#define POW2_SMALL_A(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn[0], data->node->batch);\
}

#define POW2_SMALL_B(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn[0], data->node->batch);\
}

#define POW2_SMALL_C(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn[0]);\
}

POW2_SMALL_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4096),fft_4096_ip_d1_pk)
POW2_SMALL_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2048),fft_2048_ip_d1_pk)
POW2_SMALL_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1024),fft_1024_ip_d1_pk)
POW2_SMALL_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_512),fft_512_ip_d1_pk)
POW2_SMALL_C(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_256),fft_256_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_128),fft_128_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_64),fft_64_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_32),fft_32_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_16),fft_16_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_8),fft_8_ip_d1_pk)
POW2_SMALL_B(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4),fft_4_ip_d1_pk)
POW2_SMALL_A(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2),fft_2_ip_d1_pk)
POW2_SMALL_A(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1),fft_1_ip_d1_pk)

#define POW2_LARGE_BCC_A(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(float2 *)data->node->twiddles, (float2 *)data->node->twiddles_large, (float2 *)data->bufIn[0], (float2 *)data->bufOut[0]);\
}

#define POW2_LARGE_BRC_A(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0,\
				(float2 *)data->node->twiddles, (float2 *)data->bufIn[0], (float2 *)data->bufOut[0]);\
}

POW2_LARGE_BCC_A(FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_128),fft_64_128_bcc_d1_pk)
POW2_LARGE_BCC_A(FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_256),fft_64_256_bcc_d1_pk)
POW2_LARGE_BCC_A(FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_128_256),fft_128_256_bcc_d1_pk)
POW2_LARGE_BCC_A(FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_256_256),fft_256_256_bcc_d1_pk)

POW2_LARGE_BRC_A(FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_128_64),fft_128_64_brc_d1_pk)
POW2_LARGE_BRC_A(FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_64),fft_256_64_brc_d1_pk)
POW2_LARGE_BRC_A(FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_128),fft_256_128_brc_d1_pk)
POW2_LARGE_BRC_A(FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_256),fft_256_256_brc_d1_pk)


