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

// precision, placement, iL, oL, scheme, length **, iStrides **, oStrides **
void device_call_template(void *, void *);

#define POW2_SMALL_0(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn, data->node->batch);\
}

#define POW2_SMALL_1(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn, data->node->batch);\
}

#define POW2_SMALL_2(FNAME,DNAME) \
void FNAME(void *data_p, void *back_p)\
{\
	DeviceCallIn *data = (DeviceCallIn *)data_p;\
	DeviceCallOut *back = (DeviceCallOut *)back_p;\
	hipLaunchKernel(HIP_KERNEL_NAME( DNAME<-1> ), dim3(data->gridParam.b_x), dim3(data->gridParam.tpb_x), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn);\
}

POW2_SMALL_2(FN_PRFX(dfn_sp_ip_ci_ci_stoc_4096),fft_4096_ip_d1_pk)
POW2_SMALL_2(FN_PRFX(dfn_sp_ip_ci_ci_stoc_2048),fft_2048_ip_d1_pk)
POW2_SMALL_2(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1024),fft_1024_ip_d1_pk)
POW2_SMALL_2(FN_PRFX(dfn_sp_ip_ci_ci_stoc_512),fft_512_ip_d1_pk)
POW2_SMALL_2(FN_PRFX(dfn_sp_ip_ci_ci_stoc_256),fft_256_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_128),fft_128_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_64),fft_64_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_32),fft_32_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_16),fft_16_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_8),fft_8_ip_d1_pk)
POW2_SMALL_1(FN_PRFX(dfn_sp_ip_ci_ci_stoc_4),fft_4_ip_d1_pk)
POW2_SMALL_0(FN_PRFX(dfn_sp_ip_ci_ci_stoc_2),fft_2_ip_d1_pk)
POW2_SMALL_0(FN_PRFX(dfn_sp_ip_ci_ci_stoc_1),fft_1_ip_d1_pk)

