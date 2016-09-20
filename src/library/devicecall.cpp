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



void FN_PRFX(dfn_sp_ip_ci_ci_stoc_16)(void *data_p, void *back_p)
{
	DeviceCallIn *data = (DeviceCallIn *)data_p;
	DeviceCallOut *back = (DeviceCallOut *)back_p;

	size_t WGS = 64;
	size_t NT = 16;

	const unsigned B = data->node->batch;
	const unsigned blocks = (B%NT) ? 1 + (B / NT) : (B / NT);
	const unsigned threadsPerBlock = WGS;

	hipLaunchKernel(HIP_KERNEL_NAME( fft_16_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, (float2 *)data->node->twiddles, (float2 *)data->bufIn, B);

}


