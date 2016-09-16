
#define __HIPCC__

#include <hip_runtime.h>

void *twiddles_fn()
{
#include "./kernels/twiddles_pow2.h"
#include "./kernels/twiddles_pow2_large.h"

	float2 *twt;
	hipMalloc(&twt, 16 * sizeof(float2));
	hipMemcpy(twt,   &twiddles_16[0], 16 * sizeof(float2), hipMemcpyHostToDevice);

	return twt;
}

