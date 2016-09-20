
#define __HIPCC__

#include <hip_runtime.h>

void *twiddles_create(size_t N)
{
#include "./kernels/twiddles_pow2.h"
#include "./kernels/twiddles_pow2_large.h"


	float2 *twt;
	if(N <= 4096)
	{
		hipMalloc(&twt, N * sizeof(float2));

		switch (N)
		{
		case 4096: 	hipMemcpy(twt, &twiddles_4096[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 2048: 	hipMemcpy(twt, &twiddles_2048[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 1024: 	hipMemcpy(twt, &twiddles_1024[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 512:  	hipMemcpy(twt,  &twiddles_512[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 256:  	hipMemcpy(twt,  &twiddles_256[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 128:  	hipMemcpy(twt,  &twiddles_128[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 64:   	hipMemcpy(twt,   &twiddles_64[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 32:   	hipMemcpy(twt,   &twiddles_32[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 16:   	hipMemcpy(twt,   &twiddles_16[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 8:    	hipMemcpy(twt,    &twiddles_8[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 4:    	hipMemcpy(twt,    &twiddles_4[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 2:    	hipMemcpy(twt,    &twiddles_2[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
		case 1: 	break;
		}

		return twt;
	}

	size_t n3;
	float2 *twt3;
	const void *twtc3;
	
	switch (N)
	{
	case 16777216:		
				n3 = 256*3;
				twtc3 = &twiddle_dee_16777216[0][0];
				break;
				
	case 8388608:
				n3 = 256*3;
				twtc3 = &twiddle_dee_8388608[0][0];				
				break;
				
	case 4194304:
				n3 = 256*3;
				twtc3 = &twiddle_dee_4194304[0][0];				
				break;
				
	case 2097152:
				n3 = 256*3;
				twtc3 = &twiddle_dee_2097152[0][0];
				break;
				
	case 1048576:
				n3 = 256*3;
				twtc3 = &twiddle_dee_1048576[0][0];
				break;
				
	case 524288:
				n3 = 256*3;
				twtc3 = &twiddle_dee_524288[0][0];
				break;
				
	case 262144:
				n3 = 256*3;
				twtc3 = &twiddle_dee_262144[0][0];
				break;
				
	case 131072:
				n3 = 256*3;
				twtc3 = &twiddle_dee_131072[0][0];
				break;
				
	case 65536:
				n3 = 256*2;
				twtc3 = &twiddle_dee_65536[0][0];
				break;
				
	case 32768:
				n3 = 256*2;
				twtc3 = &twiddle_dee_32768[0][0];
				break;
				
	case 16384:
				n3 = 256*2;
				twtc3 = &twiddle_dee_16384[0][0];
				break;
				
	case 8192:
				n3 = 256*2;
				twtc3 = &twiddle_dee_8192[0][0];
				break;
	default:
				assert(false); break;
	}
	

	hipMalloc(&twt3, n3*sizeof(float2));
	hipMemcpy(twt3, twtc3, n3*sizeof(float2), hipMemcpyHostToDevice);

	return twt3;
}

void twiddles_delete(void *twt)
{
	if(twt)
		hipFree(twt);
}


