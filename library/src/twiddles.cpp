/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "rocfft_hip.h"

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

	size_t ns;
	float2 *twts;
	const void *twtc;

	switch (N)
	{
	case 134217728:
				ns = 256*4;
				twtc = &twiddle_dee_134217728[0][0];
				break;

	case 67108864:
				ns = 256*4;
				twtc = &twiddle_dee_67108864[0][0];
				break;

	case 33554432:
				ns = 256*4;
				twtc = &twiddle_dee_33554432[0][0];
				break;

	case 16777216:
				ns = 256*3;
				twtc = &twiddle_dee_16777216[0][0];
				break;

	case 8388608:
				ns = 256*3;
				twtc = &twiddle_dee_8388608[0][0];
				break;

	case 4194304:
				ns = 256*3;
				twtc = &twiddle_dee_4194304[0][0];
				break;

	case 2097152:
				ns = 256*3;
				twtc = &twiddle_dee_2097152[0][0];
				break;

	case 1048576:
				ns = 256*3;
				twtc = &twiddle_dee_1048576[0][0];
				break;

	case 524288:
				ns = 256*3;
				twtc = &twiddle_dee_524288[0][0];
				break;

	case 262144:
				ns = 256*3;
				twtc = &twiddle_dee_262144[0][0];
				break;

	case 131072:
				ns = 256*3;
				twtc = &twiddle_dee_131072[0][0];
				break;

	case 65536:
				ns = 256*2;
				twtc = &twiddle_dee_65536[0][0];
				break;

	case 32768:
				ns = 256*2;
				twtc = &twiddle_dee_32768[0][0];
				break;

	case 16384:
				ns = 256*2;
				twtc = &twiddle_dee_16384[0][0];
				break;

	case 8192:
				ns = 256*2;
				twtc = &twiddle_dee_8192[0][0];
				break;
	default:
				assert(false); break;
	}


	hipMalloc(&twts, ns*sizeof(float2));
	hipMemcpy(twts, twtc, ns*sizeof(float2), hipMemcpyHostToDevice);

	return twts;
}

void twiddles_delete(void *twt)
{
	if(twt)
		hipFree(twt);
}
