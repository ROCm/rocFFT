/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#define __HIPCC__

#include <vector>
#include <iostream>
#include <fftw3.h>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include "rocfft.h"


int main()
{
	const size_t N = 16;

	// FFTW reference compute
	// ==========================================

	float2 cx[N];
	fftwf_complex *in, *out;
	fftwf_plan p;

	in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
	out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
	p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


	for (size_t i = 0; i < N; i++)
	{
		cx[i].x = in[i][0] = i + (i%3) - (i%7);
		cx[i].y = in[i][1] = 0;
	}


	fftwf_execute(p);

	for (size_t i = 0; i < N; i++)
	{
		std::cout << out[i][0] << ", " << out[i][1] << std::endl;
	}

	fftwf_destroy_plan(p);
	fftwf_free(in); fftwf_free(out);


	// rocfft gpu compute
	// ========================================

	size_t Nbytes = N * sizeof(float2);

	// Create HIP device object.
	float2 *x;
	hipMalloc(&x, Nbytes);

	//  Copy data to device
	hipMemcpy(x, &cx[0], Nbytes, hipMemcpyHostToDevice);

	// Create plan
	rocfft_plan plan = NULL;
	size_t length = N;
	rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, &length, 1, NULL);

	// Execute plan
	rocfft_execute(plan, (void**) &x, NULL, NULL);

	// Destroy plan
	rocfft_plan_destroy(plan);

	// Copy result back to host
	std::vector<float2> y(N);
	hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);

	for (size_t i = 0; i < N; i++)
	{
		std::cout << y[i].x << ", " << y[i].y << std::endl;
	}
}
