/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <vector>
#include <iostream>
#include <fftw3.h>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include "hipfft.h"


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
	hipfftHandle plan = NULL;
	size_t length = N;
	hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

	// Execute plan
	hipfftExecC2C(plan, (hipfftComplex *)x, (hipfftComplex *)x, HIPFFT_FORWARD);

	// Destroy plan
	hipfftDestroy(plan);	

	// Copy result back to host
	std::vector<float2> y(N);
	hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);

	for (size_t i = 0; i < N; i++)
	{
		std::cout << y[i].x << ", " << y[i].y << std::endl;
	}
}
