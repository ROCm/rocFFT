
#define __HIPCC__

#include <iostream>
#include <hip_runtime.h>
#include <fftw3.h>

#include <time.h>
#include <math.h>

struct Timer
{
    struct timespec start, end;

public:
    Timer() { }

    void Start() { clock_gettime(CLOCK_MONOTONIC, &start); }
    double Sample()
    {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        return time * 1E-9;
    }
};

__device__ void FwdRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

	float2 T;

	(*R1) = (*R0) - (*R1);
	(*R0) = 2.0f * (*R0) - (*R1);
	(*R3) = (*R2) - (*R3);
	(*R2) = 2.0f * (*R2) - (*R3);

	(*R2) = (*R0) - (*R2);
	(*R0) = 2.0f * (*R0) - (*R2);
	(*R3) = (*R1) + float2(-(*R3).y, (*R3).x);
	(*R1) = 2.0f * (*R1) - (*R3);

	T = (*R1); (*R1) = (*R2); (*R2) = T;

}


__device__ void FwdPass0(uint me, uint inOffset, uint outOffset,
	float2 *bufIn, float *bufOutRe, float *bufOutIm,
	float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{
	
	(*R0) = bufIn[inOffset + (0 + me * 1 + 0 + 0) * 1];
	(*R1) = bufIn[inOffset + (0 + me * 1 + 0 + 4) * 1];
	(*R2) = bufIn[inOffset + (0 + me * 1 + 0 + 8) * 1];
	(*R3) = bufIn[inOffset + (0 + me * 1 + 0 + 12) * 1];
	
	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();

	bufOutRe[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0) * 1] = (*R0).x;
	bufOutRe[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1) * 1] = (*R1).x;
	bufOutRe[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2) * 1] = (*R2).x;
	bufOutRe[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3) * 1] = (*R3).x;

	__syncthreads();

	(*R0).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 0) * 1];
	(*R1).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 4) * 1];
	(*R2).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 8) * 1];
	(*R3).x = bufOutRe[outOffset + (0 + me * 1 + 0 + 12) * 1];

	__syncthreads();

	bufOutIm[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0) * 1] = (*R0).y;
	bufOutIm[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1) * 1] = (*R1).y;
	bufOutIm[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2) * 1] = (*R2).y;
	bufOutIm[outOffset + (((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3) * 1] = (*R3).y;

	__syncthreads();

	(*R0).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 0) * 1];
	(*R1).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 4) * 1];
	(*R2).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 8) * 1];
	(*R3).y = bufOutIm[outOffset + (0 + me * 1 + 0 + 12) * 1];

	__syncthreads();

}

__device__ void FwdPass1(uint me, uint inOffset, uint outOffset,
	float2 *bufOut, float2 *twiddles,
	float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{
	
	{
		float2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 0];
		float TR, TI;
		TR = (W.x * (*R1).x) - (W.y * (*R1).y);
		TI = (W.y * (*R1).x) + (W.x * (*R1).y);
		(*R1).x = TR;
		(*R1).y = TI;
	}

	{
		float2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 1];
		float TR, TI;
		TR = (W.x * (*R2).x) - (W.y * (*R2).y);
		TI = (W.y * (*R2).x) + (W.x * (*R2).y);
		(*R2).x = TR;
		(*R2).y = TI;
	}

	{
		float2 W = twiddles[3 + 3 * ((1 * me + 0) % 4) + 2];
		float TR, TI;
		TR = (W.x * (*R3).x) - (W.y * (*R3).y);
		TI = (W.y * (*R3).x) + (W.x * (*R3).y);
		(*R3).x = TR;
		(*R3).y = TI;
	}

	FwdRad4B1(R0, R1, R2, R3);

	__syncthreads();

	bufOut[outOffset + (1 * me + 0 + 0) * 1 ] = (*R0);
	bufOut[outOffset + (1 * me + 0 + 4) * 1 ] = (*R1);
	bufOut[outOffset + (1 * me + 0 + 8) * 1 ] = (*R2);
	bufOut[outOffset + (1 * me + 0 + 12) * 1] = (*R3);


}


__global__ void fft_fwd(hipLaunchParm lp, float2 *gb, float2 *twiddles)
{
	uint me = hipThreadIdx_x;

	float2 *lb = gb + 16*hipBlockIdx_x;

	__shared__ float lds[16];

	float2 R0, R1, R2, R3;

	FwdPass0(me, 0, 0, lb, lds, lds, &R0, &R1, &R2, &R3);
	FwdPass1(me, 0, 0, lb, twiddles, &R0, &R1, &R2, &R3);
}


int main()
{

	size_t B = 1048576;
	size_t N = 16;
	float2 twiddles[] = {
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f},
		{9.2387953251128673848313610506011173e-01f, -3.8268343236508978177923268049198668e-01f},
		{7.0710678118654757273731092936941423e-01f, -7.0710678118654757273731092936941423e-01f},
		{3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f},
		{7.0710678118654757273731092936941423e-01f, -7.0710678118654757273731092936941423e-01f},
		{6.1232339957367660358688201472919830e-17f, -1.0000000000000000000000000000000000e+00f},
		{-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f},
		{3.8268343236508983729038391174981371e-01f, -9.2387953251128673848313610506011173e-01f},
		{-7.0710678118654746171500846685376018e-01f, -7.0710678118654757273731092936941423e-01f},
		{-9.2387953251128684950543856757576577e-01f, 3.8268343236508967075693021797633264e-01f},
		{ 1.0000000000000000000000000000000000e+00f, -0.0000000000000000000000000000000000e+00f },
	};

	size_t Nbytes = B * N * sizeof(float2);

	float2 *tw, *x;
	hipMalloc(&tw, N * sizeof(float2));
	hipMalloc(&x, Nbytes);
	hipMemcpy(tw, &twiddles[0], N * sizeof(float2), hipMemcpyHostToDevice);

	float2 *hy = new float2[N*B];
	float2 *hx = new float2[N*B];
	for(size_t j=0; j<B; j++)
	for(size_t i=0; i<N; i++)
	{
		hx[j*N + i].x = i*i - i;
		hx[j*N + i].y = i*10;
	}		


         fftwf_complex *in, *out;
         fftwf_plan p;
         
         in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        
	 for(size_t i=0; i<N; i++)
	 {
		in[i][0] = hx[i].x;
		in[i][1] = hx[i].y;
	 }
 
         fftwf_execute(p); /* repeat as needed */
         
         fftwf_destroy_plan(p);

	std::cout << "fftw output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << out[i][0] << ", " << out[i][1] << std::endl;
	}		


	const unsigned blocks = 1*B;
	const unsigned threadsPerBlock = 4;

	Timer t;
	double elaps = 1000000000.0;
	for(size_t p=0; p<10; p++)
	{
		hipMemcpy(x, hx, Nbytes, hipMemcpyHostToDevice);
		t.Start();
		// Launch HIP kernel
		hipLaunchKernel(HIP_KERNEL_NAME(fft_fwd), dim3(blocks), dim3(threadsPerBlock), 0, 0, x, tw);
		hipDeviceSynchronize();
		double tv = t.Sample();
		elaps = tv < elaps ? tv : elaps;	
	}

	std::cout << "exec time: " << elaps << std::endl;
	std::cout << "gflops: " << 5*B*N*log2(N)/(elaps * 1000000000.0) << std::endl;

	hipMemcpy(hy, x, Nbytes, hipMemcpyDeviceToHost);

	std::cout << "output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << hy[i].x << ", " << hy[i].y << std::endl;
	}		

	double rmse = 0;
	for(size_t i=0; i<N; i++)
	{
		rmse += (hy[i].x - out[i][0])*(hy[i].x - out[i][0]) + (hy[i].y - out[i][1])*(hy[i].y - out[i][1]);
		rmse = sqrt(rmse/N);
	}
	std::cout << "rmse: " << rmse << std::endl;

	delete[] hx;
	delete[] hy;
	
        fftwf_free(in); fftwf_free(out);
	
	hipFree(x);
	hipFree(tw);

	return 0;	
}


