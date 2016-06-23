
#define __HIPCC__

#include <iostream>
#include <hip_runtime.h>
#include <fftw3.h>

#include <time.h>
#include <math.h>

#include "twiddles_4096.h"
#include "fft_pow2_hip.h"

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


void LaunchKernel(size_t N, unsigned blocks, unsigned threadsPerBlock, float2 *twiddles, float2 *buffer, unsigned count, int dir)
{
	switch (N)
	{
	//case 4096:	hipLaunchKernel(HIP_KERNEL_NAME(fft_4096), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 2048:	hipLaunchKernel(HIP_KERNEL_NAME(fft_2048), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 1024:	hipLaunchKernel(HIP_KERNEL_NAME(fft_1024), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 512:	hipLaunchKernel(HIP_KERNEL_NAME(fft_512), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 256:	hipLaunchKernel(HIP_KERNEL_NAME(fft_256), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 128:	hipLaunchKernel(HIP_KERNEL_NAME(fft_128), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 64:	hipLaunchKernel(HIP_KERNEL_NAME(fft_64), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 32:	hipLaunchKernel(HIP_KERNEL_NAME(fft_32), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 16:	hipLaunchKernel(HIP_KERNEL_NAME(fft_16), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 8:		hipLaunchKernel(HIP_KERNEL_NAME(fft_8), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 4:		hipLaunchKernel(HIP_KERNEL_NAME(fft_4), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 2:		hipLaunchKernel(HIP_KERNEL_NAME(fft_2), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;
	case 1:		hipLaunchKernel(HIP_KERNEL_NAME(fft_1), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count, dir); break;

	default:
		std::cout << "Launch error" << std::endl;
		break;
	}

		
}

int main(int argc, char **argv)
{

	if(argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " batch_size transform_size" << std::endl;
		return -1;
	}

	size_t B = atoi(argv[1]);
	size_t N = atoi(argv[2]);

	float2 twiddles[] = {
		TWIDDLE_4096
	};

	size_t Nbytes = B * N * sizeof(float2);

	float2 *tw, *x;
	hipMalloc(&tw, 4096 * sizeof(float2));
	hipMalloc(&x, Nbytes);
	hipMemcpy(tw, &twiddles[0], 4096 * sizeof(float2), hipMemcpyHostToDevice);

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

#if 0
	std::cout << "fftw output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << out[i][0] << ", " << out[i][1] << std::endl;
	}		
#endif

	size_t WGS = 64;
	size_t NT = 1;

	switch (N)
	{
	case 4096:	WGS = 256; NT = 1; break;
	case 2048:	WGS = 256; NT = 1; break;
	case 1024:	WGS = 128; NT = 1; break;
	case 512:	WGS = 64;  NT = 1; break;
	case 256:	WGS = 64;  NT = 1; break;
	case 128:	WGS = 64;  NT = 4; break;
	case 64:	WGS = 64;  NT = 4; break;
	case 32:	WGS = 64; NT = 16; break;
	case 16:	WGS = 64; NT = 16; break;
	case 8:		WGS = 64; NT = 32; break;
	case 4:		WGS = 64; NT = 32; break;
	case 2:		WGS = 64; NT = 64; break;
	case 1:		WGS = 64; NT = 64; break;

	default:
		break;
	}


	const unsigned blocks = (B%NT) ? 1 + (B / NT) : (B / NT);
	const unsigned threadsPerBlock = WGS;

	Timer t;
	double elaps = 1000000000.0;
	for(size_t p=0; p<10; p++)
	{
		hipMemcpy(x, hx, Nbytes, hipMemcpyHostToDevice);
		t.Start();
		// Launch HIP kernel
		LaunchKernel(N, blocks, threadsPerBlock, tw, x, B, -1);
		hipDeviceSynchronize();
		double tv = t.Sample();
		elaps = tv < elaps ? tv : elaps;	
	}

	std::cout << "exec time: " << elaps << std::endl;
	std::cout << "gflops: " << 5*B*N*log2(N)/(elaps * 1000000000.0) << std::endl;

	hipMemcpy(hy, x, Nbytes, hipMemcpyDeviceToHost);

#if 0
	std::cout << "output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << hy[i].x << ", " << hy[i].y << std::endl;
	}		
#endif

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


