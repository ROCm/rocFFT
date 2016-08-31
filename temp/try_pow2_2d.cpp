
#define __HIPCC__

#include <iostream>
#include <hip_runtime.h>
#include <fftw3.h>

#include <time.h>
#include <math.h>

#include "fft_pow2_ip_entry_hip.h"
#include "fft_pow2_op_entry_hip.h"
#include "fft_pow2_large_entry_hip.h"

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

void CreateAndCopyTwiddles(float2 **tw1, float2 **tw2, size_t N0, size_t N1)
{
#include "twiddles_pow2.h"
	
	float2 *twt1, *twt2;	

	if(N0 <= 4096)
	{
		hipMalloc(&twt1, N0 * sizeof(float2));
		*tw1 = twt1;
	}

	if(N1 <= 4096)
	{
		hipMalloc(&twt2, N1 * sizeof(float2));
		*tw2 = twt2;
	}

	
	switch (N0)
	{
	case 4096: 	hipMemcpy(twt1, &twiddles_4096[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2048: 	hipMemcpy(twt1, &twiddles_2048[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1024: 	hipMemcpy(twt1, &twiddles_1024[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 512:  	hipMemcpy(twt1,  &twiddles_512[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 256:  	hipMemcpy(twt1,  &twiddles_256[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 128:  	hipMemcpy(twt1,  &twiddles_128[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 64:   	hipMemcpy(twt1,   &twiddles_64[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 32:   	hipMemcpy(twt1,   &twiddles_32[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 16:   	hipMemcpy(twt1,   &twiddles_16[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 8:    	hipMemcpy(twt1,    &twiddles_8[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 4:    	hipMemcpy(twt1,    &twiddles_4[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2:    	hipMemcpy(twt1,    &twiddles_2[0], N0 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1: 	break;

	default:
		std::cout << "Twiddle error" << std::endl;
		break;				
	}

	switch (N1)
	{
	case 4096: 	hipMemcpy(twt2, &twiddles_4096[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2048: 	hipMemcpy(twt2, &twiddles_2048[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1024: 	hipMemcpy(twt2, &twiddles_1024[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 512:  	hipMemcpy(twt2,  &twiddles_512[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 256:  	hipMemcpy(twt2,  &twiddles_256[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 128:  	hipMemcpy(twt2,  &twiddles_128[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 64:   	hipMemcpy(twt2,   &twiddles_64[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 32:   	hipMemcpy(twt2,   &twiddles_32[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 16:   	hipMemcpy(twt2,   &twiddles_16[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 8:    	hipMemcpy(twt2,    &twiddles_8[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 4:    	hipMemcpy(twt2,    &twiddles_4[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2:    	hipMemcpy(twt2,    &twiddles_2[0], N1 * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1: 	break;

	default:
		std::cout << "Twiddle error" << std::endl;
		break;				
	}
}



void LaunchKernel2(size_t N, float2 *twiddles, float2 *in, float2 *out, unsigned B, unsigned len, unsigned stride_i, unsigned stride_o, unsigned dist_i, unsigned dist_o, int dir)
{
	if(in == out)
	{
	switch(N)
	{
	case 4096:	hipLaunchKernel(HIP_KERNEL_NAME(fft_4096_ip_d2_s1<-1>), dim3(len*B), dim3(256), 0, 0, twiddles, in, len, stride_i, dist_i); break;
	case 2048:	hipLaunchKernel(HIP_KERNEL_NAME(fft_2048_ip_d2_s1<-1>), dim3(len*B), dim3(256), 0, 0, twiddles, in, len, stride_i, dist_i); break;
	case 1024:	hipLaunchKernel(HIP_KERNEL_NAME(fft_1024_ip_d2_s1<-1>), dim3(len*B), dim3(128), 0, 0, twiddles, in, len, stride_i, dist_i); break;
	case  512:	hipLaunchKernel(HIP_KERNEL_NAME( fft_512_ip_d2_s1<-1>), dim3(len*B), dim3( 64), 0, 0, twiddles, in, len, stride_i, dist_i); break;
	default:	std::cout << "code path error" << std::endl; break;
	}
	}
	else
	{
	switch(N)
	{
	case 4096:	hipLaunchKernel(HIP_KERNEL_NAME(fft_4096_op_d2_s1<-1>), dim3(len*B), dim3(256), 0, 0, twiddles, in, out, len, stride_i, stride_o, dist_i, dist_o); break;
	case 2048:	hipLaunchKernel(HIP_KERNEL_NAME(fft_2048_op_d2_s1<-1>), dim3(len*B), dim3(256), 0, 0, twiddles, in, out, len, stride_i, stride_o, dist_i, dist_o); break;
	case 1024:	hipLaunchKernel(HIP_KERNEL_NAME(fft_1024_op_d2_s1<-1>), dim3(len*B), dim3(128), 0, 0, twiddles, in, out, len, stride_i, stride_o, dist_i, dist_o); break;
	case  512:	hipLaunchKernel(HIP_KERNEL_NAME( fft_512_op_d2_s1<-1>), dim3(len*B), dim3( 64), 0, 0, twiddles, in, out, len, stride_i, stride_o, dist_i, dist_o); break;
	default:	std::cout << "code path error" << std::endl; break;
	}
	}
}

void LaunchKernel(size_t N0, size_t N1, float2 *twiddles1, float2 *twiddles2, float2 *temp, float2 *buffer, unsigned count, int dir)
{
	unsigned B = count;

	LaunchKernel2(N0, twiddles1, buffer, buffer, B, N1, N0, N0, N0*N1, N0*N1, dir);
	hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_HOR>), dim3(N0/64,(N1/64)*B), dim3(16,16), 0, 0, 0, buffer, temp, (N1/64), N0, N1+64, N0*N1, N0*(N1+64));
	LaunchKernel2(N1, twiddles2, temp, temp, B, N0, N1+64, N1+64, N0*(N1+64), N0*(N1+64), dir);
	hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_VER>), dim3(N0/64,(N1/64)*B), dim3(16,16), 0, 0, 0, temp, buffer, (N1/64), N1+64, N0, N0*(N1+64), N0*N1);

}

int main(int argc, char **argv)
{

	if(argc != 4)
	{
		std::cout << "Usage: " << argv[0] << " batch_size N0 N1" << std::endl;
		return -1;
	}

	size_t B = atoi(argv[1]);
	size_t N0 = atoi(argv[2]);
	size_t N1 = atoi(argv[3]);
	size_t N = N0 * N1;

	size_t Nbytes = B * N * sizeof(float2);

	float2 *tw1 = 0, *tw2 = 0;	
	CreateAndCopyTwiddles(&tw1, &tw2, N0, N1);
	
	float2 *x = 0, *temp = 0;
	hipMalloc(&x, Nbytes);	
	hipMalloc(&temp, 2*Nbytes);

	float2 *hy = new float2[N*B];
	float2 *hx = new float2[N*B];
	float2 *ref = new float2[N*B];

	for(size_t j=0; j<B; j++)
	{
	for(size_t i=0; i<N; i++)
	{
		hx[j*N + i].x = (rand() % 2) == 0 ? (float)(rand() % 17) : -(float)(rand() % 17);
		hx[j*N + i].y = (rand() % 2) == 0 ? (float)(rand() % 17) : -(float)(rand() % 17);
		//hx[j*N + i].x = i*i - i;
		//hx[j*N + i].y = i*10;
	}
	}		


         fftwf_complex *in, *out;
         fftwf_plan p;
         
         in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N );
         out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N );
         p = fftwf_plan_dft_2d(N1, N0, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        
	 for(size_t j=0; j<B; j++)
	 {
		 for(size_t i=0; i<N; i++)
		 {
			in[i][0] = hx[j*N + i].x;
			in[i][1] = hx[j*N + i].y;
		 }

	         fftwf_execute(p); /* repeat as needed */

		 for(size_t i=0; i<N; i++)
		 {
			ref[j*N + i].x = out[i][0];
			ref[j*N + i].y = out[i][1];
		 }
	 }
         
         fftwf_destroy_plan(p);

#if 0
	std::cout << "fftw output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << out[i][0] << ", " << out[i][1] << std::endl;
	}		
#endif



	Timer t;
	double elaps = 1e10;
	double gpu_time = 1e10;
	for(size_t p=0; p<10; p++)
	{
		hipMemcpy(x, hx, Nbytes, hipMemcpyHostToDevice);

		hipEvent_t start, stop;
		hipEventCreate(&start);
		hipEventCreate(&stop);

		t.Start();
		// Launch HIP kernel
		hipEventRecord(start);
		LaunchKernel(N0, N1, tw1, tw2, temp, x, B, -1);
		hipEventRecord(stop);
		hipEventSynchronize(stop);
		double tv = t.Sample();

		elaps = tv < elaps ? tv : elaps;

		float ms;
		hipEventElapsedTime(&ms, start, stop);
		gpu_time = ms < gpu_time ? ms : gpu_time;

		hipEventDestroy(start);
		hipEventDestroy(stop);	
	}

	std::cout << "exec time (milliseconds): " << elaps*1000.0 << std::endl;
	std::cout << "gpu exec time (milliseconds): " << gpu_time << std::endl;
	std::cout << "gflops: " << 5*B*N*log2(N)/(gpu_time * 1000000.0) << std::endl;

	hipMemcpy(hy, x, Nbytes, hipMemcpyDeviceToHost);

#if 0
	std::cout << "output: " << std::endl;
	for(size_t i=0; i<N; i++)
	{
		std::cout << hy[i].x << ", " << hy[i].y << std::endl;
	}		
#endif


	double rmse_max = 0;
	for (size_t j = 0; j < B; j++)
	{
		double rmse = 0;
		double maxv = 0;

		for (size_t i = 0; i < N; i++)
		{
			maxv = maxv > fabs(ref[j*N + i].x) ? maxv : fabs(ref[j*N + i].x);
			maxv = maxv > fabs(ref[j*N + i].y) ? maxv : fabs(ref[j*N + i].y);
		}

		for (size_t i = 0; i < N; i++)
		{
			rmse += (hy[j*N + i].x - ref[j*N + i].x)*(hy[j*N + i].x - ref[j*N + i].x);
			rmse += (hy[j*N + i].y - ref[j*N + i].y)*(hy[j*N + i].y - ref[j*N + i].y);
		}

		rmse = sqrt((rmse / maxv) / N);
		rmse_max = rmse > rmse_max ? rmse : rmse_max;
	}

	std::cout << "rrmse: " << rmse_max << std::endl;
	
	

	delete[] hx;
	delete[] hy;
	delete[] ref;
	
        fftwf_free(in); fftwf_free(out);
	
	hipFree(x);
	hipFree(temp);
	

	if(tw1 != 0)
		hipFree(tw1);
	if(tw2 != 0)
		hipFree(tw2);
	
	return 0;	
}


