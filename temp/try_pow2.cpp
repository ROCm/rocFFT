/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#define __HIPCC__

#include <iostream>
#include <hip/hip_runtime.h>
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

void CreateAndCopyTwiddles(float2 **tw, float2 **tw1, float2 **tw2, float2 **tw3, size_t N)
{
#include "twiddles_pow2.h"
#include "twiddles_pow2_large.h"
	
	float2 *twt1, *twt2, *twt3;	
	
	float2 *twt;
	if(N <= 4096)
	{
		hipMalloc(&twt, N * sizeof(float2));
		*tw = twt;
	}

	const void *twtc1, *twtc2, *twtc3;
	size_t n1, n2, n3;
	
	switch (N)
	{
	case 16777216:
				n1 = 4096;
				n2 = 4096;
				n3 = 256*3;
				
				twtc1 = &twiddles_4096[0];
				twtc2 = &twiddles_4096[0];
				twtc3 = &twiddle_dee_16777216[0][0];
				
				break;
				
	case 8388608:
				n1 = 4096;
				n2 = 2048;
				n3 = 256*3;
				
				twtc1 = &twiddles_4096[0];
				twtc2 = &twiddles_2048[0];
				twtc3 = &twiddle_dee_8388608[0][0];
				
				break;
				
	case 4194304:
				n1 = 2048;
				n2 = 2048;
				n3 = 256*3;
				
				twtc1 = &twiddles_2048[0];
				twtc2 = &twiddles_2048[0];
				twtc3 = &twiddle_dee_4194304[0][0];
				
				break;
				
	case 2097152:
				n1 = 2048;
				n2 = 1024;
				n3 = 256*3;
				
				twtc1 = &twiddles_2048[0];
				twtc2 = &twiddles_1024[0];
				twtc3 = &twiddle_dee_2097152[0][0];
				
				break;
				
	case 1048576:
				n1 = 1024;
				n2 = 1024;
				n3 = 256*3;
				
				twtc1 = &twiddles_1024[0];
				twtc2 = &twiddles_1024[0];
				twtc3 = &twiddle_dee_1048576[0][0];
				
				break;
				
	case 524288:
				n1 = 1024;
				n2 = 512;
				n3 = 256*3;
				
				twtc1 = &twiddles_1024[0];
				twtc2 = &twiddles_512[0];
				twtc3 = &twiddle_dee_524288[0][0];
				
				break;
				
	case 262144:
				n1 = 64;
				n2 = 4096;
				n3 = 256*3;
				
				twtc1 = &twiddles_64[0];
				twtc2 = &twiddles_4096[0];
				twtc3 = &twiddle_dee_262144[0][0];
				
				break;
				
	case 131072:
				n1 = 64;
				n2 = 2048;
				n3 = 256*3;
				
				twtc1 = &twiddles_64[0];
				twtc2 = &twiddles_2048[0];
				twtc3 = &twiddle_dee_131072[0][0];
				
				break;
				
	case 65536:
				n1 = 256;
				n2 = 256;
				n3 = 256*2;
				
				twtc1 = &twiddles_256[0];
				twtc2 = &twiddles_256[0];
				twtc3 = &twiddle_dee_65536[0][0];
				
				break;
				
	case 32768:
				n1 = 128;
				n2 = 256;
				n3 = 256*2;
				
				twtc1 = &twiddles_128[0];
				twtc2 = &twiddles_256[0];
				twtc3 = &twiddle_dee_32768[0][0];
				
				break;
				
	case 16384:
				n1 = 64;
				n2 = 256;
				n3 = 256*2;
				
				twtc1 = &twiddles_64[0];
				twtc2 = &twiddles_256[0];
				twtc3 = &twiddle_dee_16384[0][0];
				
				break;
				
	case 8192:
				n1 = 64;
				n2 = 128;
				n3 = 256*2;
				
				twtc1 = &twiddles_64[0];
				twtc2 = &twiddles_128[0];
				twtc3 = &twiddle_dee_8192[0][0];
				
				break;
	default:
				break;
	}
	
	if(N > 4096)			
	{
		hipMalloc(&twt1, n1*sizeof(float2));
		hipMalloc(&twt2, n2*sizeof(float2));
		hipMalloc(&twt3, n3*sizeof(float2));
		
		*tw1 = twt1;
		*tw2 = twt2;
		*tw3 = twt3;
		
		hipMemcpy(twt1, twtc1, n1*sizeof(float2), hipMemcpyHostToDevice); 
		hipMemcpy(twt2, twtc2, n2*sizeof(float2), hipMemcpyHostToDevice);
		hipMemcpy(twt3, twtc3, n3*sizeof(float2), hipMemcpyHostToDevice);
	}
	else
	{			
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
	
		default:
			std::cout << "Twiddle error" << std::endl;
			break;				
		}
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

void LaunchKernel(size_t N, float2 *twiddles, float2 *twiddles1, float2 *twiddles2, float2 *twiddles3, float2 *temp, float2 *buffer, unsigned count, int dir)
{
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

	const unsigned B = count;
	const unsigned blocks = (B%NT) ? 1 + (B / NT) : (B / NT);
	const unsigned threadsPerBlock = WGS;

	unsigned Ng, Ns;
	
	switch(N)
	{
	case 16777216:		Ng = 4096; Ns = 4096; break;
	case 8388608:		Ng = 4096; Ns = 2048; break;
	case 4194304:		Ng = 2048; Ns = 2048; break;
	case 2097152:		Ng = 2048; Ns = 1024; break;
	case 1048576:		Ng = 1024; Ns = 1024; break;
	case  524288:		Ng = 1024; Ns =  512; break; 
	}

	switch (N)
	{
	case 16777216:
	case 8388608:
	case 4194304:
	case 2097152:
	case 1048576:
	case 524288:
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_HOR>), dim3(Ns/64,(Ng/64)*B), dim3(16,16), 0, 0, 0, buffer, temp, (Ng/64), Ns, Ng+64, N, Ns*(Ng+64));
				LaunchKernel2(Ng, twiddles1, temp, buffer, B, Ns, Ng+64, Ng, Ns*(Ng+64), N, dir);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,3,TTD_IP_HOR>), dim3(Ng/64,(Ns/64)*B), dim3(16,16), 0, 0, twiddles3, buffer, temp, (Ns/64), Ng, Ns+64, N, Ng*(Ns+64));
				LaunchKernel2(Ns, twiddles2, temp, temp, B, Ng, Ns+64, Ns+64, Ng*(Ns+64), Ng*(Ns+64), dir);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_VER>), dim3(Ng/64,(Ns/64)*B), dim3(16,16), 0, 0, 0, temp, buffer, (Ns/64), Ns+64, Ng, Ng*(Ns+64), N);

				break;
				
	case 262144:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_4096_bcc_d1_pk<-1>), dim3(256*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_4096_ip_d1_pk<-1>), dim3(64*B), dim3(256), 0, 0, twiddles2, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_HOR>), dim3(64,B), dim3(16,16), 0, 0, 0, temp, buffer, 1, 4096, 64, 262144, 262144);
				break;
	case 131072:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_2048_bcc_d1_pk<-1>), dim3(128*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_2048_ip_d1_pk<-1>), dim3(64*B), dim3(256), 0, 0, twiddles2, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_var1<-1,0,TTD_IP_HOR>), dim3(32,B), dim3(16,16), 0, 0, 0, temp, buffer, 1, 2048, 64, 131072, 131072);
				break;
				
	case 65536:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_256_bcc_d1_pk<-1>), dim3(32*B), dim3(256), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_256_brc_d1_pk<-1>), dim3(32*B), dim3(256), 0, 0, twiddles2, temp, buffer);
				break;		
	case 32768:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_128_256_bcc_d1_pk<-1>), dim3(32*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_128_brc_d1_pk<-1>), dim3(16*B), dim3(256), 0, 0, twiddles2, temp, buffer);
				break;		
	case 16384:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_256_bcc_d1_pk<-1>), dim3(16*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_64_brc_d1_pk<-1>), dim3(8*B), dim3(256), 0, 0, twiddles2, temp, buffer);
				break;		
	case 8192:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_128_bcc_d1_pk<-1>), dim3(8*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_128_64_brc_d1_pk<-1>), dim3(8*B), dim3(128), 0, 0, twiddles2, temp, buffer);
				break;
		
	case 4096:	hipLaunchKernel(HIP_KERNEL_NAME(fft_4096_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer); break;
	case 2048:	hipLaunchKernel(HIP_KERNEL_NAME(fft_2048_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer); break;
	case 1024:	hipLaunchKernel(HIP_KERNEL_NAME(fft_1024_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer); break;
	case 512:	hipLaunchKernel(HIP_KERNEL_NAME( fft_512_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer); break;
	case 256:	hipLaunchKernel(HIP_KERNEL_NAME( fft_256_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer); break;

	case 128:	hipLaunchKernel(HIP_KERNEL_NAME(fft_128_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 64:	hipLaunchKernel(HIP_KERNEL_NAME( fft_64_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 32:	hipLaunchKernel(HIP_KERNEL_NAME( fft_32_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 16:	hipLaunchKernel(HIP_KERNEL_NAME( fft_16_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 8:		hipLaunchKernel(HIP_KERNEL_NAME(  fft_8_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 4:		hipLaunchKernel(HIP_KERNEL_NAME(  fft_4_ip_d1_pk<-1>), dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 2:		hipLaunchKernel(HIP_KERNEL_NAME(  fft_2_ip_d1_pk),     dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;
	case 1:		hipLaunchKernel(HIP_KERNEL_NAME(  fft_1_ip_d1_pk),     dim3(blocks), dim3(threadsPerBlock), 0, 0, twiddles, buffer, count); break;

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

	size_t Nbytes = B * N * sizeof(float2);

	float2 *tw = 0;
	float2 *tw1 = 0, *tw2 = 0, *tw3 = 0;	
	CreateAndCopyTwiddles(&tw, &tw1, &tw2, &tw3, N);
	
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
         p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        
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
		LaunchKernel(N, tw, tw1, tw2, tw3, temp, x, B, -1);
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
	
	if(tw != 0)
		hipFree(tw);

	if(tw1 != 0)
		hipFree(tw1);
	if(tw2 != 0)
		hipFree(tw2);
	if(tw3 != 0)
		hipFree(tw3);
	
	return 0;	
}


