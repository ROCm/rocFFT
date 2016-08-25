
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
	
	switch (N)
	{
	case 1048576:
				hipMalloc(&twt1, 1024*sizeof(float2));
				hipMalloc(&twt2, 1024*sizeof(float2));
				hipMalloc(&twt3, 256*3*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_1024[0], 1024*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_1024[0], 1024*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_1048576[0][0], 256*3*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 524288:
				hipMalloc(&twt1, 1024*sizeof(float2));
				hipMalloc(&twt2, 512*sizeof(float2));
				hipMalloc(&twt3, 256*3*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_1024[0], 1024*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_512[0], 512*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_524288[0][0], 256*3*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 262144:
				hipMalloc(&twt1, 64*sizeof(float2));
				hipMalloc(&twt2, 4096*sizeof(float2));
				hipMalloc(&twt3, 256*3*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_64[0], 64*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_4096[0], 4096*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_262144[0][0], 256*3*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 131072:
				hipMalloc(&twt1, 64*sizeof(float2));
				hipMalloc(&twt2, 2048*sizeof(float2));
				hipMalloc(&twt3, 256*3*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_64[0], 64*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_2048[0], 2048*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_131072[0][0], 256*3*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 65536:
				hipMalloc(&twt1, 256*sizeof(float2));
				hipMalloc(&twt2, 256*sizeof(float2));
				hipMalloc(&twt3, 256*2*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_256[0], 256*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_256[0], 256*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_65536[0][0], 256*2*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 32768:
				hipMalloc(&twt1, 128*sizeof(float2));
				hipMalloc(&twt2, 256*sizeof(float2));
				hipMalloc(&twt3, 256*2*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_128[0], 128*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_256[0], 256*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_32768[0][0], 256*2*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 16384:
				hipMalloc(&twt1, 64*sizeof(float2));
				hipMalloc(&twt2, 256*sizeof(float2));
				hipMalloc(&twt3, 256*2*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_64[0], 64*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_256[0], 256*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_16384[0][0], 256*2*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 8192:
				hipMalloc(&twt1, 64*sizeof(float2));
				hipMalloc(&twt2, 128*sizeof(float2));
				hipMalloc(&twt3, 256*2*sizeof(float2));
				
				*tw1 = twt1;
				*tw2 = twt2;
				*tw3 = twt3;
				
				hipMemcpy(twt1, &twiddles_64[0], 64*sizeof(float2), hipMemcpyHostToDevice); 
				hipMemcpy(twt2, &twiddles_128[0], 128*sizeof(float2), hipMemcpyHostToDevice);
				hipMemcpy(twt3, &twiddle_dee_8192[0][0], 256*2*sizeof(float2), hipMemcpyHostToDevice);
				
				break;
				
	case 4096: 	hipMemcpy(twt, &twiddles_4096[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2048: 	hipMemcpy(twt, &twiddles_2048[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1024: 	hipMemcpy(twt, &twiddles_1024[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 512:  	hipMemcpy(twt, &twiddles_512[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 256:  	hipMemcpy(twt, &twiddles_256[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 128:  	hipMemcpy(twt, &twiddles_128[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 64:   	hipMemcpy(twt, &twiddles_64[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 32:   	hipMemcpy(twt, &twiddles_32[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 16:   	hipMemcpy(twt, &twiddles_16[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 8:    	hipMemcpy(twt, &twiddles_8[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 4:    	hipMemcpy(twt, &twiddles_4[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 2:    	hipMemcpy(twt, &twiddles_2[0], N * sizeof(float2), hipMemcpyHostToDevice); break;
	case 1: 	break;

	default:
		std::cout << "Twiddle error" << std::endl;
		break;				
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

	
	switch (N)
	{
	case 1048576:
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_1048576_1), dim3(16,16*B), dim3(16,16), 0, 0, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_1048576_1<-1>), dim3(1024*B), dim3(128), 0, 0, twiddles1, temp, buffer, count);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_1048576_2<-1>), dim3(16,16*B), dim3(16,16), 0, 0, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_1048576_2<-1>), dim3(1024*B), dim3(128), 0, 0, twiddles2, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_1048576_3), dim3(16,16*B), dim3(16,16), 0, 0, temp, buffer, count);
				break;		
	case 524288:
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_524288_1), dim3(8,16*B), dim3(16,16), 0, 0, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_524288_1<-1>), dim3(512*B), dim3(128), 0, 0, twiddles1, temp, buffer, count);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_524288_2<-1>), dim3(16,8*B), dim3(16,16), 0, 0, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_524288_2<-1>), dim3(1024*B), dim3(64), 0, 0, twiddles2, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_524288_3), dim3(16,8*B), dim3(16,16), 0, 0, temp, buffer, count);
				break;
				
	case 262144:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_4096_bcc_pk<-1>), dim3(256*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_4096_ip_d1_pk<-1>), dim3(64*B), dim3(256), 0, 0, twiddles2, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_262144), dim3(64,B), dim3(16,16), 0, 0, temp, buffer, count);
				break;
	case 131072:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_2048_bcc_pk<-1>), dim3(128*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_2048_ip_d1_pk<-1>), dim3(64*B), dim3(256), 0, 0, twiddles2, temp);
				hipLaunchKernel(HIP_KERNEL_NAME(transpose_131072), dim3(32,B), dim3(16,16), 0, 0, temp, buffer, count);
				break;
				
	case 65536:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_256_bcc_pk<-1>), dim3(32*B), dim3(256), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_256_brc_pk<-1>), dim3(32*B), dim3(256), 0, 0, twiddles2, temp, buffer, count);
				break;		
	case 32768:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_128_256_bcc_pk<-1>), dim3(32*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_128_brc_pk<-1>), dim3(16*B), dim3(256), 0, 0, twiddles2, temp, buffer, count);
				break;		
	case 16384:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_256_bcc_pk<-1>), dim3(16*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_256_64_brc_pk<-1>), dim3(8*B), dim3(256), 0, 0, twiddles2, temp, buffer, count);
				break;		
	case 8192:
				hipLaunchKernel(HIP_KERNEL_NAME(fft_64_128_bcc_pk<-1>), dim3(8*B), dim3(128), 0, 0, twiddles1, twiddles3, buffer, temp, count);
				hipLaunchKernel(HIP_KERNEL_NAME(fft_128_64_brc_pk<-1>), dim3(8*B), dim3(128), 0, 0, twiddles2, temp, buffer, count);
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


