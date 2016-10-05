/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "./common.h"

#define NEAR_ZERO_TOLERATE true



#define FORMAT_INTERLEAVED
//#define DoublePrecision
//#define LIST_RESULT
#define FFT_FWD
//#define SEE_HEX
#define BUILD_LOG_SIZE (1024 * 1024)
#define CHECK_RESULT


int main(int argc, char ** argv)
{
	// arg list
	// filename, N, T, B

	char **ps;
	size_t flines;

	size_t B = atoi(argv[2]);
	size_t N = atoi(argv[3]);

	// 1. Get a platform.
	cl_platform_id platform;
	clGetPlatformIDs( 1, &platform, NULL );


	// 2. Find a gpu device.
	cl_device_id device;
	clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	// 3. Create a context and command queue on that device.
	cl_context context = clCreateContext( NULL, 1, &device,	NULL, NULL, NULL);

	cl_command_queue queue = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, NULL );

	// 4. Perform runtime source compilation, and obtain kernel entry point.
	flines = file_lines_get(argv[1], &ps);
	cl_program program = clCreateProgramWithSource( context, (cl_uint)flines, (const char **)ps, NULL, NULL );
	file_lines_clear(flines, ps);
	cl_int err = clBuildProgram( program, 1, &device, "-I.", NULL, NULL );
	if(err != CL_SUCCESS)
	{
		char *build_log = new char[BUILD_LOG_SIZE];
		size_t log_sz;
		fprintf(stderr, "build program failed, err=%d\n", err);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, BUILD_LOG_SIZE, build_log, &log_sz);
		if (err != CL_SUCCESS)
			fprintf(stderr, "failed to get build log, err=%d\n", err);
		else
			fprintf(stderr, "----- Build Log -----\n%s\n----- ----- --- -----\n", build_log);
		delete[] build_log;
		clReleaseProgram(program);
		return -1;

	}

	const char *KERN_NAME = NULL;

	const char *KERN_NAME_1 = NULL;
	const char *KERN_NAME_2 = NULL;
	const char *KERN_NAME_3 = NULL;
	const char *KERN_NAME_4 = NULL;
	const char *KERN_NAME_5 = NULL;

	switch (N)
	{
	case 1048576:	KERN_NAME_1 = "transpose_1048576_1";
					KERN_NAME_2 = "fft_1048576_1";
					KERN_NAME_3 = "transpose_1048576_2";
					KERN_NAME_4 = "fft_1048576_2";
					KERN_NAME_5 = "transpose_1048576_3";
					break;
	case 524288:	KERN_NAME_1 = "transpose_524288_1";
					KERN_NAME_2 = "fft_524288_1";
					KERN_NAME_3 = "transpose_524288_2";
					KERN_NAME_4 = "fft_524288_2";
					KERN_NAME_5 = "transpose_524288_3";
					break;
	case 262144:	KERN_NAME_1 = "fft_262144_1";
					KERN_NAME_2 = "fft_262144_2";
					KERN_NAME_3 = "transpose_262144";
					break;
	case 131072:	KERN_NAME_1 = "fft_131072_1";
					KERN_NAME_2 = "fft_131072_2";
					KERN_NAME_3 = "transpose_131072";
					break;
	case 65536:	KERN_NAME_1 = "fft_65536_1";
				KERN_NAME_2 = "fft_65536_2";
				break;
	case 32768:	KERN_NAME_1 = "fft_32768_1";
				KERN_NAME_2 = "fft_32768_2";
				break;
	case 16384:	KERN_NAME_1 = "fft_16384_1";
				KERN_NAME_2 = "fft_16384_2";
				break;
	case 8192:	KERN_NAME_1 = "fft_8192_1";
				KERN_NAME_2 = "fft_8192_2";
				break;
	case 4096: KERN_NAME = "fft_4096"; break;
	case 2048: KERN_NAME = "fft_2048"; break;
	case 1024: KERN_NAME = "fft_1024"; break;
	case 512:  KERN_NAME = "fft_512";  break;
	case 256:  KERN_NAME = "fft_256";  break;
	case 128:  KERN_NAME = "fft_128";  break;
	case 64:   KERN_NAME = "fft_64";   break;
	case 32:   KERN_NAME = "fft_32";   break;
	case 16:   KERN_NAME = "fft_16";   break;
	case 8:    KERN_NAME = "fft_8";    break;
	case 4:    KERN_NAME = "fft_4";    break;
	case 2:    KERN_NAME = "fft_2";    break;
	case 1:    KERN_NAME = "fft_1";    break;
	}

	cl_kernel kernel;
	cl_kernel kernel_1;
	cl_kernel kernel_2;
	cl_kernel kernel_3;
	cl_kernel kernel_4;
	cl_kernel kernel_5;

	switch (N)
	{
	case 1048576:
	case 524288:
				kernel_5 = clCreateKernel(program, KERN_NAME_5, NULL);
				kernel_4 = clCreateKernel(program, KERN_NAME_4, NULL);
	case 262144:
	case 131072:
				kernel_3 = clCreateKernel(program, KERN_NAME_3, NULL);
	case 65536:
	case 32768:
	case 16384:
	case 8192:	
				kernel_1 = clCreateKernel(program, KERN_NAME_1, NULL);
				kernel_2 = clCreateKernel(program, KERN_NAME_2, NULL);
				break;
	case 4096: 
	case 2048: 
	case 1024: 
	case 512:  
	case 256:  
	case 128:  
	case 64:   
	case 32:   
	case 16:   
	case 8:    
	case 4:    
	case 2:    
	case 1:    kernel = clCreateKernel(program, KERN_NAME, NULL);
	}
	// Start FFT

	Type *yr, *yi, *xr, *xi, *refr, *refi;
	Type *xc, *yc;
	yr = new Type [N*B];
	yi = new Type [N*B];
	xr = new Type [N*B];
	xi = new Type [N*B];
	refr = new Type [N*B];
	refi = new Type [N*B];
	xc = new Type [2*N*B];
	yc = new Type [2*N*B];

	for(uint i=0; i<N*B; i++)
	{

		xr[i] = (rand() % 2) == 0 ? (Type)(rand() % 17) : -(Type)(rand() % 17);
		xi[i] = (rand() % 2) == 0 ? (Type)(rand() % 17) : -(Type)(rand() % 17);
		xc[2*i] = xr[i];
		xc[2*i + 1] = xi[i];
	}

#ifdef LIST_RESULT
	std::cout << "**** INPUT ****" << std::endl;
	for(uint j=0; j<B; j++)
	{
		for(uint i=0; i<N; i++)
		{
			std::cout << "(" << xr[j*N + i] << ", " << xi[j*N + i] << ") " << std::endl;
		}
		std::cout << "=======================BATCH ENDS====================" << std::endl;
	}
#endif

	// Compute reference FFT
	FftwComplex *in, *out;
	Type *ind, *outd;
	FftwPlan p;
	in = (FftwComplex*)FftwMalloc(sizeof(FftwComplex) * N);
	out = (FftwComplex*)FftwMalloc(sizeof(FftwComplex) * N);
	ind = (Type *)in;
	outd = (Type *)out;

#ifdef FFT_FWD
	p = FftwPlanFn((int)N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
#else
	p = FftwPlanFn(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

	for(uint j=0; j<B; j++)
	{
		for(uint i=0; i<N; i++)
		{
			ind[2*i] = xr[j*N + i];
			ind[2*i + 1] = xi[j*N + i];
		}

		FftwExecute(p);

		for(uint i=0; i<N; i++)
		{
			refr[j*N + i] = outd[2*i];
			refi[j*N + i] = outd[2*i + 1];
		}
	}
	
#ifdef LIST_RESULT
	std::cout << "**** REF ****" << std::endl;
	for (uint j = 0; j < B; j++)
	{
		for (uint i = 0; i < N; i++)
		{
#ifndef SEE_HEX
			std::cout << "(" << refr[j*N + i] << ", " << refi[j*N + i] << ") " << std::endl;
#else
			Utype rv, iv;
			rv.f = refr[j*N + i]; iv.f = refi[j*N + i];
			printf("(%0x, %0x)\n", rv.u, iv.u);
#endif
		}
		std::cout << "=======================BATCH ENDS====================" << std::endl;
	}
#endif


#ifndef FORMAT_INTERLEAVED
	cl_mem bufferReal = clCreateBuffer(context, CL_MEM_READ_WRITE, N*B * sizeof(ClType), NULL, NULL);
	cl_mem bufferImag = clCreateBuffer(context, CL_MEM_READ_WRITE, N*B * sizeof(ClType), NULL, NULL);
#else
	cl_mem bufferCplx = clCreateBuffer(context, CL_MEM_READ_WRITE, N*B * sizeof(ClType2), NULL, NULL);
	cl_mem bufferTemp = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*N*B * sizeof(ClType2), NULL, NULL);
#endif

	// cl_mem dbg = clCreateBuffer(context, CL_MEM_READ_WRITE, N*B * sizeof(ClType2), NULL, NULL);

	// Fill buffers for FFT
#ifndef FORMAT_INTERLEAVED
	clEnqueueWriteBuffer(queue, bufferReal, CL_TRUE, 0, N*B * sizeof(ClType), (void *)xr, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, bufferImag, CL_TRUE, 0, N*B * sizeof(ClType), (void *)xi, 0, NULL, NULL);
#else
	clEnqueueWriteBuffer(queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL);
#endif


	cl_uint k_count = (cl_uint)B;
	cl_int dir = -1;

#ifndef FORMAT_INTERLEAVED
	clSetKernelArg(kernel, 0, sizeof(bufferReal), (void*)&bufferReal);
	clSetKernelArg(kernel, 1, sizeof(bufferImag), (void*)&bufferImag);
#else
	switch (N)
	{
	case 1048576:
	case 524288:
				clSetKernelArg(kernel_1, 0, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_1, 1, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_1, 2, sizeof(cl_uint), &k_count);

				clSetKernelArg(kernel_2, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_2, 1, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_2, 2, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_2, 3, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_3, 0, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_3, 1, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_3, 2, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_3, 3, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_4, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_4, 1, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_4, 2, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_5, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_5, 1, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_5, 2, sizeof(cl_uint), &k_count);
		break;
	case 262144:
	case 131072:
				clSetKernelArg(kernel_1, 0, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_1, 1, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_1, 2, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_1, 3, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_2, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_2, 1, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_2, 2, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_3, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_3, 1, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_3, 2, sizeof(cl_uint), &k_count);
		break;
	case 65536:
	case 32768:
	case 16384:
	case 8192:	
				clSetKernelArg(kernel_1, 0, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_1, 1, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_1, 2, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_1, 3, sizeof(cl_int), &dir);

				clSetKernelArg(kernel_2, 0, sizeof(bufferTemp), (void*)&bufferTemp);
				clSetKernelArg(kernel_2, 1, sizeof(bufferCplx), (void*)&bufferCplx);
				clSetKernelArg(kernel_2, 2, sizeof(cl_uint), &k_count);
				clSetKernelArg(kernel_2, 3, sizeof(cl_int), &dir);
		break;
	case 4096:
	case 2048:
	case 1024:
	case 512:
	case 256:
	case 128:
	case 64:
	case 32:
	case 16:
	case 8:
	case 4:
	case 2:
	case 1:    clSetKernelArg(kernel, 0, sizeof(bufferCplx), (void*)&bufferCplx);
	}

#endif

	size_t WGS = 64;
	size_t NT = 1;

	switch (N)
	{
	case 4096:	WGS = 256; NT =  1; break;
	case 2048:	WGS = 256; NT =  1; break;
	case 1024:	WGS = 128; NT =  1; break;
	case 512:	WGS =  64; NT =  1; break;
	case 256:	WGS =  64; NT =  1; break;
	case 128:	WGS =  64; NT =  4; break;
	case 64:	WGS =  64; NT =  4; break;
	case 32:	WGS =  64; NT = 16; break;
	case 16:	WGS =  64; NT = 16; break;
	case 8:		WGS =  64; NT = 32; break;
	case 4:		WGS =  64; NT = 32; break;
	case 2:		WGS =  64; NT = 64; break;
	case 1:		WGS =  64; NT = 64; break;

	default:
		break;
	}


	// 6. Launch the kernel
	size_t global_work_size[3];
	size_t local_work_size[3];

	size_t global_work_size_1[3];
	size_t local_work_size_1[3];
	size_t global_work_size_2[3];
	size_t local_work_size_2[3];
	size_t global_work_size_3[3];
	size_t local_work_size_3[3];
	size_t global_work_size_4[3];
	size_t local_work_size_4[3];
	size_t global_work_size_5[3];
	size_t local_work_size_5[3];

	switch (N)
	{
	case 1048576:
	{
		local_work_size_1[0] = 16;
		local_work_size_1[1] = 16;
		global_work_size_1[0] = local_work_size_1[0] * 16;
		global_work_size_1[1] = local_work_size_1[1] * 16 * B;

		local_work_size_2[0] = 128;
		global_work_size_2[0] = local_work_size_2[0] * 1024 * B;

		local_work_size_3[0] = 16;
		local_work_size_3[1] = 16;
		global_work_size_3[0] = local_work_size_3[0] * 16;
		global_work_size_3[1] = local_work_size_3[1] * 16 * B;

		local_work_size_4[0] = 128;
		global_work_size_4[0] = local_work_size_4[0] * 1024 * B;

		local_work_size_5[0] = 16;
		local_work_size_5[1] = 16;
		global_work_size_5[0] = local_work_size_5[0] * 16;
		global_work_size_5[1] = local_work_size_5[1] * 16 * B;
	}
	break;
	case 524288:
	{
		local_work_size_1[0] = 16;
		local_work_size_1[1] = 16;
		global_work_size_1[0] = local_work_size_1[0] * 8;
		global_work_size_1[1] = local_work_size_1[1] * 16 * B;

		local_work_size_2[0] = 128;
		global_work_size_2[0] = local_work_size_2[0] * 512 * B;

		local_work_size_3[0] = 16;
		local_work_size_3[1] = 16;
		global_work_size_3[0] = local_work_size_3[0] * 16;
		global_work_size_3[1] = local_work_size_3[1] * 8 * B;

		local_work_size_4[0] = 64;
		global_work_size_4[0] = local_work_size_4[0] * 1024 * B;

		local_work_size_5[0] = 16;
		local_work_size_5[1] = 16;
		global_work_size_5[0] = local_work_size_5[0] * 16;
		global_work_size_5[1] = local_work_size_5[1] * 8 * B;
	}
	break;
	case 262144:
	{
		local_work_size_1[0] = 128;
		global_work_size_1[0] = local_work_size_1[0] * 256 * B;

		local_work_size_2[0] = 256;
		global_work_size_2[0] = local_work_size_2[0] * 64 * B;

		local_work_size_3[0] = 16;
		local_work_size_3[1] = 16;
		global_work_size_3[0] = local_work_size_3[0] * 64;
		global_work_size_3[1] = local_work_size_3[1] * B;
	}
	break;
	case 131072:
	{
		local_work_size_1[0] = 128;
		global_work_size_1[0] = local_work_size_1[0] * 128 * B;

		local_work_size_2[0] = 256;
		global_work_size_2[0] = local_work_size_2[0] * 64 * B;

		local_work_size_3[0] = 16;
		local_work_size_3[1] = 16;
		global_work_size_3[0] = local_work_size_3[0] * 32;
		global_work_size_3[1] = local_work_size_3[1] * B;
	}
	break;
	case 65536:
	{
		local_work_size_1[0] = 256;
		global_work_size_1[0] = local_work_size_1[0] * 32 * B;

		local_work_size_2[0] = 256;
		global_work_size_2[0] = local_work_size_2[0] * 32 * B;
	}
	break;
	case 32768:
	{
		local_work_size_1[0] = 128;
		global_work_size_1[0] = local_work_size_1[0] * 32 * B;

		local_work_size_2[0] = 256;
		global_work_size_2[0] = local_work_size_2[0] * 16 * B;
	}
	break;
	case 16384:
		{
			local_work_size_1[0] = 128;
			global_work_size_1[0] = local_work_size_1[0] * 16 * B;

			local_work_size_2[0] = 256;
			global_work_size_2[0] = local_work_size_2[0] * 8 * B;
		}
		break;
	case 8192:
		{
			local_work_size_1[0] = 128;
			global_work_size_1[0] = local_work_size_1[0] * 8 * B;

			local_work_size_2[0] = 128;
			global_work_size_2[0] = local_work_size_2[0] * 8 * B;
		}
		break;
	case 4096:
	case 2048:
	case 1024:
	case 512:
	case 256:
	case 128:
	case 64:
	case 32:
	case 16:
	case 8:
	case 4:
	case 2:
	case 1:
		{
			clSetKernelArg(kernel, 1, sizeof(cl_uint), &k_count);
			clSetKernelArg(kernel, 2, sizeof(cl_int), &dir);

			size_t gw = (B%NT) ? 1 + (B / NT) : (B / NT);
			global_work_size[0] = WGS * gw;
			local_work_size[0] = WGS;
		}
	}




	//clSetKernelArg(kernel, 6, sizeof(dbg), (void*) &dbg);

	std::cout << "count: " << k_count << std::endl;
	std::cout << "N: " << N << std::endl;

	clFinish(queue);
	double tev = 0, time = 0;

	switch (N)
	{
	case 1048576:
	case 524288:
		{
			std::cout << "globalws_1: " << global_work_size_1[0] << ", " << global_work_size_1[1] << std::endl;
			std::cout << "localws_1: " << local_work_size_1[0] << ", " << local_work_size_1[1] << std::endl;

			std::cout << "globalws_2: " << global_work_size_2[0] << std::endl;
			std::cout << "localws_2: " << local_work_size_2[0] << std::endl;

			std::cout << "globalws_3: " << global_work_size_3[0] << ", " << global_work_size_3[1] << std::endl;
			std::cout << "localws_3: " << local_work_size_3[0] << ", " << local_work_size_3[1] << std::endl;

			std::cout << "globalws_4: " << global_work_size_4[0] << std::endl;
			std::cout << "localws_4: " << local_work_size_4[0] << std::endl;

			std::cout << "globalws_5: " << global_work_size_5[0] << ", " << global_work_size_5[1] << std::endl;
			std::cout << "localws_5: " << local_work_size_5[0] << ", " << local_work_size_5[1] << std::endl;

			for (uint i = 0; i < 10; i++)
			{
				clEnqueueWriteBuffer(queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL);

				cl_event ev1, ev2, ev3, ev4, ev5;
				Timer tr;
				tr.Start();
				err = clEnqueueNDRangeKernel(queue, kernel_1, 2, NULL, global_work_size_1, local_work_size_1, 0, NULL, &ev1);
				err = clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, global_work_size_2, local_work_size_2, 1, &ev1, &ev2);
				err = clEnqueueNDRangeKernel(queue, kernel_3, 2, NULL, global_work_size_3, local_work_size_3, 1, &ev2, &ev3);
				err = clEnqueueNDRangeKernel(queue, kernel_4, 1, NULL, global_work_size_4, local_work_size_4, 1, &ev3, &ev4);
				err = clEnqueueNDRangeKernel(queue, kernel_5, 2, NULL, global_work_size_5, local_work_size_5, 1, &ev4, &ev5);
				clWaitForEvents(1, &ev5);
				double timep = tr.Sample();

				time = time == 0 ? timep : time;
				time = timep < time ? timep : time;

				cl_int ks;
				clGetEventInfo(ev1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 1 execution not complete" << std::endl;
				clGetEventInfo(ev2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 2 execution not complete" << std::endl;
				clGetEventInfo(ev3, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 3 execution not complete" << std::endl;
				clGetEventInfo(ev4, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 4 execution not complete" << std::endl;
				clGetEventInfo(ev5, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 5 execution not complete" << std::endl;

				cl_ulong kbeg1, kbeg2, kbeg3, kbeg4, kbeg5;
				cl_ulong kend1, kend2, kend3, kend4, kend5;
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(kbeg1), &kbeg1, NULL);
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(kend1), &kend1, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(kbeg2), &kbeg2, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(kend2), &kend2, NULL);
				clGetEventProfilingInfo(ev3, CL_PROFILING_COMMAND_START, sizeof(kbeg3), &kbeg3, NULL);
				clGetEventProfilingInfo(ev3, CL_PROFILING_COMMAND_END, sizeof(kend3), &kend3, NULL);
				clGetEventProfilingInfo(ev4, CL_PROFILING_COMMAND_START, sizeof(kbeg4), &kbeg4, NULL);
				clGetEventProfilingInfo(ev4, CL_PROFILING_COMMAND_END, sizeof(kend4), &kend4, NULL);
				clGetEventProfilingInfo(ev5, CL_PROFILING_COMMAND_START, sizeof(kbeg5), &kbeg5, NULL);
				clGetEventProfilingInfo(ev5, CL_PROFILING_COMMAND_END, sizeof(kend5), &kend5, NULL);

				double tevp = 0, tev1 = 0, tev2 = 0, tev3 = 0, tev4 = 0, tev5 = 0;
				tev1 = (double)(kend1 - kbeg1);
				//std::cout << "gpu event1: " << tev1 << std::endl;
				tev2 = (double)(kend2 - kbeg2);
				//std::cout << "gpu event2: " << tev2 << std::endl;
				tev3 = (double)(kend3 - kbeg3);
				tev4 = (double)(kend4 - kbeg4);
				tev5 = (double)(kend5 - kbeg5);

				tevp = tev1 + tev2 + tev3 + tev4 + tev5;

				tev = tev == 0 ? tevp : tev;
				tev = tevp < tev ? tevp : tev;

				clReleaseEvent(ev1);
				clReleaseEvent(ev2);
				clReleaseEvent(ev3);
				clReleaseEvent(ev4);
				clReleaseEvent(ev5);
				clFinish(queue);
			}
		}

		break;
	case 262144:
	case 131072:
		{
			std::cout << "globalws_1: " << global_work_size_1[0] << std::endl;
			std::cout << "localws_1: " << local_work_size_1[0] << std::endl;

			std::cout << "globalws_2: " << global_work_size_2[0] << std::endl;
			std::cout << "localws_2: " << local_work_size_2[0] << std::endl;

			std::cout << "globalws_3: " << global_work_size_3[0] << ", " << global_work_size_3[1] << std::endl;
			std::cout << "localws_3: " << local_work_size_3[0] << ", " << local_work_size_3[1] << std::endl;

			for (uint i = 0; i < 10; i++)
			{
				clEnqueueWriteBuffer(queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL);

				cl_event ev1, ev2, ev3;
				Timer tr;
				tr.Start();
				err = clEnqueueNDRangeKernel(queue, kernel_1, 1, NULL, global_work_size_1, local_work_size_1, 0, NULL, &ev1);
				err = clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, global_work_size_2, local_work_size_2, 1, &ev1, &ev2);
				err = clEnqueueNDRangeKernel(queue, kernel_3, 2, NULL, global_work_size_3, local_work_size_3, 1, &ev2, &ev3);
				clWaitForEvents(1, &ev3);
				double timep = tr.Sample();

				time = time == 0 ? timep : time;
				time = timep < time ? timep : time;

				cl_int ks;
				clGetEventInfo(ev1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 1 execution not complete" << std::endl;
				clGetEventInfo(ev2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 2 execution not complete" << std::endl;
				clGetEventInfo(ev3, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 3 execution not complete" << std::endl;

				cl_ulong kbeg1, kbeg2, kbeg3;
				cl_ulong kend1, kend2, kend3;
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(kbeg1), &kbeg1, NULL);
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(kend1), &kend1, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(kbeg2), &kbeg2, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(kend2), &kend2, NULL);
				clGetEventProfilingInfo(ev3, CL_PROFILING_COMMAND_START, sizeof(kbeg3), &kbeg3, NULL);
				clGetEventProfilingInfo(ev3, CL_PROFILING_COMMAND_END, sizeof(kend3), &kend3, NULL);

				double tevp = 0, tev1 = 0, tev2 = 0, tev3 = 0;
				tev1 = (double)(kend1 - kbeg1);
				//std::cout << "gpu event1: " << tev1 << std::endl;
				tev2 = (double)(kend2 - kbeg2);
				//std::cout << "gpu event2: " << tev2 << std::endl;
				tev3 = (double)(kend3 - kbeg3);
				tevp = tev1 + tev2 + tev3;

				tev = tev == 0 ? tevp : tev;
				tev = tevp < tev ? tevp : tev;

				clReleaseEvent(ev1);
				clReleaseEvent(ev2);
				clReleaseEvent(ev3);
				clFinish(queue);
			}
		}

		break;
	case 65536:
	case 32768:
	case 16384:
	case 8192:
		{
			std::cout << "globalws_1: " << global_work_size_1[0] << std::endl;
			std::cout << "localws_1: " << local_work_size_1[0] << std::endl;

			std::cout << "globalws_2: " << global_work_size_2[0] << std::endl;
			std::cout << "localws_2: " << local_work_size_2[0] << std::endl;

			for (uint i = 0; i < 10; i++)
			{
				clEnqueueWriteBuffer(queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL);

				cl_event ev1, ev2;
				Timer tr;
				tr.Start();
				err = clEnqueueNDRangeKernel(queue, kernel_1, 1, NULL, global_work_size_1, local_work_size_1, 0, NULL, &ev1);
				err = clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, global_work_size_2, local_work_size_2, 1, &ev1, &ev2);
				clWaitForEvents(1, &ev2);
				double timep = tr.Sample();

				time = time == 0 ? timep : time;
				time = timep < time ? timep : time;

				cl_int ks;
				clGetEventInfo(ev1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 1 execution not complete" << std::endl;
				clGetEventInfo(ev2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel 2 execution not complete" << std::endl;

				cl_ulong kbeg1, kbeg2;
				cl_ulong kend1, kend2;
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(kbeg1), &kbeg1, NULL);
				clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(kend1), &kend1, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(kbeg2), &kbeg2, NULL);
				clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(kend2), &kend2, NULL);

				double tevp = 0, tev1 = 0, tev2 = 0;
				tev1 = (double)(kend1 - kbeg1);
				//std::cout << "gpu event1: " << tev1 << std::endl;
				tev2 = (double)(kend2 - kbeg2);
				//std::cout << "gpu event2: " << tev2 << std::endl;
				tevp = tev1 + tev2;

				tev = tev == 0 ? tevp : tev;
				tev = tevp < tev ? tevp : tev;

				clReleaseEvent(ev1);
				clReleaseEvent(ev2);
				clFinish(queue);
			}
		}

		break;
	case 4096:
	case 2048:
	case 1024:
	case 512:
	case 256:
	case 128:
	case 64:
	case 32:
	case 16:
	case 8:
	case 4:
	case 2:
	case 1:
		{
			std::cout << "NT: " << NT << std::endl;
			std::cout << "globalws: " << global_work_size[0] << std::endl;
			std::cout << "localws: " << local_work_size[0] << std::endl;


			for (uint i = 0; i < 10; i++)
			{
				clEnqueueWriteBuffer(queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL);

				cl_event ev;
				Timer tr;
				tr.Start();
				err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &ev);
				clWaitForEvents(1, &ev);
				double timep = tr.Sample();

				time = time == 0 ? timep : time;
				time = timep < time ? timep : time;

				cl_int ks;
				clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
				if (ks != CL_COMPLETE)
					std::cout << "kernel execution not complete" << std::endl;

				cl_ulong kbeg;
				cl_ulong kend;
				clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(kbeg), &kbeg, NULL);
				clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(kend), &kend, NULL);

				double tevp = 0;
				tevp = (double)(kend - kbeg);

				tev = tev == 0 ? tevp : tev;
				tev = tevp < tev ? tevp : tev;

				clReleaseEvent(ev);
				clFinish(queue);
			}
		}
	}




	std::cout << "gpu event time (milliseconds): " << tev/1000000.0 << std::endl;
	double opsconst = 5.0 * (double)N * log((double)N) / log(2.0);
	std::cout << "gflops: " << ((double)B * opsconst)/tev << std::endl;

	std::cout << "cpu time (milliseconds): " << time*1000.0 << std::endl;

	// 7. Look at the results
#ifndef FORMAT_INTERLEAVED
	clEnqueueReadBuffer( queue, bufferReal, CL_TRUE, 0, N*B * sizeof(ClType), (void *)yr, 0, NULL, NULL );
	clEnqueueReadBuffer( queue, bufferImag, CL_TRUE, 0, N*B * sizeof(ClType), (void *)yi, 0, NULL, NULL );
#else
	clEnqueueReadBuffer( queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)yc, 0, NULL, NULL );
#endif

	uint *dq = new uint[N*B];
	//clEnqueueReadBuffer( queue, dbg, CL_TRUE, 0, N*B * sizeof(cl_uint), (void *)dq, 0, NULL, NULL );
	clFinish( queue );

	//std::cout << "**** DQ ****" << std::endl;
	//for(uint i=0; i<N; i++)
	//{
	//	std::cout << dq[i] << std::endl;
	//}

#ifdef FORMAT_INTERLEAVED
	for(uint i=0; i<N*B; i++)
	{
		yr[i] = yc[2*i];
		yi[i] = yc[2*i + 1];
	}
#endif


#ifdef LIST_RESULT
	std::cout << "**** MY ****" << std::endl;
	for(uint j=0; j<B; j++)
	{
		for(uint i=0; i<N; i++)
		{
#ifndef SEE_HEX
			std::cout << "(" << yr[j*N + i] << ", " << yi[j*N + i] << ") " << std::endl;
#else
			Utype rv, iv;
			rv.f = yr[j*N + i]; iv.f = yi[j*N + i];
			printf("(%0x, %0x)\n", rv.u, iv.u);
#endif
		}
		std::cout << "=======================BATCH ENDS====================" << std::endl;
	}
#endif


#ifdef CHECK_RESULT

	double rmse_max = 0;
	for (size_t j = 0; j < B; j++)
	{
		double rmse = 0;
		double maxv = 0;

		for (size_t i = 0; i < N; i++)
		{
			maxv = maxv > abs(refr[j*N + i]) ? maxv : abs(refr[j*N + i]);
			maxv = maxv > abs(refi[j*N + i]) ? maxv : abs(refi[j*N + i]);
		}

		for (size_t i = 0; i < N; i++)
		{
			rmse += (yr[j*N + i] - refr[j*N + i])*(yr[j*N + i] - refr[j*N + i]);
			rmse += (yi[j*N + i] - refi[j*N + i])*(yi[j*N + i] - refi[j*N + i]);
		}

		rmse = sqrt((rmse / maxv) / N);
		rmse_max = rmse > rmse_max ? rmse : rmse_max;
	}

	std::cout << "rrmse: " << rmse_max << std::endl;


#endif

	FftwDestroy(p);
    FftwFree(in);
	FftwFree(out);

	delete[] yr;
	delete[] yi;
	delete[] xr;
	delete[] xi;
	delete[] refr;
	delete[] refi;
	delete[] xc;
	delete[] yc;

	delete[] dq;

	return 0;
}

