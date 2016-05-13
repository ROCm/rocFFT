
#include "./common.h"

#define NEAR_ZERO_TOLERATE true
#define KERN_NAME "fft_fwd"

#define WGS 256

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
	size_t T = atoi(argv[4]);

	size_t NT = (N >= 1024) ? 1 : 1024 / N;

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
	cl_program program = clCreateProgramWithSource( context, flines, (const char **)ps, NULL, NULL );
	file_lines_clear(flines, ps);
	cl_int err = clBuildProgram( program, 1, &device, NULL, NULL, NULL );
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
	cl_kernel kernel = clCreateKernel( program, KERN_NAME, NULL );

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
		xr[i] =   (Type)(rand()%101);
		xi[i] =   (Type)(rand()%101);
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
	p = FftwPlanFn(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
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
	for(uint j=0; j<B; j++)
	{
		for(uint i=0; i<N; i++)
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
	cl_mem bufferReal = clCreateBuffer( context, CL_MEM_READ_WRITE, N*B * sizeof(ClType), NULL, NULL );
	cl_mem bufferImag = clCreateBuffer( context, CL_MEM_READ_WRITE, N*B * sizeof(ClType), NULL, NULL );
#else
	cl_mem bufferCplx = clCreateBuffer( context, CL_MEM_READ_WRITE, N*B * sizeof(ClType2), NULL, NULL );
#endif

	cl_mem dbg = clCreateBuffer( context, CL_MEM_READ_WRITE, N*B * sizeof(cl_uint), NULL, NULL );

	// Fill buffers for FFT
#ifndef FORMAT_INTERLEAVED
	clEnqueueWriteBuffer( queue, bufferReal, CL_TRUE, 0, N*B * sizeof(ClType), (void *)xr, 0, NULL, NULL );
	clEnqueueWriteBuffer( queue, bufferImag, CL_TRUE, 0, N*B * sizeof(ClType), (void *)xi, 0, NULL, NULL );
#else
	clEnqueueWriteBuffer( queue, bufferCplx, CL_TRUE, 0, N*B * sizeof(ClType2), (void *)xc, 0, NULL, NULL );
#endif

	// 6. Launch the kernel
	size_t global_work_size[1];
	size_t local_work_size[1];
	size_t gw = (B%NT) ? 1 + (B/NT) : (B/NT);
	global_work_size[0] = WGS * gw;
	local_work_size[0] = WGS;

#ifndef FORMAT_INTERLEAVED
	clSetKernelArg(kernel, 0, sizeof(bufferReal), (void*) &bufferReal);
	clSetKernelArg(kernel, 1, sizeof(bufferImag), (void*) &bufferImag);
#else
	clSetKernelArg(kernel, 0, sizeof(bufferCplx), (void*) &bufferCplx);
#endif

	unsigned int k_count = B;
	unsigned int k_N = N;
	unsigned int k_T = T;
	unsigned int k_NT = NT;

	clSetKernelArg(kernel, 1, sizeof(unsigned int), &k_count);
	clSetKernelArg(kernel, 2, sizeof(unsigned int), &k_N);
	clSetKernelArg(kernel, 3, sizeof(unsigned int), &k_T);
	clSetKernelArg(kernel, 4, sizeof(unsigned int), &k_NT);

	//clSetKernelArg(kernel, 5, sizeof(dbg), (void*) &dbg);

	std::cout << "count: " << k_count << std::endl;
	std::cout << "N: " << k_N << std::endl;
	std::cout << "T: " << k_T << std::endl;
	std::cout << "NT: " << k_NT << std::endl;
	std::cout << "globalws: " << global_work_size[0] << std::endl;
	std::cout << "localws: " << local_work_size[0] << std::endl;

	clFinish( queue );

	cl_event ev;
	Timer tr;
	tr.Start();
	for(uint i=0; i<1; i++)
	{
		clEnqueueNDRangeKernel( queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &ev);
		clFinish( queue );
	}
	double time = tr.Sample();

	cl_int ks;
	clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ks), &ks, NULL);
	if(ks != CL_COMPLETE)
		std::cout << "kernel execution not complete" << std::endl;

	cl_ulong kbeg;
    cl_ulong kend;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(kbeg), &kbeg, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(kend), &kend, NULL);

	double tev = (double)(kend - kbeg);
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
	clEnqueueReadBuffer( queue, dbg, CL_TRUE, 0, N*B * sizeof(cl_uint), (void *)dq, 0, NULL, NULL );
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
	for(uint i=0; i<N*B; i++)
	{
		if(NEAR_ZERO_TOLERATE)
		{
			if( (abs(refr[i]) < 0.1f) && (abs(yr[i]) < 0.1f) )
				continue;
			if( (abs(refi[i]) < 0.1f) && (abs(yi[i]) < 0.1f) )
				continue;
		}

		if( abs(yr[i] -  refr[i]) > abs(0.01 * refr[i]) )
		{
			std::cout << "FAIL" << std::endl;
			std::cout << "B: " << (i/N) << " index: " << (i%N) << std::endl;
			std::cout << "refr: " << refr[i] << " yr: " << yr[i] << std::endl;
			break;
		}

		if( abs(yi[i] -  refi[i]) > abs(0.01 * refi[i]) )
		{
			std::cout << "FAIL" << std::endl;
			std::cout << "B: " << (i/N) << " index: " << (i%N) << std::endl;
			std::cout << "refi: " << refi[i] << " yi: " << yi[i] << std::endl;
			break;
		}
	}
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

