
#include <CL/cl.h>
#include <stdio.h>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <windows.h>

typedef unsigned int uint;

extern "C" size_t file_lines_get(const char *file_name, char ***retp);
extern "C" void file_lines_clear(size_t len, char **p);

int complex2complex(int argc, char ** argv);
int real2complex(int argc, char ** argv);


#ifdef DoublePrecision

#define Type		double
#define Utype		F64fu
#define ClType		cl_double
#define ClType2		cl_double2
#define FftwComplex	fftw_complex
#define FftwPlan	fftw_plan
#define FftwMalloc	fftw_malloc
#define FftwPlanFn	fftw_plan_dft_1d
#define FftwExecute	fftw_execute
#define FftwDestroy	fftw_destroy_plan
#define FftwFree	fftw_free

#else

#define Type		float
#define Utype		F32fu
#define ClType		cl_float
#define ClType2		cl_float2
#define FftwComplex	fftwf_complex
#define FftwPlan	fftwf_plan
#define FftwMalloc	fftwf_malloc
#define FftwPlanFn	fftwf_plan_dft_1d
#define FftwExecute	fftwf_execute
#define FftwDestroy	fftwf_destroy_plan
#define FftwFree	fftwf_free

#endif

typedef union {
	cl_uint  u;
	cl_int   i;
} cb_t;


typedef union {
	float f;
	unsigned int u;
} F32fu;

typedef union {
	double f;
	unsigned long long u;
} F64fu;

struct Timer
{
    LARGE_INTEGER start, stop, freq;

    public:
        Timer() { QueryPerformanceFrequency( &freq ); }

        void    Start  () { QueryPerformanceCounter(&start); }
        double Sample()
        {
            QueryPerformanceCounter  ( &stop );
            double time = (double)(stop.QuadPart-start.QuadPart) / (double)(freq.QuadPart);
            return time;
        }
};

