
#ifndef REF_CPU_H
#define REF_CPU_H

#include <dlfcn.h>
#include <cstdio>
#include <stdlib.h>
#include <iostream>

#define LOCAL_FFTW_FORWARD (-1)
#define LOCAL_FFTW_BACKWARD (+1)
#define LOCAL_FFTW_ESTIMATE (1U << 6)

typedef float local_fftwf_complex[2];

typedef void *(*ftype_fftwf_malloc)(size_t n);
typedef void (*ftype_fftwf_free)(void *p);

typedef void *(*ftype_fftwf_plan_many_dft)(int rank, const int *n, int howmany,
                             local_fftwf_complex *in, const int *inembed,
                             int istride, int idist,
                             local_fftwf_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags);

typedef void (*ftype_fftwf_execute)(void *);

typedef void (*ftype_fftwf_destroy_plan)(void *);

class RefCPUHandle
{
    void *fftw3f_lib;
    void *fftw3_lib;

	RefCPUHandle() : fftw3f_lib(nullptr), fftw3_lib(nullptr)
    {
        char *env_value_fftw3f = getenv("ROCFFT_DBG_FFTW3F_LIB");
        char *env_value_fftw3  = getenv("ROCFFT_DBG_FFTW3_LIB");

        if(!env_value_fftw3f)
        {
            std::cout << "error finding fftw3f lib, set env variable ROCFFT_DBG_FFTW3F_LIB" << std::endl;
        }
    
        if(!env_value_fftw3)
        {
            std::cout << "error finding fftw3 lib, set env variable ROCFFT_DBG_FFTW3_LIB" << std::endl;
        }
    
    
    	fftw3f_lib = dlopen(env_value_fftw3f, RTLD_NOW);
        if(!fftw3f_lib)
        {
            std::cout << "error in fftw3f dlopen" << std::endl;
        }

    	fftw3_lib = dlopen(env_value_fftw3, RTLD_NOW);
        if(!fftw3_lib)
        {
            std::cout << "error in fftw3 dlopen" << std::endl;
        }
    }

public:
	RefCPUHandle(const RefCPUHandle &) = delete; // delete is a c++11 feature, prohibit copy constructor 
	RefCPUHandle &operator=(const RefCPUHandle &) = delete; //prohibit assignment operator

	static RefCPUHandle &GetRefCPUHandle()
	{
		static RefCPUHandle refCPUHandle;
		return refCPUHandle;
	}

	~RefCPUHandle()
	{
        if(!fftw3f_lib)
        {
            dlclose(fftw3f_lib);
            fftw3f_lib = nullptr;
        }

        if(!fftw3_lib)
        {
            dlclose(fftw3_lib);
            fftw3_lib = nullptr;
        }
	}

};

void rocfft_internal_cpu_reference_op(const void *data_p, void *back_p);

#endif // REF_CPU_H

