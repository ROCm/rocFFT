
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

class RefLibHandle
{
	RefLibHandle() : fftw3f_lib(nullptr), fftw3_lib(nullptr)
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
    void *fftw3f_lib;
    void *fftw3_lib;

	RefLibHandle(const RefLibHandle &) = delete; // delete is a c++11 feature, prohibit copy constructor 
	RefLibHandle &operator=(const RefLibHandle &) = delete; //prohibit assignment operator

	static RefLibHandle &GetRefLibHandle()
	{
		static RefLibHandle refLibHandle;
		return refLibHandle;
	}

	~RefLibHandle()
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

class RefLibOp
{
    local_fftwf_complex *in;
    local_fftwf_complex *ot;

    void DataSetup(const void *data_p)
    {
        RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();
    	ftype_fftwf_malloc local_fftwf_malloc = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");
    
        DeviceCallIn *data = (DeviceCallIn *)data_p;
   
        size_t totalSize = sizeof(local_fftwf_complex);
        totalSize *= data->node->batch;
        for(size_t i=0; i<data->node->length.size(); i++) totalSize *= data->node->length[i];

        in = (local_fftwf_complex *)local_fftwf_malloc(totalSize);
        ot = (local_fftwf_complex *)local_fftwf_malloc(totalSize);
        
        memset(in, 0x40, totalSize);
        memset(ot, 0x40, totalSize);
    }

    void Execute(const void *data_p)
    {
        RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();
        ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
        ftype_fftwf_execute local_fftwf_execute = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
        ftype_fftwf_destroy_plan local_fftwf_destroy_plan = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");

        DeviceCallIn *data = (DeviceCallIn *)data_p;

        int n[1];
        n[0] = data->node->length[0];     
        void *p = local_fftwf_plan_many_dft(1, n, 1, in, NULL, 1, 0, ot, NULL, 1, 0, LOCAL_FFTW_FORWARD, LOCAL_FFTW_ESTIMATE);

        size_t in_size = data->node->iDist * data->node->batch;
        size_t in_size_bytes = in_size * 2 * sizeof(float);
        hipMemcpy(in, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

        local_fftwf_execute(p);
        local_fftwf_destroy_plan(p);
    }

public:
    RefLibOp(const void *data_p) : in(nullptr), ot(nullptr)
    {
        DataSetup(data_p);
        Execute(data_p);
    }

    void VerifyResult(const void *data_p)
    {
        DeviceCallIn *data = (DeviceCallIn *)data_p;
        size_t out_size = data->node->oDist * data->node->batch;
        size_t out_size_bytes = out_size * 2 * sizeof(float);
        hipMemcpy(ot, data->bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);
    }

    ~RefLibOp()
    {
        RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();
    	ftype_fftwf_free local_fftwf_free = (ftype_fftwf_free)dlsym(refHandle.fftw3f_lib, "fftwf_free");

        if(in)
        {
            local_fftwf_free(in);
            in = nullptr;
        }
        if(ot)
        {
            local_fftwf_free(ot);
            ot = nullptr;
        }
    }
};


#endif // REF_CPU_H

