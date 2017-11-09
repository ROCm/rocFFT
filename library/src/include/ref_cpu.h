
#ifndef REF_CPU_H
#define REF_CPU_H

#ifdef REF_DEBUG

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
    local_fftwf_complex *in; // input
    local_fftwf_complex *ot; // output from fftw 
    local_fftwf_complex *lb; // output from lib
    size_t totalSize;
    
    void DataSetup(const void *data_p)
    {
        RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();
    	ftype_fftwf_malloc local_fftwf_malloc = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");
    
        DeviceCallIn *data = (DeviceCallIn *)data_p;
   
        totalSize = data->node->batch;
        for(size_t i=0; i<data->node->length.size(); i++) totalSize *= data->node->length[i];
        size_t totalByteSize = totalSize*sizeof(local_fftwf_complex);

        in = (local_fftwf_complex *)local_fftwf_malloc(totalByteSize);
        ot = (local_fftwf_complex *)local_fftwf_malloc(totalByteSize);
        lb = (local_fftwf_complex *)local_fftwf_malloc(totalByteSize);
        
        memset(in, 0x40, totalByteSize);
        memset(ot, 0x40, totalByteSize);
        memset(lb, 0x40, totalByteSize);
    }

    void CopyVector(local_fftwf_complex *dst, local_fftwf_complex *src, size_t batch, size_t dist, std::vector<size_t> length, std::vector<size_t> stride)
    {
        size_t offset_dst = 0;
        size_t offset_src = 0;
        size_t offset_src_d = 0;
		size_t pos = 0;

		std::vector<size_t> current;
		for(size_t i=0; i<length.size(); i++) current.push_back(0);

        size_t b = 0;
        while(b<batch)
        {
            offset_src += b*dist;
        	while(true)
            {
                offset_src = offset_src_d + current[0]*stride[0];
    
    			dst[offset_dst][0] = src[offset_src][0];
    			dst[offset_dst][1] = src[offset_src][1];
    			
                current[0]++;
                offset_dst++;
        
                while(current[pos] == length[pos])
                {
                    if(pos == (length.size() - 1))
                    {
    					goto nested_exit;
                    }
        
                    current[pos] = 0;
                    pos++;
                    current[pos]++;
       
    				for(size_t i=1; i<current.size(); i++)
    					offset_src_d += current[i]*stride[i]; 
        
                }
        
                pos = 0;
            }
nested_exit:
            b++;
		}
    }

    void CopyInputVector(const void *data_p)
    {
        DeviceCallIn *data = (DeviceCallIn *)data_p;

        size_t in_size = data->node->iDist * data->node->batch;
        size_t in_size_bytes = in_size * 2 * sizeof(float);
        local_fftwf_complex *tmp_mem = (local_fftwf_complex *)malloc(in_size_bytes);
        hipMemcpy(tmp_mem, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

        CopyVector(in, tmp_mem, data->node->batch, data->node->iDist, data->node->length, data->node->inStride);

        if(tmp_mem)
            free(tmp_mem);
    }

    void Execute(const void *data_p)
    {
        DeviceCallIn *data = (DeviceCallIn *)data_p;

        if(data->node->scheme == CS_KERNEL_STOCKHAM)
        {
            RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();
            ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
            ftype_fftwf_execute local_fftwf_execute = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
            ftype_fftwf_destroy_plan local_fftwf_destroy_plan = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");
    
            int n[1];
            n[0] = data->node->length[0];
            int howmany = data->node->batch;
            for(size_t i=1; i<data->node->length.size(); i++) howmany *= data->node->length[i];
    
            void *p = local_fftwf_plan_many_dft(1, n, howmany, in, NULL, 1, n[0], ot, NULL, 1, n[0], LOCAL_FFTW_FORWARD, LOCAL_FFTW_ESTIMATE);
            CopyInputVector(data_p);
            local_fftwf_execute(p);
            local_fftwf_destroy_plan(p);
        }
        else if(data->node->scheme == CS_KERNEL_TRANSPOSE)
        {
            CopyInputVector(data_p);

            int howmany = data->node->batch;
            for(size_t i=2; i<data->node->length.size(); i++) howmany *= data->node->length[i];

            size_t cols = data->node->length[0];
            size_t rows = data->node->length[1];
            for(size_t b=0; b<howmany; b++)
            {
                for(size_t i=0; i<rows; i++)
                {
                    for(size_t j=0; j<cols; j++)
                    {
                        ot[b*rows*cols + j*rows + i][0] = in[b*rows*cols + i*cols + j][0];
                        ot[b*rows*cols + j*rows + i][1] = in[b*rows*cols + i*cols + j][1];
                    }
                }
            }
        }
        else
        {
            assert(false);
        }
    }

public:
    RefLibOp(const void *data_p) : in(nullptr), ot(nullptr), lb(nullptr), totalSize(0)
    {
        DataSetup(data_p);
        Execute(data_p);
    }

    void VerifyResult(const void *data_p)
    {
        DeviceCallIn *data = (DeviceCallIn *)data_p;
        size_t out_size = data->node->oDist * data->node->batch;
        size_t out_size_bytes = out_size * 2 * sizeof(float);
   
        local_fftwf_complex *tmp_mem = (local_fftwf_complex *)malloc(out_size_bytes);
        hipMemcpy(tmp_mem, data->bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);
        CopyVector(lb, tmp_mem, data->node->batch, data->node->oDist, data->node->length, data->node->outStride);

        double maxMag = 0.0;
        double rmse = 0.0, nrmse = 0.0;
        for(size_t i=0; i<totalSize; i++)
        {
            double mag;
            double ex_r, ex_i, ac_r, ac_i;

            ac_r = lb[i][0];
            ac_i = lb[i][1];
            ex_r = ot[i][0];
            ex_i = ot[i][1];

            mag = ex_r*ex_r + ex_i*ex_i;
            maxMag = (mag > maxMag) ? mag : maxMag;

            rmse += ((ex_r - ac_r)*(ex_r - ac_r) + (ex_i - ac_i)*(ex_i - ac_i));
        }

        maxMag = sqrt(maxMag);
        rmse = sqrt(rmse/(double)totalSize);
        nrmse = rmse/maxMag;

        std::cout << "rmse: " << rmse << std::endl << "nrmse: " << nrmse << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
         
#if 0
        std::cout << "input" << std::endl;
        for(size_t i=0; i<4; i++)
        {
            std::cout << in[i][0] << ", " << in[i][1] << std::endl;
        }

        std::cout << "lib output" << std::endl;
        for(size_t i=0; i<4; i++)
        {
            std::cout << lb[i][0] << ", " << lb[i][1] << std::endl;
        }

        std::cout << "fftw output" << std::endl;
        for(size_t i=0; i<4; i++)
        {
            std::cout << ot[i][0] << ", " << ot[i][1] << std::endl;
        }
#endif

        if(tmp_mem)
            free(tmp_mem);
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
        if(lb)
        {
            local_fftwf_free(lb);
            lb = nullptr;
        }
    }
};

#endif // REF_DEBUG

#endif // REF_CPU_H

