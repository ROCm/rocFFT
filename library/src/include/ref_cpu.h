
#ifndef REF_CPU_H
#define REF_CPU_H

#ifdef REF_DEBUG

#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>

#define LOCAL_FFTW_FORWARD (-1)
#define LOCAL_FFTW_BACKWARD (+1)
#define LOCAL_FFTW_ESTIMATE (1U << 6)

typedef float local_fftwf_complex[2]; // each elment is a float type

typedef void* (*ftype_fftwf_malloc)(size_t n);
typedef void (*ftype_fftwf_free)(void* p);

typedef void* (*ftype_fftwf_plan_many_dft)(int                  rank,
                                           const int*           n,
                                           int                  howmany,
                                           local_fftwf_complex* in,
                                           const int*           inembed,
                                           int                  istride,
                                           int                  idist,
                                           local_fftwf_complex* out,
                                           const int*           onembed,
                                           int                  ostride,
                                           int                  odist,
                                           int                  sign,
                                           unsigned             flags);

typedef void (*ftype_fftwf_execute)(void*);

typedef void (*ftype_fftwf_destroy_plan)(void*);

class RefLibHandle
{
    RefLibHandle()
        : fftw3f_lib(nullptr)
        , fftw3_lib(nullptr)
    {
        char* env_value_fftw3f = getenv("ROCFFT_DBG_FFTW3F_LIB");
        char* env_value_fftw3  = getenv("ROCFFT_DBG_FFTW3_LIB");

        if(!env_value_fftw3f)
        {
            std::cout << "error finding fftw3f lib, set env variable ROCFFT_DBG_FFTW3F_LIB"
                      << std::endl;
        }

        if(!env_value_fftw3)
        {
            std::cout << "error finding fftw3 lib, set env variable ROCFFT_DBG_FFTW3_LIB"
                      << std::endl;
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
    void* fftw3f_lib;
    void* fftw3_lib;

    RefLibHandle(const RefLibHandle&)
        = delete; // delete is a c++11 feature, prohibit copy constructor
    RefLibHandle& operator=(const RefLibHandle&) = delete; // prohibit assignment operator

    static RefLibHandle& GetRefLibHandle()
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
    local_fftwf_complex* in; // input
    local_fftwf_complex* ot; // output from fftw
    local_fftwf_complex* lb; // output from lib
    size_t               totalSize;

    void DataSetup(const void* data_p)
    {
        RefLibHandle&      refHandle = RefLibHandle::GetRefLibHandle();
        ftype_fftwf_malloc local_fftwf_malloc
            = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");

        DeviceCallIn* data = (DeviceCallIn*)data_p;

        if(data->node->scheme == CS_KERNEL_CHIRP)
        {
            totalSize = 2 * data->node->lengthBlue;
        }
        else if((data->node->scheme == CS_KERNEL_FFT_MUL)
                || (data->node->scheme == CS_KERNEL_PAD_MUL)
                || (data->node->scheme == CS_KERNEL_RES_MUL))
        {
            totalSize = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                totalSize *= data->node->length[i];
            totalSize *= data->node->lengthBlue;
        }
        else
        {
            totalSize = data->node->batch;
            for(size_t i = 0; i < data->node->length.size(); i++)
                totalSize *= data->node->length[i];
        }
        size_t totalByteSize = totalSize * sizeof(local_fftwf_complex);

        in = (local_fftwf_complex*)local_fftwf_malloc(totalByteSize);
        ot = (local_fftwf_complex*)local_fftwf_malloc(totalByteSize);
        lb = (local_fftwf_complex*)local_fftwf_malloc(totalByteSize);

        memset(in, 0x40, totalByteSize);
        memset(ot, 0x40, totalByteSize);
        memset(lb, 0x40, totalByteSize);
    }

    void CopyVector(local_fftwf_complex* dst,
                    local_fftwf_complex* src,
                    size_t               batch,
                    size_t               dist,
                    std::vector<size_t>  length,
                    std::vector<size_t>  stride)
    {
        size_t lenSize = 1;
        for(size_t i = 0; i < length.size(); i++)
            lenSize *= length[i];

        size_t b = 0;
        while(b < batch)
        {
            size_t offset_dst   = 0;
            size_t offset_src   = 0;
            size_t offset_src_d = 0;
            size_t pos          = 0;
            bool   obreak       = false;

            std::vector<size_t> current;
            for(size_t i = 0; i < length.size(); i++)
                current.push_back(0);

            while(true)
            {
                offset_src = offset_src_d + current[0] * stride[0];

                dst[offset_dst][0] = src[offset_src][0];
                dst[offset_dst][1] = src[offset_src][1];

                current[0]++;
                offset_dst++;

                while(current[pos] == length[pos])
                {
                    if(pos == (length.size() - 1))
                    {
                        obreak = true;
                        break;
                    }

                    current[pos] = 0;
                    pos++;
                    current[pos]++;

                    offset_src_d = 0;
                    for(size_t i = 1; i < current.size(); i++)
                        offset_src_d += current[i] * stride[i];
                }

                if(obreak)
                    break;

                pos = 0;
            }

            b++;
            src += dist;
            dst += lenSize;
        }
    }

    void CopyInputVector(const void* data_p, size_t offset = 0)
    {
        DeviceCallIn* data = (DeviceCallIn*)data_p;

        size_t in_size_bytes = (data->node->iDist * data->node->batch) * sizeof(float);

        if(data->node->inArrayType == rocfft_array_type_real)
        {
            in_size_bytes *= 1;
        }
        else
        {
            in_size_bytes *= 2;
        }

        void*                buf     = ((char*)data->bufIn[0] + offset);
        local_fftwf_complex* tmp_mem = (local_fftwf_complex*)malloc(in_size_bytes);
        hipMemcpy(tmp_mem, buf, in_size_bytes, hipMemcpyDeviceToHost);

        CopyVector(in,
                   tmp_mem,
                   data->node->batch,
                   data->node->iDist,
                   data->node->length,
                   data->node->inStride);

        if(tmp_mem)
            free(tmp_mem);
    }

    inline float2
        TwMul(float2* twiddles, const size_t twl, const int direction, float2 val, size_t u)
    {
        size_t j      = u & 255;
        float2 result = twiddles[j];

        float  real, imag;
        size_t h = 1;
        do
        {
            u >>= 8;
            j        = u & 255;
            real     = (result.x * twiddles[256 * h + j].x - result.y * twiddles[256 * h + j].y);
            imag     = (result.y * twiddles[256 * h + j].x + result.x * twiddles[256 * h + j].y);
            result.x = real;
            result.y = imag;
            h++;
        } while(h < twl);

        if(direction == -1)
        {
            real = (result.x * val.x) - (result.y * val.y);
            imag = (result.y * val.x) + (result.x * val.y);
        }
        else
        {
            real = (result.x * val.x) + (result.y * val.y);
            imag = -(result.y * val.x) + (result.x * val.y);
        }

        result.x = real;
        result.y = imag;

        return result;
    }

    inline void chirp(size_t N, size_t M, int dir, local_fftwf_complex* vec)
    {
        const double TWO_PI = atan(1.0) * 8.0 * (double)(-dir);

        for(size_t i = 0; i <= (M - N); i++)
        {
            double cs = cos(TWO_PI * (double)(i * i) / (2.0 * (double)N));
            double ss = sin(TWO_PI * (double)(i * i) / (2.0 * (double)N));

            if(i == 0)
            {
                vec[i][0] = cs;
                vec[i][1] = ss;

                vec[i + M][0] = cs;
                vec[i + M][1] = ss;
            }
            else if(i < N)
            {
                vec[i][0]     = cs;
                vec[i][1]     = ss;
                vec[M - i][0] = cs;
                vec[M - i][1] = ss;

                vec[i + M][0]     = cs;
                vec[i + M][1]     = ss;
                vec[M - i + M][0] = cs;
                vec[M - i + M][1] = ss;
            }
            else
            {
                vec[i][0]     = 0;
                vec[i][1]     = 0;
                vec[i + M][0] = 0;
                vec[i + M][1] = 0;
            }
        }
    }

    inline void chirp_fft(size_t N, size_t M, int dir, local_fftwf_complex* vec)
    {
        RefLibHandle&             refHandle = RefLibHandle::GetRefLibHandle();
        ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft
            = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
        ftype_fftwf_execute local_fftwf_execute
            = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
        ftype_fftwf_destroy_plan local_fftwf_destroy_plan
            = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");

        int n[1];
        n[0] = M;

        void* p;
        if(dir == -1)
            p = local_fftwf_plan_many_dft(1,
                                          n,
                                          1,
                                          vec,
                                          NULL,
                                          1,
                                          n[0],
                                          vec,
                                          NULL,
                                          1,
                                          n[0],
                                          LOCAL_FFTW_FORWARD,
                                          LOCAL_FFTW_ESTIMATE);
        else
            p = local_fftwf_plan_many_dft(1,
                                          n,
                                          1,
                                          vec,
                                          NULL,
                                          1,
                                          n[0],
                                          vec,
                                          NULL,
                                          1,
                                          n[0],
                                          LOCAL_FFTW_BACKWARD,
                                          LOCAL_FFTW_ESTIMATE);

        chirp(N, M, dir, vec);
        local_fftwf_execute(p);
        local_fftwf_destroy_plan(p);
    }

    void Execute(const void* data_p)
    {
        DeviceCallIn* data = (DeviceCallIn*)data_p;

        if(data->node->scheme == CS_KERNEL_STOCKHAM)
        {
            RefLibHandle&             refHandle = RefLibHandle::GetRefLibHandle();
            ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft
                = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
            ftype_fftwf_execute local_fftwf_execute
                = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
            ftype_fftwf_destroy_plan local_fftwf_destroy_plan
                = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");

            int n[1];
            n[0]        = data->node->length[0];
            int howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            void* p;
            if(data->node->direction == -1)
                p = local_fftwf_plan_many_dft(1,
                                              n,
                                              howmany,
                                              in,
                                              NULL,
                                              1,
                                              n[0],
                                              ot,
                                              NULL,
                                              1,
                                              n[0],
                                              LOCAL_FFTW_FORWARD,
                                              LOCAL_FFTW_ESTIMATE);
            else
                p = local_fftwf_plan_many_dft(1,
                                              n,
                                              howmany,
                                              in,
                                              NULL,
                                              1,
                                              n[0],
                                              ot,
                                              NULL,
                                              1,
                                              n[0],
                                              LOCAL_FFTW_BACKWARD,
                                              LOCAL_FFTW_ESTIMATE);

            CopyInputVector(data_p);
            local_fftwf_execute(p);
            local_fftwf_destroy_plan(p);
        }
        else if(data->node->scheme == CS_KERNEL_TRANSPOSE)
        {
            CopyInputVector(data_p);

            size_t howmany = data->node->batch;
            for(size_t i = 2; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            size_t cols = data->node->length[0];
            size_t rows = data->node->length[1];

            if(data->node->large1D == 0)
            {
                for(size_t b = 0; b < howmany; b++)
                {
                    for(size_t i = 0; i < rows; i++)
                    {
                        for(size_t j = 0; j < cols; j++)
                        {
                            ot[b * rows * cols + j * rows + i][0]
                                = in[b * rows * cols + i * cols + j][0];
                            ot[b * rows * cols + j * rows + i][1]
                                = in[b * rows * cols + i * cols + j][1];
                        }
                    }
                }
            }
            else
            {
                float2*                   twtc;
                size_t                    ns = 0;
                TwiddleTableLarge<float2> twTable(data->node->large1D);
                std::tie(ns, twtc) = twTable.GenerateTwiddleTable();

                int twl = 0;

                if(data->node->large1D > (size_t)256 * 256 * 256 * 256)
                    printf("large1D twiddle size too large error");
                else if(data->node->large1D > (size_t)256 * 256 * 256)
                    twl = 4;
                else if(data->node->large1D > (size_t)256 * 256)
                    twl = 3;
                else if(data->node->large1D > (size_t)256)
                    twl = 2;
                else
                    twl = 0;

                for(size_t b = 0; b < howmany; b++)
                {
                    for(size_t i = 0; i < rows; i++)
                    {
                        for(size_t j = 0; j < cols; j++)
                        {
                            float2 in_v, ot_v;

                            in_v.x = in[b * rows * cols + i * cols + j][0];
                            in_v.y = in[b * rows * cols + i * cols + j][1];

                            ot_v = TwMul(twtc, twl, data->node->direction, in_v, i * j);

                            ot[b * rows * cols + j * rows + i][0] = ot_v.x;
                            ot[b * rows * cols + j * rows + i][1] = ot_v.y;
                        }
                    }
                }
            }
        }
        else if(data->node->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
        {
            size_t in_size_bytes = (data->node->iDist * data->node->batch) * sizeof(float);

            float* tmp_mem = (float*)malloc(in_size_bytes);
            hipMemcpy(tmp_mem, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            size_t elements = 1;
            for(size_t d = 0; d < data->node->length.size(); d++)
            {
                elements *= data->node->length[d];
            }
            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t i = 0; i < elements; i++)
                {
                    ot[data->node->oDist * b + i][0] = tmp_mem[data->node->iDist * b + i];
                    ot[data->node->oDist * b + i][1] = 0;
                }
            }

            if(tmp_mem)
                free(tmp_mem);
        }
        else if(data->node->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            // assump the input is complex, the output is hermitian on take the first
            // [N/2 + 1] elements
            size_t in_size_bytes = (data->node->iDist * data->node->batch) * 2 * sizeof(float);

            local_fftwf_complex* tmp_mem = (local_fftwf_complex*)malloc(in_size_bytes);
            hipMemcpy(tmp_mem, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            size_t elements = 1;
            elements *= data->node->length[0] / 2 + 1;
            for(size_t d = 1; d < data->node->length.size(); d++)
            {
                elements *= data->node->length[d];
            }

            printf("iDist=%zu,oDist=%zu, in complex2hermitian kernel\n",
                   data->node->iDist,
                   data->node->oDist);
            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t i = 0; i < elements; i++) // TODO: only work for 1D cases
                {
                    ot[data->node->oDist * b + i][0] = tmp_mem[data->node->iDist * b + i][0];
                    ot[data->node->oDist * b + i][1] = tmp_mem[data->node->iDist * b + i][1];
                }
            }

            if(tmp_mem)
                free(tmp_mem);
        }
        else if(data->node->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
        {
            // assump the input is hermitian, the output is complex on take the first
            // [N/2 + 1] elements
            size_t in_size_bytes = (data->node->iDist * data->node->batch) * 2 * sizeof(float);

            local_fftwf_complex* tmp_mem = (local_fftwf_complex*)malloc(in_size_bytes);
            hipMemcpy(tmp_mem, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            size_t output_size = data->node->length[0];
            size_t input_size  = output_size / 2 + 1;

            printf("iDist=%zu,oDist=%zu, in hermitian2complex kernel\n",
                   data->node->iDist,
                   data->node->oDist);
            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t d = 0;
                    d < (data->node->length.size() == 2 ? (data->node->length[1]) : 1);
                    d++) // TODO; only work for 1D or 2D
                {
                    for(size_t i = 0; i < input_size; i++)
                    {
                        ot[data->node->oDist * b + d * output_size + i][0]
                            = tmp_mem[data->node->iDist * b + d * input_size + i][0];
                        ot[data->node->oDist * b + d * output_size + i][1]
                            = tmp_mem[data->node->iDist * b + d * input_size + i][1];

                        if(i > 0)
                        {
                            size_t mirror = output_size - i;
                            ot[data->node->oDist * b + d * output_size + mirror][0]
                                = tmp_mem[data->node->iDist * b + d * input_size + i][0];
                            ot[data->node->oDist * b + d * output_size + mirror][1]
                                = tmp_mem[data->node->iDist * b + d * input_size + i][1] * (-1);
                        }
                    }
                }
            }

            if(tmp_mem)
                free(tmp_mem);
        }
        else if(data->node->scheme == CS_KERNEL_CHIRP)
        {
            size_t N = data->node->length[0];
            size_t M = data->node->lengthBlue;
            chirp(N, M, data->node->direction, ot);
        }
        else if(data->node->scheme == CS_KERNEL_PAD_MUL)
        {
            CopyInputVector(data_p);

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            size_t N = data->node->length[0];
            size_t M = data->node->lengthBlue;

            local_fftwf_complex* chirp_mem
                = (local_fftwf_complex*)malloc(M * 2 * 2 * sizeof(float));
            chirp(N, M, data->node->direction, chirp_mem);

            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < M; i++)
                {
                    if(i < N)
                    {
                        float in_r = in[b * N + i][0];
                        float in_i = in[b * N + i][1];
                        float ch_r = chirp_mem[i][0];
                        float ch_i = chirp_mem[i][1];

                        ot[b * M + i][0] = in_r * ch_r + in_i * ch_i;
                        ot[b * M + i][1] = -in_r * ch_i + in_i * ch_r;
                    }
                    else
                    {
                        ot[b * M + i][0] = 0;
                        ot[b * M + i][1] = 0;
                    }
                }
            }

            if(chirp_mem)
                free(chirp_mem);
        }
        else if(data->node->scheme == CS_KERNEL_FFT_MUL)
        {
            size_t M = data->node->lengthBlue;
            size_t N = data->node->parent->length[0];

            CopyInputVector(data_p, M * 2 * 2 * sizeof(float));

            local_fftwf_complex* chirp_mem
                = (local_fftwf_complex*)malloc(M * 2 * 2 * sizeof(float));
            chirp_fft(N, M, data->node->direction, chirp_mem);

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < M; i++)
                {
                    float in_r = in[b * M + i][0];
                    float in_i = in[b * M + i][1];
                    float ch_r = chirp_mem[i][0];
                    float ch_i = chirp_mem[i][1];

                    ot[b * M + i][0] = in_r * ch_r - in_i * ch_i;
                    ot[b * M + i][1] = in_r * ch_i + in_i * ch_r;
                }
            }

            if(chirp_mem)
                free(chirp_mem);
        }
        else if(data->node->scheme == CS_KERNEL_RES_MUL)
        {
            size_t M = data->node->lengthBlue;
            size_t N = data->node->length[0];

            CopyInputVector(data_p, M * 2 * 2 * sizeof(float));

            local_fftwf_complex* chirp_mem
                = (local_fftwf_complex*)malloc(M * 2 * 2 * sizeof(float));
            chirp(N, M, data->node->direction, chirp_mem);

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            double MI = 1.0 / (double)M;
            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < N; i++)
                {
                    float in_r = in[b * N + i][0];
                    float in_i = in[b * N + i][1];
                    float ch_r = chirp_mem[i][0];
                    float ch_i = chirp_mem[i][1];

                    ot[b * N + i][0] = MI * (in_r * ch_r + in_i * ch_i);
                    ot[b * N + i][1] = MI * (-in_r * ch_i + in_i * ch_r);
                }
            }

            if(chirp_mem)
                free(chirp_mem);
        }
        else
        {
            // assert(false);
            // do not terminate the program but only tells not implemented
            printf("Not implemented\n");
        }
    }

public:
    RefLibOp(const void* data_p)
        : in(nullptr)
        , ot(nullptr)
        , lb(nullptr)
        , totalSize(0)
    {
        DataSetup(data_p);
        Execute(data_p);
    }

    void VerifyResult(const void* data_p)
    {
        DeviceCallIn* data           = (DeviceCallIn*)data_p;
        size_t        out_size_bytes = (data->node->oDist * data->node->batch) * sizeof(float);
        if(data->node->outArrayType == rocfft_array_type_real)
        {
            out_size_bytes *= 1;
        }
        else
        {
            out_size_bytes *= 2;
        }

        void* bufOut = data->bufOut[0];
        if(data->node->scheme == CS_KERNEL_CHIRP)
        {
            out_size_bytes *= 2;
        }
        else if((data->node->scheme == CS_KERNEL_FFT_MUL)
                || (data->node->scheme == CS_KERNEL_PAD_MUL))
        {
            bufOut = ((char*)bufOut + 2 * 2 * sizeof(float) * data->node->lengthBlue);
        }

        if(data->node->scheme == CS_KERNEL_COPY_CMPLX_TO_R)
        {
            return; // not implemented
        }

        local_fftwf_complex* tmp_mem = (local_fftwf_complex*)malloc(out_size_bytes);
        hipMemcpy(tmp_mem, bufOut, out_size_bytes, hipMemcpyDeviceToHost);

        if(data->node->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            hipMemcpy(lb,
                      tmp_mem,
                      out_size_bytes,
                      hipMemcpyHostToHost); // hermitan only works for batch=1, dense
            // packed cases
            return; // TODO
        }

        if(data->node->scheme == CS_KERNEL_TRANSPOSE)
        {
            std::vector<size_t> length_transpose_output;
            length_transpose_output.push_back(data->node->length[1]);
            length_transpose_output.push_back(data->node->length[0]);
            for(size_t i = 2; i < data->node->length.size(); i++)
                length_transpose_output.push_back(data->node->length[i]);
            CopyVector(lb,
                       tmp_mem,
                       data->node->batch,
                       data->node->oDist,
                       length_transpose_output,
                       data->node->outStride);
        }
        else if(data->node->scheme == CS_KERNEL_CHIRP)
        {
            std::vector<size_t> length_chirp;
            length_chirp.push_back(data->node->lengthBlue);
            CopyVector(lb,
                       tmp_mem,
                       2 * data->node->batch,
                       data->node->oDist,
                       length_chirp,
                       data->node->outStride);
        }
        else if(data->node->scheme == CS_KERNEL_PAD_MUL)
        {
            std::vector<size_t> length_ot;
            length_ot.push_back(data->node->lengthBlue);
            for(size_t i = 1; i < data->node->length.size(); i++)
                length_ot.push_back(data->node->length[i]);
            CopyVector(lb,
                       tmp_mem,
                       data->node->batch,
                       data->node->oDist,
                       length_ot,
                       data->node->outStride);
        }
        else
        {
            CopyVector(lb,
                       tmp_mem,
                       data->node->batch,
                       data->node->oDist,
                       data->node->length,
                       data->node->outStride);
        }

        double maxMag = 0.0;
        double rmse = 0.0, nrmse = 0.0;

        if(data->node->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            totalSize = totalSize / 2 + 1; // hermitan only check the first N/2 +1
            // elements; but only works for batch=1,
            // dense packed cases
        }

        // compare lb (library results) vs ot (CPU results)
        for(size_t i = 0; i < totalSize; i++)
        {
            double mag;
            double ex_r, ex_i, ac_r, ac_i;

            ac_r = lb[i][0];
            ac_i = lb[i][1];
            ex_r = ot[i][0];
            ex_i = ot[i][1];

            mag    = ex_r * ex_r + ex_i * ex_i;
            maxMag = (mag > maxMag) ? mag : maxMag;

            rmse += ((ex_r - ac_r) * (ex_r - ac_r) + (ex_i - ac_i) * (ex_i - ac_i));
        }

        maxMag = sqrt(maxMag);
        rmse   = sqrt(rmse / (double)totalSize);
        nrmse  = rmse / maxMag;

        std::cout << "rmse: " << rmse << std::endl << "nrmse: " << nrmse << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

#if 0 
        size_t beg_idx = 0;
        size_t end_idx = 7;
        std::cout << "input" << std::endl;
        for(size_t i=beg_idx; i<end_idx; i++)
        {
            //int *re = (int *)&in[i][0];
            //int *im = (int *)&in[i][1];
            //printf("%x, %x\n",*re,*im);
            std::cout << in[i][0] << ", " << in[i][1] << std::endl;
        }

        std::cout << "lib output" << std::endl;
        for(size_t i=beg_idx; i<end_idx; i++)
        {
            //int *re = (int *)&lb[i][0];
            //int *im = (int *)&lb[i][1];
            //printf("%x, %x\n",*re,*im);
            std::cout << lb[i][0] << ", " << lb[i][1] << std::endl;
        }

        std::cout << "fftw output" << std::endl;
        for(size_t i=beg_idx; i<end_idx; i++)
        {
            //int *re = (int *)&ot[i][0];
            //int *im = (int *)&ot[i][1];
            //printf("%x, %x\n",*re,*im);
            std::cout << ot[i][0] << ", " << ot[i][1] << std::endl;
        }
#endif

        if(tmp_mem)
            free(tmp_mem);
    }

    ~RefLibOp()
    {
        RefLibHandle&    refHandle = RefLibHandle::GetRefLibHandle();
        ftype_fftwf_free local_fftwf_free
            = (ftype_fftwf_free)dlsym(refHandle.fftw3f_lib, "fftwf_free");

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
