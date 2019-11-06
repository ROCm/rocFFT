// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef REF_CPU_H
#define REF_CPU_H

#ifdef REF_DEBUG

#include "hipfft.h"
#include <complex>
#include <cstdio>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <numeric>
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

    // delete is a c++11 feature, prohibit copy constructor:
    RefLibHandle(const RefLibHandle&) = delete;
    // prohibit assignment operator:
    RefLibHandle& operator=(const RefLibHandle&) = delete;

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

class fftwbuf
{
private:
    void* (*local_fftwf_malloc)(const size_t);
    void (*local_fftwf_free)(void*);
    void freedata()
    {
        (*local_fftwf_free)(data);
    }

public:
    fftwbuf()
        : data(NULL)
        , size(0)
        , typesize(0)
    {
        RefLibHandle& refHandle = RefLibHandle::GetRefLibHandle();
        local_fftwf_malloc      = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");
        local_fftwf_free        = (ftype_fftwf_free)dlsym(refHandle.fftw3f_lib, "fftwf_free");
    };
    fftwbuf(const size_t size0, const size_t typesize0)
        : data(0)
        , size(size0)
        , typesize(typesize0)
    {
        RefLibHandle& refHandle = RefLibHandle::GetRefLibHandle();
        local_fftwf_malloc      = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");
        local_fftwf_free        = (ftype_fftwf_free)dlsym(refHandle.fftw3f_lib, "fftwf_free");

        alloc(size0, typesize0);
    };
    ~fftwbuf()
    {
        if(data)
        {
            freedata();
        }
    };
    void alloc(const size_t size0, const size_t typesize0)
    {
        size     = size0;
        typesize = typesize0;
        if(data != NULL)
        {
            freedata();
        }
        data = (void*)(*local_fftwf_malloc)(typesize * size);
        assert(data != NULL);
    }
    size_t bufsize()
    {
        return size * typesize;
    };
    void*  data;
    size_t size;
    size_t typesize;
};

class RefLibOp
{
    // input from fftw:
    fftwbuf fftwin;

    // output from fftw:
    fftwbuf fftwout;

    // output from lib:
    fftwbuf libout;

    void DataSetup(const void* data_p)
    {
        RefLibHandle& refHandle = RefLibHandle::GetRefLibHandle();

        DeviceCallIn* data = (DeviceCallIn*)data_p;

        size_t totalSize;

        size_t in_typesize = (data->node->inArrayType == rocfft_array_type_real)
                                 ? sizeof(float)
                                 : sizeof(std::complex<float>);
        size_t out_typesize = (data->node->outArrayType == rocfft_array_type_real)
                                  ? sizeof(float)
                                  : sizeof(std::complex<float>);

        size_t     insize  = 0;
        size_t     outsize = 0;
        const auto batch   = data->node->batch;

        // TODO: what about strides, etc?
        switch(data->node->scheme)
        {
        case CS_KERNEL_CHIRP:
            insize  = 2 * data->node->lengthBlue;
            outsize = insize;
            break;
        case CS_KERNEL_FFT_MUL:
        case CS_KERNEL_PAD_MUL:
        case CS_KERNEL_RES_MUL:
            // NB: the Bluestein length is the first dimesion
            insize = std::accumulate(data->node->length.begin() + 1,
                                     data->node->length.end(),
                                     batch,
                                     std::multiplies<size_t>());
            insize *= data->node->lengthBlue;
            outsize = insize;
            break;
        case CS_KERNEL_R_TO_CMPLX:
            insize      = batch * data->node->length[0];
            outsize     = batch * (data->node->length[0] + 1);
            in_typesize = sizeof(std::complex<float>);
            break;
        case CS_KERNEL_CMPLX_TO_R:
            insize       = batch * (data->node->length[0] + 1);
            outsize      = batch * data->node->length[0];
            in_typesize  = sizeof(std::complex<float>);
            out_typesize = sizeof(std::complex<float>);
            break;
        default:
            insize  = std::accumulate(data->node->length.begin(),
                                     data->node->length.end(),
                                     batch,
                                     std::multiplies<size_t>());
            outsize = insize;
        }

        assert(insize > 0);
        assert(outsize > 0);

        fftwin.alloc(insize, in_typesize);
        fftwout.alloc(outsize, out_typesize);
        libout.alloc(outsize, out_typesize);

        memset(fftwin.data, 0x40, fftwin.bufsize());
        memset(fftwout.data, 0x40, fftwout.bufsize());
        memset(libout.data, 0x40, libout.bufsize());

#if 0
        // Initialize the code to some known value to help debug the
        // cpu reference implementation.
        std::complex<float>* input = (std::complex<float>*)fftwin.data;
        for(int r = 0; r < fftwin.size; ++r)
        {
            input[r] = std::complex<float>(r + 0.5, r * r + 3);
        }
#endif
    }

    // Copy a host vector to a host vector, taking strides and other
    // data layout elements into account.
    void CopyVector(local_fftwf_complex* dst,
                    local_fftwf_complex* src,
                    size_t               batch,
                    size_t               dist,
                    std::vector<size_t>  length,
                    std::vector<size_t>  stride)
    {
        size_t lenSize
            = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());

        size_t b = 0;
        while(b < batch)
        {
            size_t offset_dst   = 0;
            size_t offset_src   = 0;
            size_t offset_src_d = 0;
            size_t pos          = 0;
            bool   obreak       = false;

            std::vector<size_t> current;
            current.assign(length.size(), 0);

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

        if(data->node->inArrayType != rocfft_array_type_real)
        {
            in_size_bytes *= 2;
        }

        void*   buf = ((char*)data->bufIn[0] + offset);
        fftwbuf tmp_mem(data->node->iDist * data->node->batch, sizeof(std::complex<float>));
        hipMemcpy(tmp_mem.data, buf, in_size_bytes, hipMemcpyDeviceToHost);

        CopyVector((local_fftwf_complex*)fftwin.data,
                   (local_fftwf_complex*)tmp_mem.data,
                   data->node->batch,
                   data->node->iDist,
                   data->node->length,
                   data->node->inStride);
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

        int n[1] = {static_cast<int>(M)};

        void* p = local_fftwf_plan_many_dft(1,
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
                                            (dir == -1) ? LOCAL_FFTW_FORWARD : LOCAL_FFTW_BACKWARD,
                                            LOCAL_FFTW_ESTIMATE);
        chirp(N, M, dir, vec);
        local_fftwf_execute(p);
        local_fftwf_destroy_plan(p);
    }

    void Execute(const void* data_p)
    {
        DeviceCallIn* data = (DeviceCallIn*)data_p;

        switch(data->node->scheme)
        {
        case CS_KERNEL_STOCKHAM:
        {
            RefLibHandle&             refHandle = RefLibHandle::GetRefLibHandle();
            ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft
                = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
            ftype_fftwf_execute local_fftwf_execute
                = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
            ftype_fftwf_destroy_plan local_fftwf_destroy_plan
                = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");

            int n[1]    = {static_cast<int>(data->node->length[0])};
            int howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            void* p = local_fftwf_plan_many_dft(1,
                                                n,
                                                howmany,
                                                (local_fftwf_complex*)fftwin.data,
                                                NULL,
                                                1,
                                                n[0],
                                                (local_fftwf_complex*)fftwout.data,
                                                NULL,
                                                1,
                                                n[0],
                                                (data->node->direction == -1) ? LOCAL_FFTW_FORWARD
                                                                              : LOCAL_FFTW_BACKWARD,
                                                LOCAL_FFTW_ESTIMATE);
            CopyInputVector(data_p);
            local_fftwf_execute(p);
            local_fftwf_destroy_plan(p);
        }
        break;
        case CS_KERNEL_TRANSPOSE:
        {
            // TODO: what about the real transpose case?
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            std::complex<float>* in = (std::complex<float>*)fftwin.data;

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
                            ot[b * rows * cols + j * rows + i] = in[b * rows * cols + i * cols + j];
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

                            in_v.x = in[b * rows * cols + i * cols + j].real();
                            in_v.y = in[b * rows * cols + i * cols + j].imag();

                            ot_v = TwMul(twtc, twl, data->node->direction, in_v, i * j);

                            ot[b * rows * cols + j * rows + i].real(ot_v.x);
                            ot[b * rows * cols + j * rows + i].imag(ot_v.y);
                        }
                    }
                }
            }
        }
        break;
        case CS_KERNEL_COPY_R_TO_CMPLX:
        {
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            size_t in_size_bytes    = (data->node->iDist * data->node->batch) * sizeof(float);

            fftwbuf tmp_mem(data->node->iDist * data->node->batch, sizeof(std::complex<float>));

            hipMemcpy(tmp_mem.data, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            std::complex<float>* tmp_data = (std::complex<float>*)tmp_mem.data;

            size_t elements = 1;
            for(size_t d = 0; d < data->node->length.size(); d++)
            {
                elements *= data->node->length[d];
            }
            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t i = 0; i < elements; i++)
                {
                    ot[data->node->oDist * b + i]
                        = std::complex<float>(tmp_data[data->node->iDist * b + i].real(), 0.0);
                }
            }
        }
        break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
        {
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            // assump the input is complex, the output is hermitian on take the first
            // [N/2 + 1] elements
            size_t in_size_bytes = (data->node->iDist * data->node->batch) * 2 * sizeof(float);

            fftwbuf tmp_mem(data->node->iDist * data->node->batch, sizeof(std::complex<float>));

            hipMemcpy(tmp_mem.data, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            std::complex<float>* tmp_data = (std::complex<float>*)tmp_mem.data;

            size_t elements = 1;
            elements *= data->node->length[0] / 2 + 1;
            for(size_t d = 1; d < data->node->length.size(); d++)
            {
                elements *= data->node->length[d];
            }

            std::cout << "iDist: " << data->node->iDist << " Dist: " << data->node->oDist
                      << " in complex2hermitian kernel\n";
            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t i = 0; i < elements; i++) // TODO: only work for 1D cases
                {
                    ot[data->node->oDist * b + i] = tmp_data[data->node->iDist * b + i];
                }
            }
        }
        break;
        case CS_KERNEL_COPY_HERM_TO_CMPLX:
        {
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            // assump the input is hermitian, the output is complex on take the first
            // [N/2 + 1] elements
            size_t in_size_bytes = (data->node->iDist * data->node->batch) * 2 * sizeof(float);

            fftwbuf tmp_mem(data->node->iDist * data->node->batch, sizeof(std::complex<float>));

            hipMemcpy(tmp_mem.data, data->bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            std::complex<float>* tmp_data = (std::complex<float>*)tmp_mem.data;

            size_t output_size = data->node->length[0];
            size_t input_size  = output_size / 2 + 1;

            std::cout << "iDist: " << data->node->iDist << " Dist: " << data->node->oDist
                      << " in hermitian2complex kernel\n";

            for(size_t b = 0; b < data->node->batch; b++)
            {
                for(size_t d = 0;
                    d < (data->node->length.size() == 2 ? (data->node->length[1]) : 1);
                    d++) // TODO; only work for 1D or 2D
                {
                    for(size_t i = 0; i < input_size; i++)
                    {
                        ot[data->node->oDist * b + d * output_size + i]
                            = tmp_data[data->node->iDist * b + d * input_size + i];

                        if(i > 0)
                        {
                            size_t mirror = output_size - i;
                            ot[data->node->oDist * b + d * output_size + mirror]
                                = tmp_data[data->node->iDist * b + d * input_size + i];
                        }
                    }
                }
            }
        }
        break;
        case CS_KERNEL_R_TO_CMPLX:
        {
            // Post-processing stage of 1D real-to-complex transform, out-of-place
            const size_t halfN = data->node->length[0];
            const size_t batch = data->node->batch;

            assert(fftwin.size == batch * halfN);
            assert(fftwout.size == batch * (halfN + 1));

            const auto           input  = (std::complex<float>*)fftwin.data;
            std::complex<float>* output = (std::complex<float>*)fftwout.data;

            size_t output_idx_base = 0;

            const std::complex<float> I(0, 1);
            const std::complex<float> one(1, 0);
            const std::complex<float> half(0.5, 0);

            const float overN = 0.5 / halfN;

            for(int ibatch = 0; ibatch < batch; ++ibatch)
            {
                const auto bin  = input + ibatch * halfN;
                auto       bout = output + ibatch * (halfN + 1);
                bout[0]         = std::complex<float>(bin[0].real() + bin[0].imag());
                for(int r = 1; r < halfN; ++r)
                {
                    const auto omegaNr
                        = std::exp(std::complex<float>(0.0f, (float)(-2.0f * M_PI * r * overN)));
                    bout[r] = bin[r] * half * (one - I * omegaNr)
                              + conj(bin[halfN - r]) * half * (one + I * omegaNr);
                }
            }
            output[output_idx_base + halfN]
                = std::complex<float>(input[0].real() - input[0].imag(), 0);
        }
        break;
        case CS_KERNEL_CMPLX_TO_R:
        {
            // Pre-processing stage of 1D complex-to-real transform, out-of-place
            const size_t halfN = data->node->length[0];
            const size_t batch = data->node->batch;

            assert(fftwin.size == batch * (halfN + 1));
            assert(fftwout.size == batch * halfN);
            assert(fftwin.typesize == sizeof(std::complex<float>));
            assert(fftwout.typesize == sizeof(std::complex<float>));

            const std::complex<float>* input  = (std::complex<float>*)fftwin.data;
            std::complex<float>*       output = (std::complex<float>*)fftwout.data;

            const float               overN = 0.5 / halfN;
            const std::complex<float> I(0, 1);
            const std::complex<float> one(1, 0);

            for(int ibatch = 0; ibatch < batch; ++ibatch)
            {
                const auto bin  = input + ibatch * (halfN + 1);
                auto       bout = output + ibatch * halfN;
                for(int r = 0; r < halfN; ++r)
                {
                    const auto omegaNr = std::exp(std::complex<float>(0, 2.0 * M_PI * r * overN));
                    bout[r]
                        = bin[r] * (one + I * omegaNr) + conj(bin[halfN - r]) * (one - I * omegaNr);
                }
            }
        }
        break;
        case CS_KERNEL_CHIRP:
        {
            size_t N = data->node->length[0];
            size_t M = data->node->lengthBlue;
            chirp(N, M, data->node->direction, (local_fftwf_complex*)fftwout.data);
        }
        break;
        case CS_KERNEL_PAD_MUL:
        {
            std::complex<float>* in = (std::complex<float>*)fftwin.data;
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            CopyInputVector(data_p);

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            size_t N = data->node->length[0];
            size_t M = data->node->lengthBlue;

            fftwbuf chirp_mem(M * 2, sizeof(std::complex<float>));

            chirp(N, M, data->node->direction, (local_fftwf_complex*)chirp_mem.data);

            std::complex<float>* chirp_data = (std::complex<float>*)chirp_mem.data;

            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < M; i++)
                {
                    if(i < N)
                    {
                        float in_r = in[b * N + i].real();
                        float in_i = in[b * N + i].imag();
                        float ch_r = chirp_data[i].real();
                        float ch_i = chirp_data[i].imag();

                        ot[b * M + i].real(in_r * ch_r + in_i * ch_i);
                        ot[b * M + i].imag(-in_r * ch_i + in_i * ch_r);
                    }
                    else
                    {
                        ot[b * M + i].real(0);
                        ot[b * M + i].imag(0);
                    }
                }
            }
        }
        break;
        case CS_KERNEL_FFT_MUL:
        {
            std::complex<float>* in = (std::complex<float>*)fftwin.data;
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            size_t               M  = data->node->lengthBlue;
            size_t               N  = data->node->parent->length[0];

            CopyInputVector(data_p, M * 2 * 2 * sizeof(float));

            fftwbuf chirp_mem(M * 2, sizeof(std::complex<float>));

            chirp_fft(N, M, data->node->direction, (local_fftwf_complex*)chirp_mem.data);

            std::complex<float>* chirp_data = (std::complex<float>*)chirp_mem.data;

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < M; i++)
                {
                    float in_r = in[b * M + i].real();
                    float in_i = in[b * M + i].imag();
                    float ch_r = chirp_data[i].real();
                    float ch_i = chirp_data[i].imag();

                    ot[b * M + i].real(in_r * ch_r - in_i * ch_i);
                    ot[b * M + i].imag(in_r * ch_i + in_i * ch_r);
                }
            }
        }
        break;
        case CS_KERNEL_RES_MUL:
        {
            std::complex<float>* in = (std::complex<float>*)fftwin.data;
            std::complex<float>* ot = (std::complex<float>*)fftwout.data;
            size_t               M  = data->node->lengthBlue;
            size_t               N  = data->node->length[0];

            CopyInputVector(data_p, M * 2 * 2 * sizeof(float));

            fftwbuf chirp_mem(M * 2, sizeof(std::complex<float>));

            chirp(N, M, data->node->direction, (local_fftwf_complex*)chirp_mem.data);

            std::complex<float>* chirp_data = (std::complex<float>*)chirp_mem.data;

            size_t howmany = data->node->batch;
            for(size_t i = 1; i < data->node->length.size(); i++)
                howmany *= data->node->length[i];

            double MI = 1.0 / (double)M;
            for(size_t b = 0; b < howmany; b++)
            {
                for(size_t i = 0; i < N; i++)
                {
                    float in_r = in[b * N + i].real();
                    float in_i = in[b * N + i].imag();
                    float ch_r = chirp_data[i].real();
                    float ch_i = chirp_data[i].imag();

                    ot[b * N + i].real(MI * (in_r * ch_r + in_i * ch_i));
                    ot[b * N + i].imag(MI * (-in_r * ch_i + in_i * ch_r));
                }
            }
        }
        break;
        default:
            // assert(false);
            // do not terminate the program but only tells not implemented
            std::cout << "Not implemented\n";
        }
    }

public:
    RefLibOp(const void* data_p)
    {
        DataSetup(data_p);
        Execute(data_p);
    }

    void VerifyResult(const void* data_p)
    {
        DeviceCallIn* data        = (DeviceCallIn*)data_p;
        size_t        out_size    = (data->node->oDist * data->node->batch);
        size_t        checklength = data->node->length[0];
        for(int i = 1; i < data->node->length.size(); ++i)
        {
            checklength *= data->node->length[i];
        }
        void* bufOut = data->bufOut[0];

        switch(data->node->scheme)
        {
        case CS_KERNEL_CHIRP:
            out_size *= 2;
            break;
        case CS_KERNEL_FFT_MUL:
        case CS_KERNEL_PAD_MUL:
            // TODO: document
            bufOut = ((char*)bufOut + 2 * 2 * sizeof(float) * data->node->lengthBlue);
            break;
        case CS_KERNEL_COPY_CMPLX_TO_R:
        case CS_KERNEL_COPY_HERM_TO_CMPLX:
        case CS_KERNEL_STOCKHAM_BLOCK_RC:
        case CS_KERNEL_STOCKHAM_BLOCK_CC:
            return; // not implemented
            break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
            checklength = libout.size / 2 + 1;
            break;
        case CS_KERNEL_CMPLX_TO_R:
            checklength = libout.size / 2;
            break;
        default:
            break;
        }

        fftwbuf tmp_mem(out_size, sizeof(std::complex<float>));

        // Copy the device information to out local buffer:
        hipMemcpy(tmp_mem.data, bufOut, tmp_mem.bufsize(), hipMemcpyDeviceToHost);

        switch(data->node->scheme)
        {
        case CS_KERNEL_TRANSPOSE:
        {
            std::vector<size_t> length_transpose_output;
            length_transpose_output.push_back(data->node->length[1]);
            length_transpose_output.push_back(data->node->length[0]);
            for(size_t i = 2; i < data->node->length.size(); i++)
                length_transpose_output.push_back(data->node->length[i]);
            CopyVector((local_fftwf_complex*)libout.data,
                       (local_fftwf_complex*)tmp_mem.data,
                       data->node->batch,
                       data->node->oDist,
                       length_transpose_output,
                       data->node->outStride);
        }
        break;
        case CS_KERNEL_CHIRP:
        {
            std::vector<size_t> length_chirp;
            length_chirp.push_back(data->node->lengthBlue);
            CopyVector((local_fftwf_complex*)libout.data,
                       (local_fftwf_complex*)tmp_mem.data,
                       2 * data->node->batch,
                       data->node->oDist,
                       length_chirp,
                       data->node->outStride);
        }
        break;
        case CS_KERNEL_PAD_MUL:
        {
            std::vector<size_t> length_ot;
            length_ot.push_back(data->node->lengthBlue);
            for(size_t i = 1; i < data->node->length.size(); i++)
                length_ot.push_back(data->node->length[i]);
            CopyVector((local_fftwf_complex*)libout.data,
                       (local_fftwf_complex*)tmp_mem.data,
                       data->node->batch,
                       data->node->oDist,
                       length_ot,
                       data->node->outStride);
        }
        break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
            hipMemcpy(libout.data,
                      tmp_mem.data,
                      tmp_mem.bufsize(),
                      hipMemcpyHostToHost); // hermitan only works for batch=1, dense
            // packed cases
            //assert(batch == 1);
            return; // TODO
            break;
        default:
        {
            CopyVector((local_fftwf_complex*)libout.data,
                       (local_fftwf_complex*)tmp_mem.data,
                       data->node->batch,
                       data->node->oDist,
                       data->node->length,
                       data->node->outStride);
        }
        }

        double maxMag = 0.0;
        double rmse   = 0.0;

        // compare library results vs CPU results
        // TODO: what about real-valued outputs?
        std::complex<float>* lb = (std::complex<float>*)libout.data;
        std::complex<float>* ot = (std::complex<float>*)fftwout.data;
        for(size_t i = 0; i < checklength; i++)
        {
            double ac_r = lb[i].real();
            double ac_i = lb[i].imag();
            double ex_r = ot[i].real();
            double ex_i = ot[i].imag();

            double mag = ex_r * ex_r + ex_i * ex_i;
            maxMag     = (mag > maxMag) ? mag : maxMag;

            rmse += ((ex_r - ac_r) * (ex_r - ac_r) + (ex_i - ac_i) * (ex_i - ac_i));
        }

        maxMag       = sqrt(maxMag);
        rmse         = sqrt(rmse / (double)checklength);
        double nrmse = rmse / maxMag;

        std::cout << "rmse: " << rmse << std::endl << "nrmse: " << nrmse << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

#if 0
        std::complex<float>* in      = (std::complex<float>*)fftwin.data;

        std::cout << "input:" << std::endl;
        for(size_t i = 0; i < fftwin.size; ++i)
        {
            std::cout << i << "\t(" << in[i].real() << ", " << in[i].imag() << ")\n";
        }

        std::cout << "lib output vs cpu output:" << std::endl;
        for(size_t i = 0; i < libout.size; ++i)
        {
            std::cout << i << "\t(" << lb[i].real() << "," << lb[i].imag() << ")"
                      << "\t(" << ot[i].real() << "," << ot[i].imag() << ")\n";
        }
#endif
    }
};

#endif // REF_DEBUG

#endif // REF_CPU_H
