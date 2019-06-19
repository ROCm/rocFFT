/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#pragma once
#if !defined(TESTCONSTANTS_H)
#define TESTCONSTANTS_H

#include "rocfft.h"
#include <stdexcept>
#include <string>

enum data_pattern
{
    impulse,
    sawtooth,
    value,
    erratic
};

// Pre-callback function strings
#define PRE_MULVAL                                             \
    float2 mulval_pre(void* in, uint offset, void* userdata)\n \
    {                                                          \
        \n float  scalar = *((float*)userdata + offset);       \
        \n float2 ret    = *((float2*)in + offset) * scalar;   \
        \n return ret;                                         \
        \n                                                     \
    }

#define PRE_MULVAL_UDT                                            \
    typedef struct USER_DATA                                      \
    {                                                             \
        float scalar1;                                            \
        float scalar2;                                            \
    } USER_DATA;                                                  \
    \n float2 mulval_pre(void* in, uint offset, void* userdata)\n \
    {                                                             \
        \n USER_DATA* data   = ((USER_DATA*)userdata + offset);   \
        \n float      scalar = data->scalar1 * data->scalar2;     \
        \n float2     ret    = *((float2*)in + offset) * scalar;  \
        \n return ret;                                            \
        \n                                                        \
    }

#define PRE_MULVAL_DP                                           \
    double2 mulval_pre(void* in, uint offset, void* userdata)\n \
    {                                                           \
        \n double  scalar = *((double*)userdata + offset);      \
        \n double2 ret    = *((double2*)in + offset) * scalar;  \
        \n return ret;                                          \
        \n                                                      \
    }

#define PRE_MULVAL_PLANAR                                                    \
    float2 mulval_pre(void* inRe, void* inIm, uint offset, void* userdata)\n \
    {                                                                        \
        \n float  scalar = *((float*)userdata + offset);                     \
        \n float2 ret;                                                       \
        \n        ret.x = *((float*)inRe + offset) * scalar;                 \
        \n        ret.y = *((float*)inIm + offset) * scalar;                 \
        \n return ret;                                                       \
        \n                                                                   \
    }

#define PRE_MULVAL_PLANAR_DP                                                  \
    double2 mulval_pre(void* inRe, void* inIm, uint offset, void* userdata)\n \
    {                                                                         \
        \n double  scalar = *((double*)userdata + offset);                    \
        \n double2 ret;                                                       \
        \n         ret.x = *((double*)inRe + offset) * scalar;                \
        \n         ret.y = *((double*)inIm + offset) * scalar;                \
        \n return ret;                                                        \
        \n                                                                    \
    }

#define PRE_MULVAL_REAL                                       \
    float mulval_pre(void* in, uint offset, void* userdata)\n \
    {                                                         \
        \n float scalar = *((float*)userdata + offset);       \
        \n float ret    = *((float*)in + offset) * scalar;    \
        \n return ret;                                        \
        \n                                                    \
    }

#define PRE_MULVAL_REAL_DP                                     \
    double mulval_pre(void* in, uint offset, void* userdata)\n \
    {                                                          \
        \n double scalar = *((double*)userdata + offset);      \
        \n double ret    = *((double*)in + offset) * scalar;   \
        \n return ret;                                         \
        \n                                                     \
    }

// Precallback test for LDS - works when 1 WI works on one input element
#define PRE_MULVAL_LDS                                                                 \
    float2 mulval_pre(void* in, uint offset, void* userdata, __local void* localmem)\n \
    {                                                                                  \
        \n uint lid              = get_local_id(0);                                    \
        \n __local float* lds    = (__local float*)localmem + lid;                     \
        \n                lds[0] = *((float*)userdata + offset);                       \
        \n                __synthreads();                                              \
        \n float          prev = offset <= 0 ? 0 : *(lds - 1);                         \
        \n float          next = offset >= get_global_size(0) ? 0 : *(lds + 1);        \
        \n float          avg  = (prev + *lds + next) / 3.0f;                          \
        \n float2         ret  = *((float2*)in + offset) * avg;                        \
        \n return ret;                                                                 \
        \n                                                                             \
    }

// Post-callback function strings
#define POST_MULVAL                                                                    \
    void mulval_post(void* output, uint outoffset, void* userdata, float2 fftoutput)\n \
    {                                                                                  \
        \n float scalar                  = *((float*)userdata + outoffset);            \
        \n*((float2*)output + outoffset) = fftoutput * scalar;                         \
        \n                                                                             \
    }

#define POST_MULVAL_DP                                                                  \
    void mulval_post(void* output, uint outoffset, void* userdata, double2 fftoutput)\n \
    {                                                                                   \
        \n double scalar                  = *((double*)userdata + outoffset);           \
        \n*((double2*)output + outoffset) = fftoutput * scalar;                         \
        \n                                                                              \
    }

#define POST_MULVAL_PLANAR                                                   \
    void mulval_post(void*  outputRe,                                        \
                     void*  outputIm,                                        \
                     size_t outoffset,                                       \
                     void*  userdata,                                        \
                     float  fftoutputRe,                                     \
                     float  fftoutputIm)\n                                    \
    {                                                                        \
        \n float scalar                   = *((float*)userdata + outoffset); \
        \n*((float*)outputRe + outoffset) = fftoutputRe * scalar;            \
        \n*((float*)outputIm + outoffset) = fftoutputIm * scalar;            \
        \n                                                                   \
    }

#define POST_MULVAL_PLANAR_DP                                                  \
    void mulval_post(void*  outputRe,                                          \
                     void*  outputIm,                                          \
                     size_t outoffset,                                         \
                     void*  userdata,                                          \
                     double fftoutputRe,                                       \
                     double fftoutputIm)\n                                     \
    {                                                                          \
        \n double scalar                   = *((double*)userdata + outoffset); \
        \n*((double*)outputRe + outoffset) = fftoutputRe * scalar;             \
        \n*((double*)outputIm + outoffset) = fftoutputIm * scalar;             \
        \n                                                                     \
    }

// Postcallback test for LDS - works when 1 WI works on one element.
// Assumes 1D FFT of length 64.
#define POST_MULVAL_LDS                                                                            \
    void mulval_post(                                                                              \
        void* output, uint outoffset, void* userdata, float2 fftoutput, __local void* localmem)\n  \
    {                                                                                              \
        \n uint lid = get_local_id(0);                                                             \
        \n __local float* lds;                                                                     \
        \n if(outoffset < 16) \n                                                                   \
        {                                                                                          \
            \n lds    = (__local float*)localmem + lid * 4;                                        \
            \n lds[0] = *((float*)userdata + lid * 4);                                             \
            \n lds[1] = *((float*)userdata + lid * 4 + 1);                                         \
            \n lds[2] = *((float*)userdata + lid * 4 + 2);                                         \
            \n lds[3] = *((float*)userdata + lid * 4 + 3);                                         \
            \n                                                                                     \
        }                                                                                          \
        \n       __synthreads();                                                                   \
        \n       lds                     = (__local float*)localmem + outoffset;                   \
        \n float prev                    = outoffset <= 0 ? 0 : *(lds - 1);                        \
        \n float next                    = outoffset >= (get_global_size(0) - 1) ? 0 : *(lds + 1); \
        \n float avg                     = (prev + *lds + next) / 3.0f;                            \
        \n*((float2*)output + outoffset) = fftoutput * avg;                                        \
        \n                                                                                         \
    }

#define POST_MULVAL_REAL                                                              \
    void mulval_post(void* output, uint outoffset, void* userdata, float fftoutput)\n \
    {                                                                                 \
        \n float scalar                 = *((float*)userdata + outoffset);            \
        \n*((float*)output + outoffset) = fftoutput * scalar;                         \
        \n                                                                            \
    }

#define POST_MULVAL_REAL_DP                                                            \
    void mulval_post(void* output, uint outoffset, void* userdata, double fftoutput)\n \
    {                                                                                  \
        \n double scalar                 = *((double*)userdata + outoffset);           \
        \n*((double*)output + outoffset) = fftoutput * scalar;                         \
        \n                                                                             \
    }

typedef struct USER_DATA
{
    float scalar1;
    float scalar2;
} USER_DATA;

#define CALLBCKSTR(...) #__VA_ARGS__
#define STRINGIFY(...) CALLBCKSTR(__VA_ARGS__)

enum
{
    REAL = 0,
    IMAG = 1
};
enum
{
    dimx = 0,
    dimy = 1,
    dimz = 2
};
enum fftw_dim
{
    one_d   = 1,
    two_d   = 2,
    three_d = 3
};
enum
{
    one_interleaved_buffer              = 1,
    separate_real_and_imaginary_buffers = 2
};
const bool  use_explicit_intermediate_buffer = true;
const bool  autogenerate_intermediate_buffer = false;
const bool  pointwise_compare                = true;
const bool  root_mean_square                 = false;
extern bool comparison_type;
extern bool suppress_output;

// this thing is horrible. horrible! i am not proud.
extern size_t super_duper_global_seed;

const size_t small2  = 32;
const size_t normal2 = 1024;
const size_t large2  = 8192;
const size_t dlarge2 = 4096;

const size_t small3  = 9;
const size_t normal3 = 729;
const size_t large3  = 6561;
const size_t dlarge3 = 2187;

const size_t small5  = 25;
const size_t normal5 = 625;
const size_t large5  = 15625;
const size_t dlarge5 = 3125;

const size_t small7  = 49;
const size_t normal7 = 343;
const size_t large7  = 16807;
const size_t dlarge7 = 2401;

const size_t large_batch_size                       = 2048;
const size_t do_not_output_any_mismatches           = 0;
const size_t default_number_of_mismatches_to_output = 10;
const size_t max_dimension                          = 3;

const double magnitude_lower_limit = 1.0E-100;

extern float  tolerance;
extern double rmse_tolerance;

extern size_t number_of_random_tests;
extern time_t random_test_parameter_seed;
extern bool   verbose;

void handle_exception(const std::exception& except);

// Creating this template function and specializations to control the length
// inputs to the tests;
// these should be removed once the size restriction on transfrom lengths (SP
// 2^24 and DP 2^22)
// is removed; the dlarge* constants can then be removed

template <typename T>
inline size_t MaxLength2D(size_t rad)
{
    return 0;
}

template <>
inline size_t MaxLength2D<float>(size_t rad)
{
    switch(rad)
    {
    case 2:
        return large2;
    case 3:
        return large3;
    case 5:
        return large5;
    case 7:
        return large7;
    default:
        return 0;
    }
}

template <>
inline size_t MaxLength2D<double>(size_t rad)
{
    switch(rad)
    {
    case 2:
        return dlarge2;
    case 3:
        return dlarge3;
    case 5:
        return dlarge5;
    case 7:
        return dlarge7;
    default:
        return 0;
    }
}

#endif
