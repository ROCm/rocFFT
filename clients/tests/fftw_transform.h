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

#pragma once
#if !defined(FFTWTRANSFORM_H)
#define FFTWTRANSFORM_H

#include "test_constants.h"
#include <complex>
#include <fftw3.h>
#include <vector>

enum data_pattern
{
    impulse,
    sawtooth,
    value,
    erratic
};

enum fftw_direction
{
    forward  = -1,
    backward = +1
};

enum fftw_transform_type
{
    c2c,
    r2c,
    c2r
};

// Function to return maximum error for float and double types.
template <typename Tfloat>
inline double type_epsilon();
template <>
inline double type_epsilon<float>()
{
    return 1e-6;
}
template <>
inline double type_epsilon<double>()
{
    return 1e-7;
}

// C++ traits to translate float->fftwf_complex and
// double->fftw_complex.
// The correct FFTW complex type can be accessed via, for example,
// using complex_t = typename fftw_complex_trait<Tfloat>::complex_t;
template <typename Tfloat>
struct fftw_trait;
template <>
struct fftw_trait<float>
{
    using fftw_complex_type = fftwf_complex;
    using fftw_plan_type    = fftwf_plan;
};
template <>
struct fftw_trait<double>
{
    using fftw_complex_type = fftw_complex;
    using fftw_plan_type    = fftw_plan;
};

// Template wrappers for real-valued FFTW allocators:
template <typename Tfloat>
inline Tfloat* fftw_alloc_real_type(size_t n);
template <>
inline float* fftw_alloc_real_type<float>(size_t n)
{
    return fftwf_alloc_real(n);
}
template <>
inline double* fftw_alloc_real_type<double>(size_t n)
{
    return fftw_alloc_real(n);
}

// Template wrappers for complex-valued FFTW allocators:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_complex_type* fftw_alloc_complex_type(size_t n);
template <>
inline typename fftw_trait<float>::fftw_complex_type* fftw_alloc_complex_type<float>(size_t n)
{
    return fftwf_alloc_complex(n);
}
template <>
inline typename fftw_trait<double>::fftw_complex_type* fftw_alloc_complex_type<double>(size_t n)
{
    return fftw_alloc_complex(n);
}

template <typename fftw_type>
inline fftw_type* fftw_alloc_type(size_t n);
template <>
inline float* fftw_alloc_type<float>(size_t n)
{
    return fftw_alloc_real_type<float>(n);
}
template <>
inline double* fftw_alloc_type<double>(size_t n)
{
    return fftw_alloc_real_type<double>(n);
}
template <>
inline fftwf_complex* fftw_alloc_type<fftwf_complex>(size_t n)
{
    return fftw_alloc_complex_type<float>(n);
}
template <>
inline fftw_complex* fftw_alloc_type<fftw_complex>(size_t n)
{
    return fftw_alloc_complex_type<double>(n);
}
template <>
inline std::complex<float>* fftw_alloc_type<std::complex<float>>(size_t n)
{
    return (std::complex<float>*)fftw_alloc_complex_type<float>(n);
}
template <>
inline std::complex<double>* fftw_alloc_type<std::complex<double>>(size_t n)
{
    return (std::complex<double>*)fftw_alloc_complex_type<double>(n);
}

// Allocator / deallocator for FFTW arrays.
template <typename fftw_type>
class fftw_allocator : public std::allocator<fftw_type>
{
public:
    template <typename U>
    struct rebind
    {
        typedef fftw_allocator<fftw_type> other;
    };
    fftw_type* allocate(size_t n)
    {
        return fftw_alloc_type<fftw_type>(n);
    }
    void deallocate(fftw_type* data, std::size_t size)
    {
        fftw_free(data);
    }
};

template <typename fftw_type>
using fftw_vector = std::vector<fftw_type, fftw_allocator<fftw_type>>;

// Template wrappers for FFTW plan executors:
template <typename Tfloat>
inline void fftw_execute_type(typename fftw_trait<Tfloat>::fftw_plan_type plan);
template <>
inline void fftw_execute_type<float>(typename fftw_trait<float>::fftw_plan_type plan)
{
    return fftwf_execute(plan);
}
template <>
inline void fftw_execute_type<double>(typename fftw_trait<double>::fftw_plan_type plan)
{
    return fftw_execute(plan);
}

// Template wrappers for FFTW plan destroyers:
template <typename Tfftw_plan>
inline void fftw_destroy_plan_type(Tfftw_plan plan);
template <>
inline void fftw_destroy_plan_type<fftwf_plan>(fftwf_plan plan)
{
    return fftwf_destroy_plan(plan);
}
template <>
inline void fftw_destroy_plan_type<fftw_plan>(fftw_plan plan)
{
    return fftw_destroy_plan(plan);
}

// Template wrappers for FFTW c2c planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_dft(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         typename fftw_trait<Tfloat>::fftw_complex_type* in,
                         typename fftw_trait<Tfloat>::fftw_complex_type* out,
                         int                                             sign,
                         unsigned                                        flags);
template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_dft<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                typename fftw_trait<float>::fftw_complex_type* in,
                                typename fftw_trait<float>::fftw_complex_type* out,
                                int                                            sign,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}
template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_dft<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 typename fftw_trait<double>::fftw_complex_type* in,
                                 typename fftw_trait<double>::fftw_complex_type* out,
                                 int                                             sign,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}

// Template wrappers for FFTW r2c planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_r2c(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         Tfloat*                                         in,
                         typename fftw_trait<Tfloat>::fftw_complex_type* out,
                         unsigned                                        flags);
template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_r2c<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                float*                                         in,
                                typename fftw_trait<float>::fftw_complex_type* out,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}
template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_r2c<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 double*                                         in,
                                 typename fftw_trait<double>::fftw_complex_type* out,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}

// Template wrappers for FFTW c2r planners:
template <typename Tfloat>
inline typename fftw_trait<Tfloat>::fftw_plan_type
    fftw_plan_guru64_c2r(int                                             rank,
                         const fftw_iodim64*                             dims,
                         int                                             howmany_rank,
                         const fftw_iodim64*                             howmany_dims,
                         typename fftw_trait<Tfloat>::fftw_complex_type* in,
                         Tfloat*                                         out,
                         unsigned                                        flags);
template <>
inline typename fftw_trait<float>::fftw_plan_type
    fftw_plan_guru64_c2r<float>(int                                            rank,
                                const fftw_iodim64*                            dims,
                                int                                            howmany_rank,
                                const fftw_iodim64*                            howmany_dims,
                                typename fftw_trait<float>::fftw_complex_type* in,
                                float*                                         out,
                                unsigned                                       flags)
{
    return fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}
template <>
inline typename fftw_trait<double>::fftw_plan_type
    fftw_plan_guru64_c2r<double>(int                                             rank,
                                 const fftw_iodim64*                             dims,
                                 int                                             howmany_rank,
                                 const fftw_iodim64*                             howmany_dims,
                                 typename fftw_trait<double>::fftw_complex_type* in,
                                 double*                                         out,
                                 unsigned                                        flags)
{
    return fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
}

#endif
