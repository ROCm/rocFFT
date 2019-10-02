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

#include "buffer.h"
#include <fftw3.h>
#include <vector>

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
    return 1e-8;
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

template <typename Tfloat>
class fftw_wrapper
{
};

template <>
class fftw_wrapper<float>
{
public:
    using fftw_complex_type = typename fftw_trait<float>::fftw_complex_type;

    fftwf_plan plan;

    void make_plan(int                       x,
                   int                       y,
                   int                       z,
                   int                       num_dimensions,
                   int                       batch_size,
                   fftw_complex_type*        input_ptr,
                   fftw_complex_type*        output_ptr,
                   int                       num_points_in_single_batch,
                   const std::vector<size_t> istride,
                   const std::vector<size_t> ostride,
                   fftw_direction            direction,
                   fftw_transform_type       type)
    {
        // We need to swap x,y,z dimensions because of a row-column
        // discrepancy between rocfft and fftw.
        int lengths[max_dimension] = {z, y, x};

        // Because we swapped dimensions up above, we need to start at
        // the end of the array and count backwards to get the correct
        // dimensions passed in to fftw.
        // e.g. if max_dimension is 3 and number_of_dimensions is 2:
        // lengths = {dimz, dimy, dimx}
        // lengths + 3 - 2 = lengths + 1
        // so we will skip dimz and pass in a pointer to {dimy, dimx}

        switch(type)
        {
        case c2c:
            plan = fftwf_plan_many_dft(num_dimensions,
                                       lengths + max_dimension - num_dimensions,
                                       batch_size,
                                       input_ptr,
                                       NULL,
                                       istride[0],
                                       num_points_in_single_batch * istride[0],
                                       output_ptr,
                                       NULL,
                                       ostride[0],
                                       num_points_in_single_batch * ostride[0],
                                       direction,
                                       FFTW_ESTIMATE);
            break;
        case r2c:
            plan = fftwf_plan_many_dft_r2c(num_dimensions,
                                           lengths + max_dimension - num_dimensions,
                                           batch_size,
                                           reinterpret_cast<float*>(input_ptr),
                                           NULL,
                                           1,
                                           num_points_in_single_batch, // TODO for strides
                                           output_ptr,
                                           NULL,
                                           1,
                                           (x / 2 + 1) * y * z,
                                           FFTW_ESTIMATE);
            break;
        case c2r:
            plan = fftwf_plan_many_dft_c2r(num_dimensions,
                                           lengths + max_dimension - num_dimensions,
                                           batch_size,
                                           input_ptr,
                                           NULL,
                                           1,
                                           (x / 2 + 1) * y * z,
                                           reinterpret_cast<float*>(output_ptr),
                                           NULL,
                                           1,
                                           num_points_in_single_batch, // TODO for strides
                                           FFTW_ESTIMATE);
            break;
        default:
            throw std::runtime_error("invalid transform type in <float>make_plan");
        }
    }

    fftw_wrapper(int                       x,
                 int                       y,
                 int                       z,
                 int                       num_dimensions,
                 int                       batch_size,
                 fftw_complex_type*        input_ptr,
                 fftw_complex_type*        output_ptr,
                 int                       num_points_in_single_batch,
                 const std::vector<size_t> istride,
                 const std::vector<size_t> ostride,
                 fftw_direction            direction,
                 fftw_transform_type       type)
    {
        make_plan(x,
                  y,
                  z,
                  num_dimensions,
                  batch_size,
                  input_ptr,
                  output_ptr,
                  num_points_in_single_batch,
                  istride,
                  ostride,
                  direction,
                  type);
    }

    void destroy_plan()
    {
        fftwf_destroy_plan(plan);
    }

    ~fftw_wrapper()
    {
        destroy_plan();
    }

    void execute()
    {
        fftwf_execute(plan);
    }
};

template <>
class fftw_wrapper<double>
{
public:
    using fftw_complex_type = typename fftw_trait<double>::fftw_complex_type;

    fftw_plan plan;

    void make_plan(int                       x,
                   int                       y,
                   int                       z,
                   int                       num_dimensions,
                   int                       batch_size,
                   fftw_complex_type*        input_ptr,
                   fftw_complex_type*        output_ptr,
                   int                       num_points_in_single_batch,
                   const std::vector<size_t> istride,
                   const std::vector<size_t> ostride,
                   fftw_direction            direction,
                   fftw_transform_type       type)
    {
        // we need to swap x,y,z dimensions because of a row-column discrepancy
        // between rocfft and fftw
        int lengths[max_dimension] = {z, y, x};

        // Because we swapped dimensions up above, we need to start at
        // the end of the array and count backwards to get the correct
        // dimensions passed in to fftw.
        // e.g. if max_dimension is 3 and number_of_dimensions is 2:
        // lengths = {dimz, dimy, dimx}
        // lengths + 3 - 2 = lengths + 1
        // so we will skip dimz and pass in a pointer to {dimy, dimx}

        switch(type)
        {
        case c2c:
            plan = fftw_plan_many_dft(num_dimensions,
                                      lengths + max_dimension - num_dimensions,
                                      batch_size,
                                      input_ptr,
                                      NULL,
                                      istride[0],
                                      num_points_in_single_batch * istride[0],
                                      output_ptr,
                                      NULL,
                                      ostride[0],
                                      num_points_in_single_batch * ostride[0],
                                      direction,
                                      FFTW_ESTIMATE);
            break;
        case r2c:
            plan = fftw_plan_many_dft_r2c(num_dimensions,
                                          lengths + max_dimension - num_dimensions,
                                          batch_size,
                                          reinterpret_cast<double*>(input_ptr),
                                          NULL,
                                          1,
                                          num_points_in_single_batch, // TODO for strides
                                          output_ptr,
                                          NULL,
                                          1,
                                          (x / 2 + 1) * y * z,
                                          FFTW_ESTIMATE);
            break;
        case c2r:
            plan = fftw_plan_many_dft_c2r(num_dimensions,
                                          lengths + max_dimension - num_dimensions,
                                          batch_size,
                                          input_ptr,
                                          NULL,
                                          1,
                                          (x / 2 + 1) * y * z,
                                          reinterpret_cast<double*>(output_ptr),
                                          NULL,
                                          1,
                                          num_points_in_single_batch, // TODO for strides
                                          FFTW_ESTIMATE);
            break;
        default:
            throw std::runtime_error("invalid transform type in <double>make_plan");
        }
    }

    fftw_wrapper(int                       x,
                 int                       y,
                 int                       z,
                 int                       num_dimensions,
                 int                       batch_size,
                 fftw_complex_type*        input_ptr,
                 fftw_complex_type*        output_ptr,
                 int                       num_points_in_single_batch,
                 const std::vector<size_t> istride,
                 const std::vector<size_t> ostride,
                 fftw_direction            direction,
                 fftw_transform_type       type)
    {
        make_plan(x,
                  y,
                  z,
                  num_dimensions,
                  batch_size,
                  input_ptr,
                  output_ptr,
                  num_points_in_single_batch,
                  istride,
                  ostride,
                  direction,
                  type);
    }

    void destroy_plan()
    {
        fftw_destroy_plan(plan);
    }

    ~fftw_wrapper()
    {
        destroy_plan();
    }

    void execute()
    {
        fftw_execute(plan);
    }
};

template <typename Tfloat>
class fftw
{
private:
    using fftw_complex_type                     = typename fftw_trait<Tfloat>::fftw_complex_type;
    static const size_t tightly_packed_distance = 0;

    fftw_direction      _direction;
    fftw_transform_type _type;
    rocfft_array_type   _input_layout, _output_layout;

    std::vector<size_t> _lengths;
    size_t              _batch_size;

    std::vector<size_t> istride;
    std::vector<size_t> ostride;

    buffer<Tfloat>       input;
    buffer<Tfloat>       output;
    fftw_wrapper<Tfloat> fftw_guts;

    Tfloat _forward_scale, _backward_scale;

public:
    fftw(const std::vector<size_t>     lengths_in,
         const size_t                  batch_size_in,
         std::vector<size_t>           istride_in,
         std::vector<size_t>           ostride_in,
         const rocfft_result_placement placement_in,
         fftw_transform_type           type_in)
        : _lengths(lengths_in)
        , _batch_size(batch_size_in)
        , istride(istride_in)
        , ostride(ostride_in)
        , _direction(fftw_direction::forward)
        , _type(type_in)
        , _input_layout(initialized_input_layout()) // chose interleaved layout artificially
        , _output_layout(initialized_output_layout())
        , input(lengths_in.size(),
                lengths_in.data(),
                istride_in.data(),
                batch_size_in,
                tightly_packed_distance,
                _input_layout,
                rocfft_placement_notinplace) // FFTW always use outof place
        // transformation
        , output(lengths_in.size(),
                 lengths_in.data(),
                 ostride_in.data(),
                 batch_size_in,
                 tightly_packed_distance,
                 _output_layout,
                 rocfft_placement_notinplace)
        , _forward_scale(1.0f)
        , _backward_scale(1.0f)
        , fftw_guts((int)_lengths[dimx],
                    (int)_lengths[dimy],
                    (int)_lengths[dimz],
                    (int)lengths_in.size(),
                    (int)batch_size_in,
                    reinterpret_cast<fftw_complex_type*>(input_ptr()),
                    reinterpret_cast<fftw_complex_type*>(output_ptr()),
                    (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                    istride,
                    ostride,
                    _direction,
                    _type)
    {
        clear_data_buffer();
    }

    ~fftw() {}

    rocfft_array_type initialized_input_layout()
    {
        switch(_type)
        {
        case c2c:
            return rocfft_array_type_complex_interleaved;
        case r2c:
            return rocfft_array_type_real;
        case c2r:
            return rocfft_array_type_hermitian_interleaved;
        default:
            throw std::runtime_error("invalid transform type in initialized_input_layout");
        }
    }

    rocfft_array_type initialized_output_layout()
    {
        switch(_type)
        {
        case c2c:
            return rocfft_array_type_complex_interleaved;
        case r2c:
            return rocfft_array_type_hermitian_interleaved;
        case c2r:
            return rocfft_array_type_real;
        default:
            throw std::runtime_error("invalid transform type in initialized_input_layout");
        }
    }

    std::vector<size_t> initialized_lengths(const size_t  number_of_dimensions,
                                            const size_t* lengths_in)
    {
        std::vector<size_t> lengths(3, 1); // start with 1, 1, 1
        for(size_t i = 0; i < number_of_dimensions; i++)
        {
            lengths[i] = lengths_in[i];
        }
        return lengths;
    }

    Tfloat* input_ptr()
    {
        switch(_input_layout)
        {
        case rocfft_array_type_real:
            return input.real_ptr();
        case rocfft_array_type_complex_interleaved:
            return input.interleaved_ptr();
        case rocfft_array_type_hermitian_interleaved:
            return input.interleaved_ptr();
        default:
            throw std::runtime_error("invalid layout in fftw::input_ptr");
        }
    }

    Tfloat* output_ptr()
    {
        if(_output_layout == rocfft_array_type_real)
            return output.real_ptr();
        else if(_output_layout == rocfft_array_type_complex_interleaved)
            return output.interleaved_ptr();
        else if(_output_layout == rocfft_array_type_hermitian_interleaved)
            return output.interleaved_ptr();
        else
            throw std::runtime_error("invalid layout in fftw::output_ptr");
    }

    // you must call either set_forward_transform() or
    // set_backward_transform() before setting the input buffer
    void set_forward_transform()
    {
        if(_type != c2c)
            throw std::runtime_error(
                "do not use set_forward_transform() except with c2c transforms");

        if(_direction != fftw_direction::forward)
        {
            _direction = fftw_direction::forward;
            fftw_guts.destroy_plan();
            fftw_guts.make_plan((int)_lengths[dimx],
                                (int)_lengths[dimy],
                                (int)_lengths[dimz],
                                (int)input.number_of_dimensions(),
                                (int)input.batch_size(),
                                reinterpret_cast<fftw_complex_type*>(input.interleaved_ptr()),
                                reinterpret_cast<fftw_complex_type*>(output.interleaved_ptr()),
                                (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                                istride,
                                ostride,
                                _direction,
                                _type);
        }
    }

    void set_backward_transform()
    {
        if(_type != c2c)
            throw std::runtime_error(
                "do not use set_backward_transform() except with c2c transforms");

        if(_direction != backward)
        {
            _direction = backward;
            fftw_guts.destroy_plan();
            fftw_guts.make_plan((int)_lengths[dimx],
                                (int)_lengths[dimy],
                                (int)_lengths[dimz],
                                (int)input.number_of_dimensions(),
                                (int)input.batch_size(),
                                reinterpret_cast<fftw_complex_type*>(input.interleaved_ptr()),
                                reinterpret_cast<fftw_complex_type*>(output.interleaved_ptr()),
                                (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                                istride,
                                ostride,
                                _direction,
                                _type);
        }
    }

    size_t size_of_data_in_bytes()
    {
        return input.size_in_bytes();
    }

    void forward_scale(Tfloat in)
    {
        _forward_scale = in;
    }

    void backward_scale(Tfloat in)
    {
        _backward_scale = in;
    }

    Tfloat forward_scale()
    {
        return _forward_scale;
    }

    Tfloat backward_scale()
    {
        return _backward_scale;
    }

    void set_data_to_value(Tfloat value)
    {
        input.set_all_to_value(value);
    }

    void set_data_to_value(Tfloat real_value, Tfloat imag_value)
    {
        input.set_all_to_value(real_value, imag_value);
    }

    void set_data_to_sawtooth(Tfloat max)
    {
        input.set_all_to_sawtooth(max);
    }

    void set_data_to_increase_linearly()
    {
        input.set_all_to_linear_increase();
    }

    void set_data_to_impulse()
    {
        input.set_all_to_impulse();
    }

    void set_data_to_random()
    {
        input.set_all_to_random();
    }

    void set_data_to_buffer(buffer<Tfloat> other_buffer)
    {
        input = other_buffer;
    }

    void clear_data_buffer()
    {
        if(_input_layout == rocfft_array_type_real)
        {
            set_data_to_value(0.0f);
        }
        else
        {
            set_data_to_value(0.0f, 0.0f);
        }
    }

    void transform()
    {
        fftw_guts.execute();

        if(_type == c2c)
        {
            if(_direction == fftw_direction::forward)
            {
                output.scale_data(static_cast<Tfloat>(forward_scale()));
            }
            else if(_direction == backward)
            {
                output.scale_data(static_cast<Tfloat>(backward_scale()));
            }
        }
        else if(_type == r2c)
        {
            output.scale_data(static_cast<Tfloat>(forward_scale()));
        }
        else if(_type == c2r)
        {
            output.scale_data(static_cast<Tfloat>(backward_scale()));
        }
        else
            throw std::runtime_error("invalid transform type in fftw::transform()");
    }

    buffer<Tfloat>& result()
    {
        return output;
    }

    buffer<Tfloat>& input_buffer()
    {
        return input;
    }
};

#endif
