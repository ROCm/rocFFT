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
// only buffer interface need rocfft variable

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

template <typename T, typename fftw_T>
class fftw_wrapper
{
};

template <>
class fftw_wrapper<float, fftwf_complex>
{
public:
    fftwf_plan plan;

    void make_plan(int                       x,
                   int                       y,
                   int                       z,
                   int                       num_dimensions,
                   int                       batch_size,
                   fftwf_complex*            input_ptr,
                   fftwf_complex*            output_ptr,
                   int                       num_points_in_single_batch,
                   const std::vector<size_t> input_strides,
                   const std::vector<size_t> output_strides,
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
                                       input_strides[0],
                                       num_points_in_single_batch * input_strides[0],
                                       output_ptr,
                                       NULL,
                                       output_strides[0],
                                       num_points_in_single_batch * output_strides[0],
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
                 fftwf_complex*            input_ptr,
                 fftwf_complex*            output_ptr,
                 int                       num_points_in_single_batch,
                 const std::vector<size_t> input_strides,
                 const std::vector<size_t> output_strides,
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
                  input_strides,
                  output_strides,
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
class fftw_wrapper<double, fftw_complex>
{
public:
    fftw_plan plan;

    void make_plan(int                       x,
                   int                       y,
                   int                       z,
                   int                       num_dimensions,
                   int                       batch_size,
                   fftw_complex*             input_ptr,
                   fftw_complex*             output_ptr,
                   int                       num_points_in_single_batch,
                   const std::vector<size_t> input_strides,
                   const std::vector<size_t> output_strides,
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
                                      input_strides[0],
                                      num_points_in_single_batch * input_strides[0],
                                      output_ptr,
                                      NULL,
                                      output_strides[0],
                                      num_points_in_single_batch * output_strides[0],
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
                 fftw_complex*             input_ptr,
                 fftw_complex*             output_ptr,
                 int                       num_points_in_single_batch,
                 const std::vector<size_t> input_strides,
                 const std::vector<size_t> output_strides,
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
                  input_strides,
                  output_strides,
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

template <typename T, typename fftw_T>
class fftw
{
private:
    static const size_t tightly_packed_distance = 0;

    fftw_direction      _direction;
    fftw_transform_type _type;
    rocfft_array_type   _input_layout, _output_layout;

    std::vector<size_t> _lengths;
    size_t              _batch_size;

    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;

    buffer<T>               input;
    buffer<T>               output;
    fftw_wrapper<T, fftw_T> fftw_guts;

    T _forward_scale, _backward_scale;

public:
    fftw(const std::vector<size_t>     lengths_in,
         const size_t                  batch_size_in,
         std::vector<size_t>           input_strides_in,
         std::vector<size_t>           output_strides_in,
         const rocfft_result_placement placement_in,
         fftw_transform_type           type_in)
        : _lengths(lengths_in)
        , _batch_size(batch_size_in)
        , input_strides(input_strides_in)
        , output_strides(output_strides_in)
        , _direction(fftw_direction::forward)
        , _type(type_in)
        , _input_layout(initialized_input_layout()) // chose interleaved layout artificially
        , _output_layout(initialized_output_layout())
        , input(lengths_in.size(),
                lengths_in.data(),
                input_strides_in.data(),
                batch_size_in,
                tightly_packed_distance,
                _input_layout,
                rocfft_placement_notinplace) // FFTW always use outof place
        // transformation
        , output(lengths_in.size(),
                 lengths_in.data(),
                 output_strides_in.data(),
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
                    reinterpret_cast<fftw_T*>(input_ptr()),
                    reinterpret_cast<fftw_T*>(output_ptr()),
                    (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                    input_strides,
                    output_strides,
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

    T* input_ptr()
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

    T* output_ptr()
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
                                reinterpret_cast<fftw_T*>(input.interleaved_ptr()),
                                reinterpret_cast<fftw_T*>(output.interleaved_ptr()),
                                (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                                input_strides,
                                output_strides,
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
                                reinterpret_cast<fftw_T*>(input.interleaved_ptr()),
                                reinterpret_cast<fftw_T*>(output.interleaved_ptr()),
                                (int)(_lengths[dimx] * _lengths[dimy] * _lengths[dimz]),
                                input_strides,
                                output_strides,
                                _direction,
                                _type);
        }
    }

    size_t size_of_data_in_bytes()
    {
        return input.size_in_bytes();
    }

    void forward_scale(T in)
    {
        _forward_scale = in;
    }

    void backward_scale(T in)
    {
        _backward_scale = in;
    }

    T forward_scale()
    {
        return _forward_scale;
    }

    T backward_scale()
    {
        return _backward_scale;
    }

    void set_data_to_value(T value)
    {
        input.set_all_to_value(value);
    }

    void set_data_to_value(T real_value, T imag_value)
    {
        input.set_all_to_value(real_value, imag_value);
    }

    void set_data_to_sawtooth(T max)
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

    void set_data_to_buffer(buffer<T> other_buffer)
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
                output.scale_data(static_cast<T>(forward_scale()));
            }
            else if(_direction == backward)
            {
                output.scale_data(static_cast<T>(backward_scale()));
            }
        }
        else if(_type == r2c)
        {
            output.scale_data(static_cast<T>(forward_scale()));
        }
        else if(_type == c2r)
        {
            output.scale_data(static_cast<T>(backward_scale()));
        }
        else
            throw std::runtime_error("invalid transform type in fftw::transform()");
    }

    buffer<T>& result()
    {
        return output;
    }

    buffer<T>& input_buffer()
    {
        return input;
    }
};

#endif
