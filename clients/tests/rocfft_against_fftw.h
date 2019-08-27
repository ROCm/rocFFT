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

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_transform.h"
#include "test_constants.h"

/*****************************************************
           Complex to Complex
*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template <class T, class fftw_T>
void complex_to_complex(data_pattern            pattern,
                        rocfft_transform_type   transform_type,
                        std::vector<size_t>     lengths,
                        size_t                  batch,
                        std::vector<size_t>     input_strides,
                        std::vector<size_t>     output_strides,
                        size_t                  input_distance,
                        size_t                  output_distance,
                        rocfft_array_type       in_array_type,
                        rocfft_array_type       out_array_type,
                        rocfft_result_placement placeness,
                        T                       scale = 1.0f)
{

    rocfft<T> test_fft(lengths,
                       batch,
                       input_strides,
                       output_strides,
                       input_distance,
                       output_distance,
                       in_array_type,
                       out_array_type,
                       placeness,
                       transform_type,
                       scale);

    fftw<T, fftw_T> reference(lengths, batch, input_strides, output_strides, placeness, c2c);

    switch(pattern)
    {
    case sawtooth:
        test_fft.set_data_to_sawtooth(1.0f);
        break;
    case value:
        test_fft.set_data_to_value(2.0f, 2.5f);
        break;
    case impulse:
        test_fft.set_data_to_impulse();
        break;
    case erratic:
        test_fft.set_data_to_random();
        break;
    default:
        throw std::runtime_error("invalid pattern type in complex_to_complex()");
    }

    reference.set_data_to_buffer(test_fft.input_buffer());

    // scale is already set in plan create called in constructor of
    // class rocfft
    switch(transform_type)
    {
    case rocfft_transform_type_complex_forward:
        reference.set_forward_transform();
        reference.forward_scale(scale);
        break;
    case rocfft_transform_type_complex_inverse:
        reference.set_backward_transform();
        reference.backward_scale(scale);
        break;
    default:
        throw std::runtime_error("invalid transform_type  in complex_to_complex()");
    }

    reference.transform();
    test_fft.transform();

    EXPECT_EQ(true, test_fft.result() == reference.result());
}

/*****************************************************
           Real to Hermitian
*****************************************************/

// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always real
template <class T, class fftw_T>
void real_to_hermitian(data_pattern            pattern,
                       rocfft_transform_type   transform_type,
                       std::vector<size_t>     lengths,
                       size_t                  batch,
                       std::vector<size_t>     input_strides,
                       std::vector<size_t>     output_strides,
                       size_t                  input_distance,
                       size_t                  output_distance,
                       rocfft_array_type       in_array_type,
                       rocfft_array_type       out_array_type,
                       rocfft_result_placement placeness,
                       T                       scale = 1.0f)
{
    rocfft<T> test_fft(lengths,
                       batch,
                       input_strides,
                       output_strides,
                       input_distance,
                       output_distance,
                       in_array_type,
                       out_array_type,
                       placeness,
                       transform_type,
                       scale);

    fftw<T, fftw_T> reference(lengths, batch, input_strides, output_strides, placeness, r2c);

    switch(pattern)
    {
    case sawtooth:
        test_fft.set_data_to_sawtooth(1.0f);
        break;
    case value:
        test_fft.set_data_to_value(2.0f);
        break;
    case impulse:
        test_fft.set_data_to_impulse();
        break;
    case erratic:
        test_fft.set_data_to_random();
        break;
    default:
        throw std::runtime_error("invalid pattern type in real_to_hermitian()");
    }

    reference.set_data_to_buffer(test_fft.input_buffer());

    // test_fft.forward_scale( scale );//TODO
    // reference.forward_scale( scale );//TODO

    test_fft.transform();
    reference.transform();

    EXPECT_EQ(true, test_fft.result() == reference.result());
}

/*****************************************************
           Hermitian to Real
*****************************************************/

// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always hermitian
template <class T, class fftw_T>
void hermitian_to_real(data_pattern            pattern,
                       rocfft_transform_type   transform_type,
                       std::vector<size_t>     lengths,
                       size_t                  batch,
                       std::vector<size_t>     input_strides,
                       std::vector<size_t>     output_strides,
                       size_t                  input_distance,
                       size_t                  output_distance,
                       rocfft_array_type       in_array_type,
                       rocfft_array_type       out_array_type,
                       rocfft_result_placement placeness,
                       T                       scale = 1.0f)
{
    // will perform a real to hermitian first
    fftw<T, fftw_T> data_maker(lengths, batch, output_strides, input_strides, placeness, r2c);

    switch(pattern)
    {
    case sawtooth:
        data_maker.set_data_to_sawtooth(1.0f);
        break;
    case value:
        data_maker.set_data_to_value(2.0f);
        break;
    case impulse:
        data_maker.set_data_to_impulse();
        break;
    case erratic:
        data_maker.set_data_to_random();
        break;
    default:
        throw std::runtime_error("invalid pattern type in hermitian_to_real()");
    }

    data_maker.transform();

    rocfft<T> test_fft(lengths,
                       batch,
                       input_strides,
                       output_strides,
                       input_distance,
                       output_distance,
                       in_array_type,
                       out_array_type,
                       placeness,
                       transform_type,
                       scale);
    test_fft.set_data_to_buffer(data_maker.result());

    fftw<T, fftw_T> reference(lengths, batch, input_strides, output_strides, placeness, c2r);
    reference.set_data_to_buffer(data_maker.result());

    // if we're starting with unequal data, we're destined for failure
    EXPECT_EQ(true, test_fft.input_buffer() == reference.input_buffer());

    // test_fft.backward_scale( scale );// rocFFT kernels do not take scale for
    // inverse
    // reference.backward_scale( scale );// FFTW kernels do not take scale for
    // inverse

    test_fft.transform();
    reference.transform();

    EXPECT_EQ(true, test_fft.result() == reference.result());
}
