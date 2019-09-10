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

#include <algorithm>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

// Set parameters

static std::vector<std::vector<size_t>> pow2_range
    = {{2, 4}, {8, 16}, {32, 128}, {256, 512}, {1024, 2048}, {4096, 8192}};
// The even-length c2r fails 4096x8192.
// TODO: make test precision vary with problem size, then re-enable.
static std::vector<std::vector<size_t>> pow2_range_c2r
    = {{2, 4}, {8, 16}, {32, 128}, {256, 512}, {1024, 2048}};
static std::vector<std::vector<size_t>> pow3_range  = {{3, 9}, {27, 81}, {243, 729}, {2187, 6561}};
static std::vector<std::vector<size_t>> pow5_range  = {{5, 25}, {125, 625}, {3125, 15625}};
static std::vector<std::vector<size_t>> prime_range = {
    {7, 25}, {11, 625}, {13, 15625}, {1, 11}, {11, 1}, {8191, 243}, {7, 11}, {7, 32}, {1009, 1009}};

static size_t batch_range[] = {1};

static size_t stride_range[] = {1}; // 1: assume packed data

static rocfft_result_placement placeness_range[]
    = {rocfft_placement_notinplace, rocfft_placement_inplace};

static rocfft_transform_type transform_range[]
    = {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse};

static data_pattern pattern_range[] = {sawtooth};

// Test suite classes:

class accuracy_test_complex_2D : public ::testing::TestWithParam<std::tuple<std::vector<size_t>,
                                                                            size_t,
                                                                            rocfft_result_placement,
                                                                            rocfft_transform_type,
                                                                            size_t,
                                                                            data_pattern>>
{
};
class accuracy_test_real_2D
    : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, size_t, data_pattern>>
{
};

//  Complex to complex

// Templated test function for complex to complex:
template <class T>
void normal_2D_complex_interleaved_to_complex_interleaved(std::vector<size_t>     lengths,
                                                          size_t                  batch,
                                                          rocfft_result_placement placeness,
                                                          rocfft_transform_type   transform_type,
                                                          size_t                  stride,
                                                          data_pattern            pattern)
{
    size_t total_size
        = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>());
    if(total_size * sizeof(T) * 2 >= 2e8)
    {
        // printf("No test is really launched; MB byte size = %f is too big; will
        // return \n", total_size * sizeof(T) * 2/1e6);
        return; // memory size over 200MB is too big
    }
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);
    for(int i = 1; i < lengths.size(); i++)
    {
        input_strides.push_back(input_strides[i - 1] * lengths[i - 1]);
        output_strides.push_back(output_strides[i - 1] * lengths[i - 1]);
    }

    size_t            input_distance  = 0;
    size_t            output_distance = 0;
    rocfft_array_type in_array_type   = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type  = rocfft_array_type_complex_interleaved;

    complex_to_complex<T>(pattern,
                          transform_type,
                          lengths,
                          batch,
                          input_strides,
                          output_strides,
                          input_distance,
                          output_distance,
                          in_array_type,
                          out_array_type,
                          placeness);
}

// Implemetation of complex-to-complex tests for float and double:

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_single_precision)
{
    std::vector<size_t>     lengths        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    rocfft_transform_type   transform_type = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<float>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_double_precision)
{

    std::vector<size_t>     lengths        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    rocfft_transform_type   transform_type = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<double>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Populate test cases from parameter combinations:

// Real to complex

// Templated test function for real to complex:
template <class T>
void normal_2D_real_to_complex_interleaved(std::vector<size_t>     lengths,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{

    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            input_distance  = 0; // 0 means the data are densely packed
    size_t            output_distance = 0; // 0 means the data are densely packed
    rocfft_array_type in_array_type   = rocfft_array_type_real;
    rocfft_array_type out_array_type  = rocfft_array_type_hermitian_interleaved;

    real_to_complex<T>(pattern,
                       transform_type,
                       lengths,
                       batch,
                       input_strides,
                       output_strides,
                       input_distance,
                       output_distance,
                       in_array_type,
                       out_array_type,
                       rocfft_placement_notinplace); // must be non-inplace tranform
}

// Implemetation of real-to-complex tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_single_precision)
{
    std::vector<size_t>     lengths   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    data_pattern            pattern   = std::get<2>(GetParam());
    rocfft_result_placement placeness = rocfft_placement_notinplace; // must be non-inplace
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward
    size_t stride = 1;

    try
    {
        normal_2D_real_to_complex_interleaved<float>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_double_precision)
{
    std::vector<size_t>     lengths   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    data_pattern            pattern   = std::get<2>(GetParam());
    rocfft_result_placement placeness = rocfft_placement_notinplace; // must be non-inplace
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward
    size_t stride = 1;

    try
    {
        normal_2D_real_to_complex_interleaved<double>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex to teal

// Templated test function for complex to real:
template <class T>
void normal_2D_complex_interleaved_to_real(std::vector<size_t>     lengths,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            input_distance  = 0; // 0 means the data are densely packed
    size_t            output_distance = 0; // 0 means the data are densely packed
    rocfft_array_type in_array_type   = rocfft_array_type_hermitian_interleaved;
    rocfft_array_type out_array_type  = rocfft_array_type_real;

    complex_to_real<T>(pattern,
                       transform_type,
                       lengths,
                       batch,
                       input_strides,
                       output_strides,
                       input_distance,
                       output_distance,
                       in_array_type,
                       out_array_type,
                       rocfft_placement_notinplace); // must be non-inplace tranform
}

// Implemetation of real-to-complex tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_single_precision)
{
    std::vector<size_t>     lengths   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    data_pattern            pattern   = std::get<2>(GetParam());
    rocfft_result_placement placeness = rocfft_placement_notinplace; // must be non-inplace
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse
    size_t stride = 1;

    try
    {
        normal_2D_complex_interleaved_to_real<float>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_double_precision)
{
    std::vector<size_t>     lengths   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    data_pattern            pattern   = std::get<2>(GetParam());
    rocfft_result_placement placeness = rocfft_placement_notinplace; // must be non-inplace
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse
    size_t stride = 1;

    try
    {
        normal_2D_complex_interleaved_to_real<double>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(transform_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(transform_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(transform_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(prime_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(transform_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

// Complex to real and real-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow2_range_c2r),
                                           ValuesIn(batch_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(pattern_range)));
