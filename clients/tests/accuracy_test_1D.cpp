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

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

static std::vector<size_t> pow2_range
    = {2,      4,      8,       16,      32,      128,     256,      512,
       1024,   2048,   4096,    8192,    16384,   32768,   65536,    131072,
       262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432};
static std::vector<size_t> pow3_range
    = {3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323};
static std::vector<size_t> pow5_range
    = {5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125};
static std::vector<size_t> mix_range
    = {6,   10,   12,   15,   20,   30,   120,  150,  225,  240,  300,   486,   600,
       900, 1250, 1500, 1875, 2160, 2187, 2250, 2500, 3000, 4000, 12000, 24000, 72000};
static std::vector<size_t> prime_range
    = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

static size_t batch_range[] = {1};

static size_t stride_range[] = {1};

static rocfft_result_placement placeness_range[]
    = {rocfft_placement_notinplace, rocfft_placement_inplace};

static rocfft_transform_type transform_range[]
    = {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse};

static std::vector<size_t> generate_random(size_t number_run)
{
    std::vector<size_t> output;
    size_t              RAND_MAX_NUMBER = 6;
    for(size_t r = 0; r < number_run; r++)
    {
        // generate a integer number between [0, RAND_MAX-1]
        size_t i = (size_t)(rand() % RAND_MAX_NUMBER);
        size_t j = (size_t)(rand() % RAND_MAX_NUMBER);
        size_t k = (size_t)(rand() % RAND_MAX_NUMBER);
        output.push_back(pow(2, i) * pow(3, j) * pow(5, k));
    }
    return output;
}

class accuracy_test_complex
    : public ::testing::TestWithParam<
          std::tuple<size_t, size_t, rocfft_result_placement, size_t, rocfft_transform_type>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

class accuracy_test_real
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, rocfft_result_placement, size_t>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

template <class T>
void normal_1D_complex_interleaved_to_complex_interleaved(size_t                  N,
                                                          size_t                  batch,
                                                          rocfft_result_placement placeness,
                                                          rocfft_transform_type   transform_type,
                                                          size_t                  stride)
{
    std::vector<size_t> lengths;
    lengths.push_back(N);

    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            input_distance  = 0;
    size_t            output_distance = 0;
    rocfft_array_type in_array_type   = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type  = rocfft_array_type_complex_interleaved;

    data_pattern pattern = sawtooth;
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

// Complex to Complex

TEST_P(accuracy_test_complex, normal_1D_complex_interleaved_to_complex_interleaved_single_precision)
{
    size_t                  N              = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    rocfft_transform_type   transform_type = std::get<4>(GetParam());

    try
    {
        normal_1D_complex_interleaved_to_complex_interleaved<float>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_complex, normal_1D_complex_interleaved_to_complex_interleaved_double_precision)
{
    size_t                  N              = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    rocfft_transform_type   transform_type = std::get<4>(GetParam());

    try
    {
        normal_1D_complex_interleaved_to_complex_interleaved<double>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Real to complex

template <class T>
void normal_1D_real_to_complex_interleaved(size_t                  N,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride)
{
    std::vector<size_t> lengths;
    lengths.push_back(N);
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            input_distance  = 0; // 0 means the data are densely packed
    size_t            output_distance = 0; // 0 means the data are densely packed
    rocfft_array_type in_array_type   = rocfft_array_type_real;
    rocfft_array_type out_array_type  = rocfft_array_type_hermitian_interleaved;

    data_pattern pattern = sawtooth;
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

TEST_P(accuracy_test_real, normal_1D_real_to_complex_interleaved_single_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward

    try
    {
        normal_1D_real_to_complex_interleaved<float>(N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real, normal_1D_real_to_complex_interleaved_double_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward

    try
    {
        normal_1D_real_to_complex_interleaved<double>(N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex to Real

template <class T>
void normal_1D_complex_interleaved_to_real(size_t                  N,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride)
{
    std::vector<size_t> lengths;
    lengths.push_back(N);
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            input_distance  = 0; // 0 means the data are densely packed
    size_t            output_distance = 0; // 0 means the data are densely packed
    rocfft_array_type in_array_type   = rocfft_array_type_hermitian_interleaved;
    rocfft_array_type out_array_type  = rocfft_array_type_real;

    data_pattern pattern = sawtooth;
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

TEST_P(accuracy_test_real, normal_1D_complex_interleaved_to_real_single_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_1D_complex_interleaved_to_real<float>(N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real, normal_1D_complex_interleaved_to_real_double_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_1D_complex_interleaved_to_real<double>(N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// COMPLEX TO COMPLEX
INSTANTIATE_TEST_CASE_P(rocfft_pow2_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_mix_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(mix_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_random_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(generate_random(20)),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_1D,
                        accuracy_test_complex,
                        ::testing::Combine(ValuesIn(prime_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(transform_range)));

// Real/complex
INSTANTIATE_TEST_CASE_P(rocfft_pow2_1D,
                        accuracy_test_real,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_1D,
                        accuracy_test_real,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_1D,
                        accuracy_test_real,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_mix_1D,
                        accuracy_test_real,
                        ::testing::Combine(ValuesIn(mix_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_1D,
                        accuracy_test_real,
                        ::testing::Combine(ValuesIn(prime_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range)));

// TESTS disabled by default since they take a long time to execute
// TO enable this tests
// 1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
// 2. pass --gtest_also_run_disabled_tests to TEST.exe

#define CLFFT_TEST_HUGE
#ifdef CLFFT_TEST_HUGE

class accuracy_test_complex_pow2_single : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

class accuracy_test_complex_pow2_double : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

#define HUGE_TEST_MAKE(test_name, len, bat)                                              \
    template <class T>                                                                   \
    void test_name()                                                                     \
    {                                                                                    \
        std::vector<size_t> lengths;                                                     \
        lengths.push_back(len);                                                          \
        size_t batch = bat;                                                              \
                                                                                         \
        std::vector<size_t>     input_strides;                                           \
        std::vector<size_t>     output_strides;                                          \
        size_t                  input_distance  = 0;                                     \
        size_t                  output_distance = 0;                                     \
        rocfft_array_type       in_array_type   = rocfft_array_type_complex_planar;      \
        rocfft_array_type       out_array_type  = rocfft_array_type_complex_planar;      \
        rocfft_result_placement placeness       = rocfft_placement_inplace;              \
        rocfft_transform_type   transform_type  = rocfft_transform_type_complex_forward; \
                                                                                         \
        data_pattern pattern = sawtooth;                                                 \
        complex_to_complex<T>(pattern,                                                   \
                              transform_type,                                            \
                              lengths,                                                   \
                              batch,                                                     \
                              input_strides,                                             \
                              output_strides,                                            \
                              input_distance,                                            \
                              output_distance,                                           \
                              in_array_type,                                             \
                              out_array_type,                                            \
                              placeness);                                                \
    }

#define SP_HUGE_TEST(test_name, len, bat)                \
                                                         \
    HUGE_TEST_MAKE(test_name, len, bat)                  \
                                                         \
    TEST_F(accuracy_test_complex_pow2_single, test_name) \
    {                                                    \
        try                                              \
        {                                                \
            test_name<float>();                          \
        }                                                \
        catch(const std::exception& err)                 \
        {                                                \
            handle_exception(err);                       \
        }                                                \
    }

#define DP_HUGE_TEST(test_name, len, bat)                \
                                                         \
    HUGE_TEST_MAKE(test_name, len, bat)                  \
                                                         \
    TEST_F(accuracy_test_complex_pow2_double, test_name) \
    {                                                    \
        try                                              \
        {                                                \
            test_name<double>();                         \
        }                                                \
        catch(const std::exception& err)                 \
        {                                                \
            handle_exception(err);                       \
        }                                                \
    }

SP_HUGE_TEST(DISABLED_huge_sp_test_1, 1048576, 11)
SP_HUGE_TEST(DISABLED_huge_sp_test_2, 1048576 * 2, 7)
SP_HUGE_TEST(DISABLED_huge_sp_test_3, 1048576 * 4, 3)
SP_HUGE_TEST(DISABLED_huge_sp_test_4, 1048576 * 8, 5)
SP_HUGE_TEST(DISABLED_huge_sp_test_5, 1048576 * 16, 3)
SP_HUGE_TEST(DISABLED_huge_sp_test_6, 1048576 * 32, 2)
SP_HUGE_TEST(DISABLED_huge_sp_test_7, 1048576 * 64, 1)

DP_HUGE_TEST(DISABLED_huge_dp_test_1, 524288, 11)
DP_HUGE_TEST(DISABLED_huge_dp_test_2, 524288 * 2, 7)
DP_HUGE_TEST(DISABLED_huge_dp_test_3, 524288 * 4, 3)
DP_HUGE_TEST(DISABLED_huge_dp_test_4, 524288 * 8, 5)
DP_HUGE_TEST(DISABLED_huge_dp_test_5, 524288 * 16, 3)
DP_HUGE_TEST(DISABLED_huge_dp_test_6, 524288 * 32, 2)
DP_HUGE_TEST(DISABLED_huge_dp_test_7, 524288 * 64, 1)

SP_HUGE_TEST(DISABLED_large_sp_test_1, 8192, 11)
SP_HUGE_TEST(DISABLED_large_sp_test_2, 8192 * 2, 7)
SP_HUGE_TEST(DISABLED_large_sp_test_3, 8192 * 4, 3)
SP_HUGE_TEST(DISABLED_large_sp_test_4, 8192 * 8, 5)
SP_HUGE_TEST(DISABLED_large_sp_test_5, 8192 * 16, 3)
SP_HUGE_TEST(DISABLED_large_sp_test_6, 8192 * 32, 21)
SP_HUGE_TEST(DISABLED_large_sp_test_7, 8192 * 64, 17)

DP_HUGE_TEST(DISABLED_large_dp_test_1, 4096, 11)
DP_HUGE_TEST(DISABLED_large_dp_test_2, 4096 * 2, 7)
DP_HUGE_TEST(DISABLED_large_dp_test_3, 4096 * 4, 3)
DP_HUGE_TEST(DISABLED_large_dp_test_4, 4096 * 8, 5)
DP_HUGE_TEST(DISABLED_large_dp_test_5, 4096 * 16, 3)
DP_HUGE_TEST(DISABLED_large_dp_test_6, 4096 * 32, 21)
DP_HUGE_TEST(DISABLED_large_dp_test_7, 4096 * 64, 17)

#endif
