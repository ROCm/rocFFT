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
#include <unistd.h>
#include <vector>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"
#include "test_constants.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_complex_pow2_single : public ::testing::Test
{
protected:
    accuracy_test_complex_pow2_single() {}
    virtual ~accuracy_test_complex_pow2_single() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_complex_pow2_double : public ::testing::Test
{
protected:
    accuracy_test_complex_pow2_double() {}
    virtual ~accuracy_test_complex_pow2_double() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};
// 65536=pow(2,16)                                 //8388608 = pow(2,23)
#define POW2_RANGE                                                                               \
    2, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, \
        524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432
#define POW3_RANGE 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323
#define POW5_RANGE 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125
#define MIX_RANGE                                                                                \
    6, 10, 12, 15, 20, 30, 120, 150, 225, 240, 300, 486, 600, 900, 1250, 1500, 1875, 2160, 2187, \
        2250, 2500, 3000, 4000, 12000, 24000, 72000
#define PRIME_RANGE \
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97

#define LARGE_RANGE                                                                               \
    4096, 4050, 4000, 3888, 3840, 3750, 3645, 3600, 3456, 3375, 3240, 3200, 3125, 3072, 3000,     \
        2916, 2880, 2700, 2592, 2560, 2500, 2430, 2400, 2304, 2250, 2187, 2160, \ 2048, 2025,     \
        2000, 1944, 1920, 1875, 1800, 1728, 1620, 1600, 1536, 1500, 1458, 1440, 1350, 1296, 1280, \
        1250, 1215, 1200, 1152, 1125, 1080, 1024, 1000, 972, 960, \ 900, 864, 810, 800, 768, 750, \
        729, 720, 675, 648, 640, 625, 600, 576, 540, 512, 500, 486, 480, 450, 432, 405, 400, 384, \
        375, 360, 324, 320, 300, 288, 270, 256,  \ 250, 243, 240, 225, 216, 200, 192, 180, 162,   \
        160, 150, 144, 135, 128, 125, 120, 108, 100, 96, 90, 81, 80, 75, 72, 64, 60, 54, 50, 48,  \
        45, 40, 36, 32, 30, 27, 25,\ 24, 20, 18, 16, 15, 12, 10, 9, 8, 6, 5, 4, 3, 2, 1

static std::vector<size_t> pow2_range  = {POW2_RANGE};
static std::vector<size_t> pow3_range  = {POW3_RANGE};
static std::vector<size_t> pow5_range  = {POW5_RANGE};
static std::vector<size_t> mix_range   = {MIX_RANGE};
static std::vector<size_t> prime_range = {PRIME_RANGE};

static size_t batch_range[] = {1};

static size_t stride_range[] = {1};

static rocfft_result_placement placeness_range[]
    = {rocfft_placement_notinplace, rocfft_placement_inplace};

static rocfft_transform_type transform_range[]
    = {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse};

static std::vector<size_t> generate_random(size_t number_run)
{
    std::vector<size_t> output;

    size_t i, j, k;

    size_t RAND_MAX_NUMBER = 6;

    for(size_t r = 0; r < number_run; r++)
    {

        i = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        j = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        k = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]

        size_t value = pow(2, i) * pow(3, j) * pow(5, k);
        output.push_back(value);
    }

    return output;
}

class accuracy_test_complex
    : public ::TestWithParam<
          std::tuple<size_t, size_t, rocfft_result_placement, rocfft_transform_type, size_t>>
{
protected:
    accuracy_test_complex() {}
    virtual ~accuracy_test_complex() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class accuracy_test_real
    : public ::TestWithParam<std::tuple<size_t, size_t, rocfft_result_placement>>
{
protected:
    accuracy_test_real() {}
    virtual ~accuracy_test_real() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

template <class T, class fftw_T>
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
    complex_to_complex<T, fftw_T>(pattern,
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
    usleep(1e4);
}

// *****************************************************
//             Complex to Complex
// *****************************************************

TEST_P(accuracy_test_complex, normal_1D_complex_interleaved_to_complex_interleaved_single_precision)
{
    size_t                  N              = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    rocfft_transform_type   transform_type = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());

    try
    {
        normal_1D_complex_interleaved_to_complex_interleaved<float, fftwf_complex>(
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
    rocfft_transform_type   transform_type = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());

    try
    {
        normal_1D_complex_interleaved_to_complex_interleaved<double, fftw_complex>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// *****************************************************
//             Real to Hermitian
// *****************************************************

template <class T, class fftw_T>
void normal_1D_real_interleaved_to_hermitian_interleaved(size_t                  N,
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
    real_to_hermitian<T, fftw_T>(pattern,
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

    usleep(1e4);
}

TEST_P(accuracy_test_real, normal_1D_real_interleaved_to_hermitian_interleaved_single_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward
    size_t stride = 1; // FIXME: enable strides

    try
    {
        normal_1D_real_interleaved_to_hermitian_interleaved<float, fftwf_complex>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real, normal_1D_real_interleaved_to_hermitian_interleaved_double_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_forward; // must be real forward
    size_t stride = 1; // FIXME: enable strides

    try
    {
        normal_1D_real_interleaved_to_hermitian_interleaved<double, fftw_complex>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// *****************************************************
//             Hermitian to Real
// *****************************************************

template <class T, class fftw_T>
void normal_1D_hermitian_interleaved_to_real_interleaved(size_t                  N,
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
    hermitian_to_real<T, fftw_T>(pattern,
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

    usleep(1e4);
}

TEST_P(accuracy_test_real, normal_1D_hermitian_interleaved_to_real_interleaved_single_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse
    size_t stride = 1;

    try
    {
        normal_1D_hermitian_interleaved_to_real_interleaved<float, fftwf_complex>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real, normal_1D_hermitian_interleaved_to_real_interleaved_double_precision)
{
    size_t                  N         = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse
    size_t stride = 1;

    try
    {
        normal_1D_hermitian_interleaved_to_real_interleaved<double, fftw_complex>(
            N, batch, placeness, transform_type, stride);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Values is for a single item; ValuesIn is for an array
// ValuesIn take each element (a vector) and combine them and feed them to
// test_p
// *****************************************************
// COMPLEX TO COMPLEX
// *****************************************************
INSTANTIATE_TEST_CASE_P(rocfft_pow2_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(pow2_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(pow3_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(pow5_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_mix_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(mix_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_random_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(generate_random(20)),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_1D,
                        accuracy_test_complex,
                        Combine(ValuesIn(prime_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range)));

// *****************************************************
// REAL HERMITIAN
// *****************************************************
INSTANTIATE_TEST_CASE_P(rocfft_pow2_1D,
                        accuracy_test_real,
                        Combine(ValuesIn(pow2_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_1D,
                        accuracy_test_real,
                        Combine(ValuesIn(pow3_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_1D,
                        accuracy_test_real,
                        Combine(ValuesIn(pow5_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow_mix_1D,
                        accuracy_test_real,
                        Combine(ValuesIn(mix_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_1D,
                        accuracy_test_real,
                        Combine(ValuesIn(prime_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range)));

// *****************************************************
// *****************************************************
// TESTS disabled by default since they take a long time to execute
// TO enable this tests
// 1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
// 2. pass --gtest_also_run_disabled_tests to TEST.exe

#define CLFFT_TEST_HUGE
#ifdef CLFFT_TEST_HUGE

#define HUGE_TEST_MAKE(test_name, len, bat)                                              \
    template <class T, class fftw_T>                                                     \
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
        complex_to_complex<T, fftw_T>(pattern,                                           \
                                      transform_type,                                    \
                                      lengths,                                           \
                                      batch,                                             \
                                      input_strides,                                     \
                                      output_strides,                                    \
                                      input_distance,                                    \
                                      output_distance,                                   \
                                      in_array_type,                                     \
                                      out_array_type,                                    \
                                      placeness);                                        \
    }

#define SP_HUGE_TEST(test_name, len, bat)                \
                                                         \
    HUGE_TEST_MAKE(test_name, len, bat)                  \
                                                         \
    TEST_F(accuracy_test_complex_pow2_single, test_name) \
    {                                                    \
        try                                              \
        {                                                \
            test_name<float, fftwf_complex>();           \
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
            test_name<double, fftw_complex>();           \
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
