
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
class accuracy_test_complex_2D_pow2_single : public ::testing::Test
{
protected:
    accuracy_test_complex_2D_pow2_single() {}
    virtual ~accuracy_test_complex_2D_pow2_single() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_complex_2D_pow2_double : public ::testing::Test
{
protected:
    accuracy_test_complex_2D_pow2_double() {}
    virtual ~accuracy_test_complex_2D_pow2_double() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};
// 65536=pow(2,16)                                 //8388608 = pow(2,23)
#define POW2_RANGE                                        \
    {2, 4}, {8, 16}, {32, 128}, {256, 512}, {1024, 2048}, \
    {                                                     \
        4096, 8192                                        \
    } /* malloc fail on 4GB Fiji Nano on the following size \                    \
, {16384, 32768}, {65536, 131072}, {262144, 524288} */

#define POW3_RANGE                \
    {3, 9}, {27, 81}, {243, 729}, \
    {                             \
        2187, 6561                \
    } /* malloc fail on 4GB Fiji Nano on the following size , {19683, 59049},    \
       {177147, 531441} */
#define POW5_RANGE \
    {5, 25}, {125, 625}, {3125, 15625}, /* malloc fail on 4GB Fiji Nano on the   \
                                         following size , {78125, 390625},     \
                                         {1953125, 9765625} */

static std::vector<std::vector<size_t>> pow2_range = {POW2_RANGE};
static std::vector<std::vector<size_t>> pow3_range = {POW3_RANGE};
static std::vector<std::vector<size_t>> pow5_range = {POW5_RANGE};

static size_t batch_range[] = {1};

static size_t stride_range[] = {1}; // 1: assume packed data

static rocfft_result_placement placeness_range[] = {
    /*rocfft_placement_notinplace,*/ rocfft_placement_inplace};

static rocfft_transform_type transform_range[] = {
    /*rocfft_transform_type_complex_forward, */ rocfft_transform_type_complex_inverse};

static data_pattern pattern_range[] = {sawtooth};

static std::vector<std::vector<size_t>> generate_random(size_t number_run)
{
    std::vector<std::vector<size_t>> output;

    size_t i, j, k, l, m, n;

    size_t RAND_MAX_NUMBER = 6;

    for(size_t r = 0; r < number_run; r++)
    {
        std::vector<size_t> tmp;

        i = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        j = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        k = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]

        size_t value = pow(2, i) * pow(3, j) * pow(5, k);
        tmp.push_back(value);

        l = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        m = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]
        n = (size_t)(rand() % RAND_MAX_NUMBER); // generate a integer number between [0, RAND_MAX-1]

        value = pow(2, l) * pow(3, m) * pow(5, n);

        tmp.push_back(value);

        output.push_back(tmp);
    }

    return output;
}

class accuracy_test_complex_2D : public ::TestWithParam<std::tuple<std::vector<size_t>,
                                                                   size_t,
                                                                   rocfft_result_placement,
                                                                   rocfft_transform_type,
                                                                   size_t,
                                                                   data_pattern>>
{
protected:
    accuracy_test_complex_2D() {}
    virtual ~accuracy_test_complex_2D() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class accuracy_test_real_2D
    : public ::TestWithParam<std::tuple<std::vector<size_t>, size_t, data_pattern>>
{
protected:
    accuracy_test_real_2D() {}
    virtual ~accuracy_test_real_2D() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

template <class T, class fftw_T>
void normal_2D_complex_interleaved_to_complex_interleaved(std::vector<size_t>     lengths,
                                                          size_t                  batch,
                                                          rocfft_result_placement placeness,
                                                          rocfft_transform_type   transform_type,
                                                          size_t                  stride,
                                                          data_pattern            pattern)
{
    size_t total_size = 1;
    for(int i = 0; i < lengths.size(); i++)
    {
        total_size *= lengths[i];
    }
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
        normal_2D_complex_interleaved_to_complex_interleaved<float, fftwf_complex>(
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
        normal_2D_complex_interleaved_to_complex_interleaved<double, fftw_complex>(
            lengths, batch, placeness, transform_type, stride, pattern);
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
void normal_2D_real_interleaved_to_hermitian_interleaved(std::vector<size_t>     lengths,
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

TEST_P(accuracy_test_real_2D, normal_2D_real_interleaved_to_hermitian_interleaved_single_precision)
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
        normal_2D_real_interleaved_to_hermitian_interleaved<float, fftwf_complex>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_real_interleaved_to_hermitian_interleaved_double_precision)
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
        normal_2D_real_interleaved_to_hermitian_interleaved<double, fftw_complex>(
            lengths, batch, placeness, transform_type, stride, pattern);
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
void normal_2D_hermitian_interleaved_to_real_interleaved(std::vector<size_t>     lengths,
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

TEST_P(accuracy_test_real_2D, normal_2D_hermitian_interleaved_to_real_interleaved_single_precision)
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
        normal_2D_hermitian_interleaved_to_real_interleaved<float, fftwf_complex>(
            lengths, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_hermitian_interleaved_to_real_interleaved_double_precision)
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
        normal_2D_hermitian_interleaved_to_real_interleaved<double, fftw_complex>(
            lengths, batch, placeness, transform_type, stride, pattern);
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
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_complex_2D,
                        Combine(ValuesIn(pow2_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range),
                                ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_complex_2D,
                        Combine(ValuesIn(pow3_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range),
                                ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_complex_2D,
                        Combine(ValuesIn(pow5_range),
                                ValuesIn(batch_range),
                                ValuesIn(placeness_range),
                                ValuesIn(transform_range),
                                ValuesIn(stride_range),
                                ValuesIn(pattern_range)));

// *****************************************************
// REAL  HERMITIAN
// *****************************************************
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_real_2D,
                        Combine(ValuesIn(pow2_range),
                                ValuesIn(batch_range),
                                ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_real_2D,
                        Combine(ValuesIn(pow3_range),
                                ValuesIn(batch_range),
                                ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_real_2D,
                        Combine(ValuesIn(pow5_range),
                                ValuesIn(batch_range),
                                ValuesIn(pattern_range)));
