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
#include <vector>

#include "../client_utils.h"

#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

// TODO: handle special case where length=2 for real/complex transforms.
static std::vector<size_t> pow2_range
    = {4,      8,       16,      32,      128,     256,      512,     1024,
       2048,   4096,    8192,    16384,   32768,   65536,    131072,  262144,
       524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432};
static std::vector<size_t> pow3_range
    = {3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323};
static std::vector<size_t> pow5_range
    = {5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125};
static std::vector<size_t> mix_range
    = {6,   10,   12,   15,   20,   30,   120,  150,  225,  240,  300,   486,   600,
       900, 1250, 1500, 1875, 2160, 2187, 2250, 2500, 3000, 4000, 12000, 24000, 72000};
static std::vector<size_t> prime_range
    = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

static std::vector<size_t> stride_range = {1};

static std::vector<size_t> stride_range_for_prime
    = {1, 2, 3, 64, 65}; //TODO: this will be merged back to stride_range

static std::vector<size_t> generate_random(size_t number_run)
{
    std::vector<size_t> output(number_run);
    const size_t        RAND_MAX_NUMBER = 6;
    for(size_t r = 0; r < number_run; r++)
    {
        // generate a integer number between [0, RAND_MAX - 1]
        size_t i, j, k;
        do
        {
            i = (size_t)(rand() % RAND_MAX_NUMBER);
            j = (size_t)(rand() % RAND_MAX_NUMBER);
            k = (size_t)(rand() % RAND_MAX_NUMBER);
        } while(i + j + k == 0);
        output[i] = pow(2, i) * pow(3, j) * pow(5, k);
    }
    return output;
}

static std::vector<std::vector<size_t>> vpow2_range = {pow2_range};
INSTANTIATE_TEST_CASE_P(
    pow2_1D_complex_forward,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow2_1D_complex_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_inverse)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow2_1D_real_direct,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_real_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow2_1D_real_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_real_inverse)),
                       ::testing::Values(place_range)));

static std::vector<std::vector<size_t>> vpow3_range = {pow3_range};
INSTANTIATE_TEST_CASE_P(
    pow3_1D_complex_forward,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow3_1D_complex_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_inverse)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow3_1D_real_direct,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_real_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow3_1D_real_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_real_inverse)),
                       ::testing::Values(place_range)));

static std::vector<std::vector<size_t>> vpow5_range = {pow5_range};
INSTANTIATE_TEST_CASE_P(
    pow5_1D_complex_forward,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow5_1D_complex_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_inverse)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow5_1D_real_direct,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_real_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    pow5_1D_real_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_real_inverse)),
                       ::testing::Values(place_range)));

static std::vector<std::vector<size_t>> vprime_range = {prime_range};
INSTANTIATE_TEST_CASE_P(
    prime_1D_complex_forward,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vprime_range)),
                       ::testing::Values(stride_range_for_prime),
                       ::testing::Values(stride_range_for_prime),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    prime_1D_complex_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vprime_range)),
                       ::testing::Values(stride_range_for_prime),
                       ::testing::Values(stride_range_for_prime),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_complex_inverse)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    prime_1D_real_direct,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vprime_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_forward),
                       ::testing::Values(iotypes(rocfft_transform_type_real_forward)),
                       ::testing::Values(place_range)));
INSTANTIATE_TEST_CASE_P(
    prime_1D_real_inverse,
    accuracy_test,
    ::testing::Combine(ValuesIn(generate_lengths(vprime_range)),
                       ::testing::Values(stride_range),
                       ::testing::Values(stride_range),
                       ::testing::Values(batch_range),
                       ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_real_inverse),
                       ::testing::Values(iotypes(rocfft_transform_type_real_inverse)),
                       ::testing::Values(place_range)));
