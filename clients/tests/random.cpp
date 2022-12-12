// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <random>

#include "../rocfft_params.h"
#include "accuracy_test.h"
#include "rocfft_accuracy_test.h"

static const int n_random_tests = 10;

class random_params
    : public ::testing::TestWithParam<
          std::tuple<int, int, fft_precision, fft_result_placement, fft_transform_type>>
{
};

// TODO: Add batch and stride

TEST_P(random_params, vs_fftw)
{
    const int  random_seed_salt = std::get<0>(GetParam());
    const int  dimension        = std::get<1>(GetParam());
    const auto precision        = std::get<2>(GetParam());
    const auto placement        = std::get<3>(GetParam());
    const auto transform_type   = std::get<4>(GetParam());

    rocfft_params params;
    params.transform_type = fft_transform_type_complex_forward;
    params.placement      = placement;
    params.precision      = precision;
    params.transform_type = transform_type;

    int maxlen = 0;
    switch(dimension)
    {
    case 1:
        maxlen = 1 << 15;
        break;
    case 2:
        maxlen = 1 << 10;
        break;
    case 3:
        maxlen = 1 << 6;
        break;
    default:
        ASSERT_TRUE(false);
    }

    std::mt19937 rgen(random_seed + random_seed_salt);
    // Mean value of the exponential distribution is maxlen:
    std::exponential_distribution<double> distribution(1.0 / maxlen);

    for(int idim = 0; idim < dimension; ++idim)
    {
        // NB: the distribution can return 0, so add 1 to avoid this issue.
        params.length.push_back(1 + (size_t)distribution(rgen));
    }

    params.validate();

    if(verbose > 1)
    {
        std::cout << "Random test params:"
                  << "\n\t" << params.str("\n\t") << std::endl;
    }
    if(verbose)
    {
        std::cout << "Token: " << params.token() << std::endl;
    }

    if(!params.valid(verbose))
    {
        std::cout << "Params are not valid\n";
    }

    fft_vs_reference(params);
}

INSTANTIATE_TEST_SUITE_P(random_complex_1d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({1}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn(place_range),
                                            ::testing::ValuesIn(trans_type_range_complex)));

INSTANTIATE_TEST_SUITE_P(random_complex_2d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({2}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn(place_range),
                                            ::testing::ValuesIn(trans_type_range_complex)));

INSTANTIATE_TEST_SUITE_P(random_complex_3d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({3}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn(place_range),
                                            ::testing::ValuesIn(trans_type_range_complex)));

INSTANTIATE_TEST_SUITE_P(random_real_1d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({1}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn({fft_placement_notinplace}),
                                            ::testing::ValuesIn(trans_type_range_real)));

INSTANTIATE_TEST_SUITE_P(random_real_2d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({2}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn({fft_placement_notinplace}),
                                            ::testing::ValuesIn(trans_type_range_real)));

INSTANTIATE_TEST_SUITE_P(random_real_3d,
                         random_params,
                         ::testing::Combine(::testing::Range(0, n_random_tests),
                                            ::testing::ValuesIn({3}),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::ValuesIn({fft_placement_notinplace}),
                                            ::testing::ValuesIn(trans_type_range_real)));
