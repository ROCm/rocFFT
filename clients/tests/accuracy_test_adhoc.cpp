// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "../client_utils.h"

#include "accuracy_test.h"

std::vector<std::vector<size_t>> adhoc_sizes = {
    // sizes that exercise L1D_TRTRT subplan of 2D_RTRT or 3D_TRTRTR
    {1, 35},
    {1, 70},
    // currently fails for some transforms
    // {81, 63, 36},

    // L1D_CC subplan of 3D_TRTRTR
    {4, 4, 8192},
};

const static std::vector<std::vector<size_t>> stride_range = {{1}};

static std::vector<std::vector<size_t>> ioffset_range_zero = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero = {{0, 0}};

static std::vector<std::vector<size_t>> ioffset_range = {{0, 0}, {1, 1}};
static std::vector<std::vector<size_t>> ooffset_range = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(adhoc_sizes,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range_zero,
                                                             ooffset_range_zero,
                                                             place_range)),
                         accuracy_test::TestName);

INSTANTIATE_TEST_SUITE_P(DISABLED_offset_adhoc,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator(adhoc_sizes,
                                                             precision_range,
                                                             batch_range,
                                                             stride_range,
                                                             stride_range,
                                                             ioffset_range,
                                                             ooffset_range,
                                                             place_range)),
                         accuracy_test::TestName);

inline auto param_permissive_iodist()
{
    std::vector<std::vector<size_t>> lengths = adhoc_sizes;
    lengths.push_back({4});

    std::vector<rocfft_params> params;
    for(const auto precision : precision_range)
    {
        for(const auto trans_type : trans_type_range)
        {
            for(const auto& types : generate_types(trans_type, place_range))
            {
                for(const auto& len : lengths)
                {
                    rocfft_params param;

                    param.length         = len;
                    param.precision      = precision;
                    param.idist          = 2;
                    param.odist          = 3;
                    param.transform_type = std::get<0>(types);
                    param.placement      = std::get<1>(types);
                    param.itype          = std::get<2>(types);
                    param.otype          = std::get<3>(types);
                    params.push_back(param);
                }
            }
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(adhoc_dist,
                         accuracy_test,
                         ::testing::ValuesIn(param_permissive_iodist()),
                         accuracy_test::TestName);