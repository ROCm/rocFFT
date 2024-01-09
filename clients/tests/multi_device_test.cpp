// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/accuracy_test.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

static const std::vector<std::vector<size_t>> multi_gpu_sizes = {
    {256},
    {256, 256},
    {256, 256, 256},
};

std::vector<fft_params> param_generator_multi_gpu()
{
    int deviceCount = 0;
    (void)hipGetDeviceCount(&deviceCount);

    // need multiple devices to test anything
    if(deviceCount < 2)
        return {};

    auto params_complex = param_generator_complex(multi_gpu_sizes,
                                                  precision_range_sp_dp,
                                                  {1, 10},
                                                  stride_generator({{1}}),
                                                  stride_generator({{1}}),
                                                  {{0, 0}},
                                                  {{0, 0}},
                                                  {fft_placement_inplace, fft_placement_notinplace},
                                                  false);

    auto params_real = param_generator_real(multi_gpu_sizes,
                                            precision_range_sp_dp,
                                            {1, 10},
                                            stride_generator({{1}}),
                                            stride_generator({{1}}),
                                            {{0, 0}},
                                            {{0, 0}},
                                            {fft_placement_notinplace},
                                            false);

    std::vector<fft_params> all_params;

    auto distribute_params = [&all_params, deviceCount](const std::vector<fft_params>& params) {
        for(auto& p : params)
        {
            // run tests for:
            // - multi-device input, normal output
            // - multi-device output, normal input
            // - multi-device both
            auto p_in = p;
            p_in.distribute_input(deviceCount);
            auto p_out = p;
            p_out.distribute_output(deviceCount);
            auto p_both = p;
            p_both.distribute_input(deviceCount);
            p_both.distribute_output(deviceCount);

            // "placement" flag is meaningless if exactly one of
            // input+output is a field.  So just add those cases if
            // the flag is "out-of-place", since "in-place" is
            // exactly the same test case.
            if(p.placement == fft_placement_notinplace)
            {
                all_params.emplace_back(std::move(p_in));
                all_params.emplace_back(std::move(p_out));
            }
            all_params.emplace_back(std::move(p_both));
        }
    };

    distribute_params(params_complex);
    distribute_params(params_real);

    return all_params;
}

INSTANTIATE_TEST_SUITE_P(multi_gpu,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu()),
                         accuracy_test::TestName);
