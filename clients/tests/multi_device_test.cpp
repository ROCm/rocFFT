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
#include "../../shared/rocfft_params.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

static const std::vector<std::vector<size_t>> multi_gpu_sizes = {
    {256},
    {256, 256},
    {256, 256, 256},
};

std::vector<fft_params> param_generator_multi_gpu(const fft_params::SplitType input_split,
                                                  const fft_params::SplitType output_split,
                                                  size_t                      min_fft_rank = 1)
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

    auto distribute_params = [=, &all_params](const std::vector<fft_params>& params) {
        for(auto& p : params)
        {
            if(p.length.size() < min_fft_rank)
                continue;

            auto p_dist = p;
            p_dist.distribute_input(deviceCount, input_split);
            p_dist.distribute_output(deviceCount, output_split);

            // "placement" flag is meaningless if exactly one of
            // input+output is a field.  So just add those cases if
            // the flag is "out-of-place", since "in-place" is
            // exactly the same test case.
            if(p_dist.placement == fft_placement_inplace
               && p_dist.ifields.empty() != p_dist.ofields.empty())
                continue;
            all_params.push_back(std::move(p_dist));
        }
    };

    distribute_params(params_complex);
    distribute_params(params_real);

    return all_params;
}

// split both input and output on slowest FFT dim
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(
                             fft_params::SplitType::SLOWEST, fft_params::SplitType::SLOWEST)),
                         accuracy_test::TestName);

// split slowest FFT dim only on input, or only on output
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_input_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(
                             fft_params::SplitType::SLOWEST, fft_params::SplitType::NONE)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_output_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(
                             fft_params::SplitType::NONE, fft_params::SplitType::SLOWEST)),
                         accuracy_test::TestName);

// split input on slowest FFT and output on fastest, to minimize data
// movement (only makes sense for rank-2 and higher FFTs)
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowin_fastout,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(
                             fft_params::SplitType::SLOWEST, fft_params::SplitType::FASTEST, 2)),
                         accuracy_test::TestName);

TEST(multi_gpu_validate, catch_validation_errors)
{
    const auto all_split_types = {fft_params::SplitType::NONE,
                                  fft_params::SplitType::SLOWEST,
                                  fft_params::SplitType::FASTEST};

    for(auto input_split : all_split_types)
    {
        for(auto output_split : all_split_types)
        {
            if(input_split == fft_params::SplitType::NONE
               && output_split == fft_params::SplitType::NONE)
                continue;

            // gather all of the multi-GPU test cases
            auto params = param_generator_multi_gpu(input_split, output_split);

            for(size_t i = 0; i < params.size(); ++i)
            {
                auto& param = params[i];

                std::vector<fft_params::fft_field*> available_fields;
                if(input_split != fft_params::SplitType::NONE)
                    available_fields.push_back(&param.ifields.front());
                if(output_split != fft_params::SplitType::NONE)
                    available_fields.push_back(&param.ofields.front());

                // get iterator to the brick we will modify
                auto field      = available_fields[i % available_fields.size()];
                auto brick_iter = field->bricks.begin() + i % field->bricks.size();

                // iterate through the 5 cases we want to test:
                switch(i % 5)
                {
                case 0:
                {
                    // missing brick
                    field->bricks.erase(brick_iter);
                    break;
                }
                case 1:
                {
                    // a brick's lower index too small by one
                    size_t& index = brick_iter->lower[i % brick_iter->lower.size()];
                    // don't worry about underflow since that should also
                    // produce an invalid brick layout
                    --index;
                    break;
                }
                case 2:
                {
                    // a brick's lower index too large by one
                    size_t& index = brick_iter->lower[i % brick_iter->lower.size()];
                    ++index;
                    break;
                }
                case 3:
                {
                    // a brick's upper index too small by one
                    size_t& index = brick_iter->upper[i % brick_iter->lower.size()];
                    // don't worry about underflow since that should also
                    // produce an invalid brick layout
                    --index;
                    break;
                }
                case 4:
                {
                    // a brick's upper index too large by one
                    size_t& index = brick_iter->upper[i % brick_iter->lower.size()];
                    ++index;
                    break;
                }
                }

                rocfft_params rparam{param};
                // brick layout is invalid, so this should fail
                try
                {
                    rparam.setup_structs();
                }
                catch(std::runtime_error&)
                {
                    continue;
                }
                // didn't get an exception, fail the test
                GTEST_FAIL() << "invalid brick layout " << rparam.token()
                             << " should have failed, but plan was created successfully";
            }
        }
    }
}
