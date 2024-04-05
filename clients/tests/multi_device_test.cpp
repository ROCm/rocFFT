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
#include "params_gen.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

static const std::vector<std::vector<size_t>> multi_gpu_sizes = {
    {256},
    {256, 256},
    {256, 256, 256},
};

enum SplitType
{
    // split both input and output on slow FFT dimension
    SLOW_INOUT,
    // split only input on slow FFT dimension, output is not split
    SLOW_IN,
    // split only output on slow FFT dimension, input is not split
    SLOW_OUT,
    // split input on slow FFT dimension, and output on fast FFT dimension
    SLOW_IN_FAST_OUT,
    // 3D pencil decomposition - one dimension is contiguous on input
    // and another dimension contiguous on output, remaining dims are
    // both split
    PENCIL_3D,
};

std::vector<fft_params> param_generator_multi_gpu(const SplitType type)
{
    int deviceCount = 0;
    (void)hipGetDeviceCount(&deviceCount);

    // need multiple devices to test anything
    if(deviceCount < 2)
        return {};

    auto params_complex = param_generator_complex(multi_gpu_sizes,
                                                  precision_range_sp_dp,
                                                  {4, 1},
                                                  stride_generator({{1}}),
                                                  stride_generator({{1}}),
                                                  {{0, 0}},
                                                  {{0, 0}},
                                                  {fft_placement_inplace, fft_placement_notinplace},
                                                  false);

    auto params_real = param_generator_real(multi_gpu_sizes,
                                            precision_range_sp_dp,
                                            {4, 1},
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
            // start with all-ones in grids
            std::vector<unsigned int> input_grid(p.length.size() + 1, 1);
            std::vector<unsigned int> output_grid(p.length.size() + 1, 1);

            auto p_dist = p;
            switch(type)
            {
            case SLOW_INOUT:
                input_grid[1]  = deviceCount;
                output_grid[1] = deviceCount;
                break;
            case SLOW_IN:
                input_grid[1] = deviceCount;
                break;
            case SLOW_OUT:
                output_grid[1] = deviceCount;
                break;
            case SLOW_IN_FAST_OUT:
                // requires at least rank-2 FFT
                if(p.length.size() < 2)
                    continue;
                input_grid[1]      = deviceCount;
                output_grid.back() = deviceCount;
                break;
            case PENCIL_3D:
                // need at least 2 bricks per split dimension, or 4 devices.
                // also needs to be a 3D problem.
                if(deviceCount < 4 || p.length.size() != 3)
                    continue;

                // make fast dimension contiguous on input
                input_grid[1] = static_cast<unsigned int>(sqrt(deviceCount));
                input_grid[2] = deviceCount / input_grid[1];
                // make middle dimension contiguous on output
                output_grid[1] = input_grid[1];
                output_grid[3] = input_grid[2];
                break;
            }

            p_dist.distribute_input(input_grid);
            p_dist.distribute_output(output_grid);

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
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_INOUT)),
                         accuracy_test::TestName);

// split slowest FFT dim only on input, or only on output
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_input_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_output_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_OUT)),
                         accuracy_test::TestName);

// split input on slowest FFT and output on fastest, to minimize data
// movement (only makes sense for rank-2 and higher FFTs)
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowin_fastout,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN_FAST_OUT)),
                         accuracy_test::TestName);

// 3D pencil decompositions
INSTANTIATE_TEST_SUITE_P(multi_gpu_3d_pencils,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(PENCIL_3D)),
                         accuracy_test::TestName);

TEST(multi_gpu_validate, catch_validation_errors)
{
    const auto all_split_types = {
        SLOW_INOUT,
        SLOW_IN,
        SLOW_OUT,
        SLOW_IN_FAST_OUT,
        PENCIL_3D,
    };

    for(auto type : all_split_types)
    {
        // gather all of the multi-GPU test cases
        auto params = param_generator_multi_gpu(type);

        for(size_t i = 0; i < params.size(); ++i)
        {
            auto& param = params[i];

            std::vector<fft_params::fft_field*> available_fields;
            if(!param.ifields.empty())
                available_fields.push_back(&param.ifields.front());
            if(!param.ofields.empty())
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
