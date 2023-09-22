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

#include "accuracy_test.h"
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

    std::vector<fft_params> all_params
        = param_generator_complex(multi_gpu_sizes,
                                  precision_range_sp_dp,
                                  {1},
                                  stride_generator({{1}}),
                                  stride_generator({{1}}),
                                  {{0, 0}},
                                  {{0, 0}},
                                  {fft_placement_inplace, fft_placement_notinplace},
                                  false);

    for(auto& params : all_params)
    {
        // split up the slowest FFT dimension among the available
        // devices
        size_t slowLen = params.length.front();
        if(slowLen < static_cast<unsigned int>(deviceCount))
            continue;

        // add input and output fields
        auto& ifield = params.ifields.emplace_back();
        auto& ofield = params.ofields.emplace_back();

        for(int i = 0; i < deviceCount; ++i)
        {
            // start at origin
            std::vector<size_t> field_lower(params.length.size());
            std::vector<size_t> field_upper(params.length.size());

            // note: slowest FFT dim is index 1 in these coordinates
            field_lower[0] = slowLen / deviceCount * i;
            // last brick needs to include the whole slow len
            if(i == deviceCount - 1)
                field_upper[0] = slowLen;
            else
                field_upper[0] = std::min(slowLen, field_lower[0] + slowLen / deviceCount);

            for(unsigned int upperDim = 1; upperDim < field_upper.size(); ++upperDim)
            {
                field_upper[upperDim] = params.length[upperDim];
            }

            // field coordinates also need to include batch
            field_lower.insert(field_lower.begin(), 0);
            field_upper.insert(field_upper.begin(), params.nbatch);

            auto field_istride = params.istride;
            field_istride.insert(field_istride.begin(), params.idist);
            auto field_ostride = params.ostride;
            field_ostride.insert(field_ostride.begin(), params.odist);

            ifield.bricks.push_back(
                fft_params::fft_brick{field_lower, field_upper, field_istride, i});
            ofield.bricks.push_back(
                fft_params::fft_brick{field_lower, field_upper, field_ostride, i});
        }
    }
    return all_params;
}

INSTANTIATE_TEST_SUITE_P(multi_gpu,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu()),
                         accuracy_test::TestName);
