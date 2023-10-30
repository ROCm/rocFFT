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
                                  {1, 10},
                                  stride_generator({{1}}),
                                  stride_generator({{1}}),
                                  {{0, 0}},
                                  {{0, 0}},
                                  {fft_placement_inplace, fft_placement_notinplace},
                                  false);

    auto all_params_real = param_generator_real(multi_gpu_sizes,
                                                precision_range_sp_dp,
                                                {1, 10},
                                                stride_generator({{1}}),
                                                stride_generator({{1}}),
                                                {{0, 0}},
                                                {{0, 0}},
                                                {fft_placement_notinplace},
                                                false);
    std::copy(all_params_real.begin(), all_params_real.end(), std::back_inserter(all_params));
    all_params_real.clear();

    for(auto& params : all_params)
    {
        // split up the slowest FFT dimension among the available
        // devices
        size_t islowLen = params.length.front();
        size_t oslowLen = params.olength().front();
        if(islowLen < static_cast<unsigned int>(deviceCount)
           || oslowLen < static_cast<unsigned int>(deviceCount))
            continue;

        // add input and output fields
        auto& ifield = params.ifields.emplace_back();
        auto& ofield = params.ofields.emplace_back();

        for(int i = 0; i < deviceCount; ++i)
        {
            // start at origin
            std::vector<size_t> ifield_lower(params.length.size());
            std::vector<size_t> ifield_upper(params.length.size());
            std::vector<size_t> ofield_lower(params.length.size());
            std::vector<size_t> ofield_upper(params.length.size());

            // note: slowest FFT dim is index 1 in these coordinates
            ifield_lower[0] = islowLen / deviceCount * i;
            ofield_lower[0] = oslowLen / deviceCount * i;
            // last brick needs to include the whole slow len
            if(i == deviceCount - 1)
            {
                ifield_upper[0] = islowLen;
                ofield_upper[0] = oslowLen;
            }
            else
            {
                ifield_upper[0] = std::min(islowLen, ifield_lower[0] + islowLen / deviceCount);
                ofield_upper[0] = std::min(oslowLen, ofield_lower[0] + oslowLen / deviceCount);
            }

            for(unsigned int upperDim = 1; upperDim < params.length.size(); ++upperDim)
            {
                ifield_upper[upperDim] = params.length[upperDim];
                ofield_upper[upperDim] = params.olength()[upperDim];
            }

            // field coordinates also need to include batch
            ifield_lower.insert(ifield_lower.begin(), 0);
            ofield_lower.insert(ofield_lower.begin(), 0);
            ifield_upper.insert(ifield_upper.begin(), params.nbatch);
            ofield_upper.insert(ofield_upper.begin(), params.nbatch);

            // bricks have contiguous strides
            size_t              brick_idist = 1;
            size_t              brick_odist = 1;
            std::vector<size_t> brick_istride(ifield_lower.size());
            std::vector<size_t> brick_ostride(ofield_lower.size());
            for(size_t i = 0; i < ifield_lower.size(); ++i)
            {
                // fill strides from fastest to slowest
                *(brick_istride.rbegin() + i) = brick_idist;
                brick_idist *= *(ifield_upper.rbegin() + i) - *(ifield_lower.rbegin() + i);

                *(brick_ostride.rbegin() + i) = brick_odist;
                brick_odist *= *(ofield_upper.rbegin() + i) - *(ofield_lower.rbegin() + i);
            }

            ifield.bricks.push_back(
                fft_params::fft_brick{ifield_lower, ifield_upper, brick_istride, i});
            ofield.bricks.push_back(
                fft_params::fft_brick{ofield_lower, ofield_upper, brick_ostride, i});
        }
    }
    return all_params;
}

INSTANTIATE_TEST_SUITE_P(multi_gpu,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu()),
                         accuracy_test::TestName);
