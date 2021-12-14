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

#include "../../shared/gpubuf.h"
#include "../rocfft_params.h"
#include "accuracy_test.h"
#include "rocfft.h"
#include <functional>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <thread>
#include <vector>

TEST(rocfft_UnitTest, 1D_hermitian)
{
    // Run two 1D C2R transforms, on:
    // * random input
    // * identical random input, but modified to be Hermitian-symmetric
    // We should tolerate the input being having non-zero imaginary part in the DC mode
    // and the Nyquist frequency (of the length is even).

    rocfft_params p;
    p.length         = {8};
    p.precision      = fft_precision_double;
    p.transform_type = fft_transform_type_real_inverse;
    p.placement      = fft_placement_notinplace;
    p.validate();

    if(verbose)
    {
        std::cout << p.str("\n\t") << std::endl;
    }

    ASSERT_TRUE(p.valid(verbose));

    std::vector<std::complex<double>> h_input(p.isize[0]);

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for(auto& val : h_input)
    {
        val = std::complex<double>(dis(gen), dis(gen));
    }

    if(verbose)
    {
        std::cout << "non-Hermitian input:";
        for(const auto& val : h_input)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    gpubuf ibuf;
    ASSERT_TRUE(ibuf.alloc(p.ibuffer_sizes()[0]) == hipSuccess);
    ASSERT_TRUE(hipMemcpy(ibuf.data(), h_input.data(), ibuf.size(), hipMemcpyHostToDevice)
                == hipSuccess);

    gpubuf obuf;
    ASSERT_TRUE(obuf.alloc(p.obuffer_sizes()[0]) == hipSuccess);

    gpubuf wbuffer;
    ASSERT_TRUE(p.setup() == fft_status_success);

    std::vector<void*> pibuf = {ibuf.data()};
    std::vector<void*> pobuf = {obuf.data()};
    ASSERT_TRUE(p.execute(pibuf.data(), pobuf.data()) == fft_status_success);

    std::vector<double> h_output(p.osize[0]);
    ASSERT_TRUE(hipMemcpy(h_output.data(), obuf.data(), obuf.size(), hipMemcpyDeviceToHost)
                == hipSuccess);

    ASSERT_TRUE(hipDeviceSynchronize() == hipSuccess);

    if(verbose)
    {
        std::cout << "output:";
        for(const auto& val : h_output)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    std::vector<std::complex<double>> h_input1 = h_input;

    // Impose Hermitian symmetry on the input:
    h_input1[0].imag(0.0);
    if(p.length[0] % 2 == 0)
    {
        h_input1.back().imag(0.0);
    }
    if(verbose)
    {
        std::cout << "Hermitian input:";
        for(const auto& val : h_input1)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    double maxdiff = 0.0;
    for(int i = 0; i < h_input.size(); ++i)
    {
        auto val = std::abs(h_input[i] - h_input1[i]);
        if(val > maxdiff)
            maxdiff = val;
    }
    ASSERT_TRUE(maxdiff > 0.0);

    ASSERT_TRUE(hipMemcpy(ibuf.data(), h_input1.data(), ibuf.size(), hipMemcpyHostToDevice)
                == hipSuccess);
    ASSERT_TRUE(p.execute(pibuf.data(), pobuf.data()) == fft_status_success);
    std::vector<double> h_output1(p.osize[0]);
    ASSERT_TRUE(hipMemcpy(h_output1.data(), obuf.data(), obuf.size(), hipMemcpyDeviceToHost)
                == hipSuccess);

    if(verbose)
    {
        std::cout << "output:";
        for(const auto& val : h_output1)
        {
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }

    double maxerr = 0;
    for(int i = 0; i < h_output.size(); ++i)
    {
        auto val = std::abs(h_output[i] - h_output1[i]);
        if(val > maxerr)
            maxerr = val;
    }

    if(verbose)
        std::cout << maxerr << std::endl;

    EXPECT_TRUE(maxerr == 0.0);
}
