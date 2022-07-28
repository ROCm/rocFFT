// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TWIDDLE_FACTORS_H
#define TWIDDLE_FACTORS_H

#include "common.h"

static const unsigned int TWIDDLES_THREADS = 32;
static constexpr double   TWO_PI           = -6.283185307179586476925286766559;

template <typename T>
__global__ void __launch_bounds__(TWIDDLES_THREADS* TWIDDLES_THREADS)
    GenerateTwiddleTableKernel(size_t        length_limit,
                               size_t        num_radices,
                               const size_t* radices,
                               const size_t* radices_prod,
                               const size_t* radices_sum_prod,
                               T*            output)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < num_radices - 1)
    {
        auto L     = radices_prod[i];
        auto radix = radices[i + 1];
        auto k     = threadIdx.y + blockIdx.y * blockDim.y;

        if(k < L / radix)
        {
            double theta = TWO_PI * (k) / (L);
            auto   index = radices_sum_prod[i] + k * (radices[i + 1] - 1);

            for(size_t j = 1; j < radix && index < length_limit; ++j)
            {
                output[index].x = cos((j)*theta);
                output[index].y = sin((j)*theta);

                ++index;
            }
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(TWIDDLES_THREADS)
    GenerateTwiddleTableKernel(size_t length_limit, size_t N, T* output)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N && i < length_limit)
    {
        double c = cos(TWO_PI * i / N);
        double s = sin(TWO_PI * i / N);

        output[i].x = c;
        output[i].y = s;
    }
}

template <typename T>
__global__ void __launch_bounds__(TWIDDLES_THREADS)
    GenerateHalfNTableKernel(size_t half_N, size_t N, T* output)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < half_N)
    {
        double c = cos(TWO_PI * i / (2 * N));
        double s = sin(TWO_PI * i / (2 * N));

        output[i].x = c;
        output[i].y = s;
    }
}

template <typename T>
__global__ void __launch_bounds__(TWIDDLES_THREADS* TWIDDLES_THREADS)
    GenerateTwiddleTableLargeKernel(double phi, size_t base, size_t X, size_t Y, T* output)
{
    auto iY = threadIdx.y + blockIdx.y * blockDim.y;

    if(iY < Y)
    {
        auto iX = threadIdx.x + blockIdx.x * blockDim.x;

        if(iX < X)
        {
            auto j = (static_cast<size_t>(1) << (iY * base)) * iX;

            double c = cos(phi * j);
            double s = sin(phi * j);

            auto index = iY * X + iX;

            output[index].x = c;
            output[index].y = s;
        }
    }
}

#endif // TWIDDLE_FACTORS_H