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

// interface for generation of stockham kernels

#pragma once
#include "rocfft.h"
#include <numeric>
#include <string>
#include <vector>

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

struct StockhamGeneratorSpecs
{
    StockhamGeneratorSpecs(const std::vector<unsigned int>& factors,
                           const std::vector<unsigned int>& factors2d,
                           const std::vector<unsigned int>& precisions,
                           unsigned int                     threads_per_block,
                           const std::string&               scheme)
        : factors(factors)
        , factors2d(factors2d)
        , precisions(precisions)
        , length(product(factors.begin(), factors.end()))
        , length2d(product(factors2d.begin(), factors2d.end()))
        , threads_per_block(threads_per_block)
        , scheme(scheme)
    {
    }

    std::vector<unsigned int> factors;
    std::vector<unsigned int> factors2d;
    std::vector<unsigned int> precisions; // mapped from rocfft_precision
    unsigned int              length;
    unsigned int              length2d = 0;

    unsigned int threads_per_block;
    unsigned int threads_per_transform = 0;
    unsigned int block_width           = 0;
    bool         half_lds              = false;
    std::string  scheme;
};

// generate default stockham variants for ahead-of-time compilation
std::string stockham_variants(StockhamGeneratorSpecs& specs, StockhamGeneratorSpecs& specs2d);
