// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef PLAN_H
#define PLAN_H

#include <array>
#include <cstring>
#include <vector>

#include "function_pool.h"
#include "tree_node.h"

// Calculate the maximum pow number with the given base number
template <int base>
constexpr size_t PowMax()
{
    size_t u = base;
    while(u < std::numeric_limits<size_t>::max() / base)
    {
        u = u * base;
    }
    return u;
}

// Generic function to check is pow of a given base number or not
template <int base>
static inline bool IsPow(size_t u)
{
    constexpr size_t max = PowMax<base>(); //Practically, we could save this by using 3486784401
    return (u > 0 && max % u == 0);
}

struct rocfft_plan_description_t
{
    rocfft_array_type inArrayType  = rocfft_array_type_complex_interleaved;
    rocfft_array_type outArrayType = rocfft_array_type_complex_interleaved;

    std::array<size_t, 3> inStrides  = {0, 0, 0};
    std::array<size_t, 3> outStrides = {0, 0, 0};

    size_t inDist  = 0;
    size_t outDist = 0;

    std::array<size_t, 2> inOffset  = {0, 0};
    std::array<size_t, 2> outOffset = {0, 0};

    double scale = 1.0;

    rocfft_plan_description_t() = default;
};

struct rocfft_plan_t
{
    size_t                rank    = 1;
    std::array<size_t, 3> lengths = {1, 1, 1};
    size_t                batch   = 1;

    rocfft_result_placement placement      = rocfft_placement_inplace;
    rocfft_transform_type   transformType  = rocfft_transform_type_complex_forward;
    rocfft_precision        precision      = rocfft_precision_single;
    size_t                  base_type_size = sizeof(float);

    rocfft_plan_description_t desc;

    rocfft_plan_t() = default;

    ExecPlan execPlan;
};

bool PlanPowX(ExecPlan& execPlan);

#endif // PLAN_H
