// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "load_store_ops.h"
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
    rocfft_array_type inArrayType  = rocfft_array_type_unset;
    rocfft_array_type outArrayType = rocfft_array_type_unset;

    std::vector<size_t> inStrides;
    std::vector<size_t> outStrides;

    size_t inDist  = 0;
    size_t outDist = 0;

    std::array<size_t, 2> inOffset  = {0, 0};
    std::array<size_t, 2> outOffset = {0, 0};

    LoadOps  loadOps;
    StoreOps storeOps;

    rocfft_plan_description_t() = default;

    // A plan description is created in a vacuum and does not know what
    // type of transform it will be for.  Once that's known, we can
    // initialize default values for in/out type, stride, dist if they're
    // unspecified.
    void init_defaults(rocfft_transform_type      transformType,
                       rocfft_result_placement    placement,
                       const std::vector<size_t>& lengths);
};

struct rocfft_plan_t
{
    size_t              rank = 1;
    std::vector<size_t> lengths;
    size_t              batch = 1;

    rocfft_result_placement placement      = rocfft_placement_inplace;
    rocfft_transform_type   transformType  = rocfft_transform_type_complex_forward;
    rocfft_precision        precision      = rocfft_precision_single;
    size_t                  base_type_size = sizeof(float);

    rocfft_plan_description_t desc;

    rocfft_plan_t() = default;

    ExecPlan execPlan;

    // Users can provide lengths+strides in any order, but we'll
    // construct the most sensible plans if they're in row-major order.
    // Sort the FFT dimensions.
    //
    // This should be done when the plan parameters are known, but
    // before we start creating any child nodes from the root plan.
    void sort();
};

bool PlanPowX(ExecPlan& execPlan);
bool GetTuningKernelInfo(ExecPlan& execPlan);

#endif // PLAN_H
