/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef PLAN_H
#define PLAN_H

#include <cstring>
#include <vector>

#include "tree_node.h"

#define MIN(A, B) (((A) < (B)) ? (A) : (B))

static inline bool IsPo2(size_t u)
{
    return (u != 0) && (0 == (u & (u - 1)));
}

inline bool SupportedLength(size_t len)
{
    size_t p = len;
    while(!(p % 2))
        p /= 2;
    while(!(p % 3))
        p /= 3;
    while(!(p % 5))
        p /= 5;
    // while(!(p%7)) p /= 7;

    if(p == 1)
        return true;
    else
        return false;
}

inline size_t FindBlue(size_t len)
{
    size_t p = 1;
    while(p < len)
        p <<= 1;
    return 2 * p;
}

struct rocfft_plan_description_t
{

    rocfft_array_type inArrayType, outArrayType;

    size_t inStrides[3];
    size_t outStrides[3];

    size_t inDist;
    size_t outDist;

    size_t inOffset[2];
    size_t outOffset[2];

    double scale;

    rocfft_plan_description_t()
    {
        inArrayType  = rocfft_array_type_complex_interleaved;
        outArrayType = rocfft_array_type_complex_interleaved;

        inStrides[0] = 0;
        inStrides[1] = 0;
        inStrides[2] = 0;

        outStrides[0] = 0;
        outStrides[1] = 0;
        outStrides[2] = 0;

        inDist  = 0;
        outDist = 0;

        inOffset[0]  = 0;
        inOffset[1]  = 0;
        outOffset[0] = 0;
        outOffset[1] = 0;

        scale = 1.0;
    }
};

struct rocfft_plan_t
{
    size_t rank;
    size_t lengths[3];
    size_t batch;

    rocfft_result_placement placement;
    rocfft_transform_type   transformType;
    rocfft_precision        precision;
    size_t                  base_type_size;

    rocfft_plan_description_t desc;

    rocfft_plan_t()
        : placement(rocfft_placement_inplace)
        , rank(1)
        , batch(1)
        , transformType(rocfft_transform_type_complex_forward)
        , precision(rocfft_precision_single)
        , base_type_size(sizeof(float))
    {
        lengths[0] = 1;
        lengths[1] = 1;
        lengths[2] = 1;
    }

    bool operator<(const rocfft_plan_t& b) const
    {
        const rocfft_plan_t& a = *this;

        return (memcmp(&a, &b, sizeof(rocfft_plan_t)) < 0 ? true : false);
    }
};

void PlanPowX(ExecPlan& execPlan);

#endif // PLAN_H
