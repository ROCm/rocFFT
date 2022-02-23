/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(TWIDDLES_H)
#define TWIDDLES_H

#include "../../../shared/gpubuf.h"
#include "rocfft.h"
#include <cassert>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static const size_t LTWD_BASE_DEFAULT       = 8;
static const size_t LARGE_TWIDDLE_THRESHOLD = 4096;

static inline bool IsPo2(size_t u)
{
    return (u != 0) && (0 == (u & (u - 1)));
}

//	help function: Find the smallest power of 2 that is >= n; return its
//  power of 2 factor
//	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
static inline size_t CeilPo2(size_t n)
{
    size_t v = 1, t = 0;
    while(v < n)
    {
        v <<= 1;
        t++;
    }

    return t;
}

template <typename T>
static inline T DivRoundingUp(T a, T b)
{
    return (a + (b - 1)) / b;
}

// N is the logical size of the table.  length_limit, if nonzero,
// limits the physical generated table length (e.g. for real-complex
// pre/post, where we need a table for length-N but will only use 1/4
// of it).
gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       bool                       large,
                       size_t                     largeTwdBase,
                       bool                       no_radices,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices);
gpubuf twiddles_create_2D(size_t N1, size_t N2, rocfft_precision precision);

#endif // defined( TWIDDLES_H )
