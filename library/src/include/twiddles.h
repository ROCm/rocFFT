/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(TWIDDLES_H)
#define TWIDDLES_H

#include "../../../shared/gpubuf.h"
#include "arithmetic.h"
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

gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       size_t                     largeTwdBase,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices);
gpubuf twiddles_create_2D(size_t N1, size_t N2, rocfft_precision precision);

#endif // defined( TWIDDLES_H )
