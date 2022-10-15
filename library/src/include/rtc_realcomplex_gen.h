// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef RTC_REAL2COMPLEX_EMBED_GEN
#define RTC_REAL2COMPLEX_EMBED_GEN

#include "../device/kernels/common.h"
#include "compute_scheme.h"
#include "rocfft.h"
#include "rtc_kernel.h"

#include <vector>

struct RealComplexSpecs
{
    ComputeScheme     scheme;
    size_t            dim;
    rocfft_precision  precision;
    rocfft_array_type inArrayType;
    rocfft_array_type outArrayType;
    bool              enable_callbacks;
    bool              enable_scaling;
};

struct RealComplexEvenSpecs : public RealComplexSpecs
{
    RealComplexEvenSpecs(RealComplexSpecs&& baseSpecs, bool Ndiv4)
        : RealComplexSpecs(baseSpecs)
        , Ndiv4(Ndiv4)
    {
    }
    bool Ndiv4;
};

// generate name for RTC realcomplex kernel
std::string realcomplex_rtc_kernel_name(const RealComplexSpecs& specs);
std::string realcomplex_even_rtc_kernel_name(const RealComplexEvenSpecs& specs);

// generate source for RTC realcomplex kernel.
std::string realcomplex_rtc(const std::string& kernel_name, const RealComplexSpecs& specs);
std::string realcomplex_even_rtc(const std::string& kernel_name, const RealComplexEvenSpecs& specs);

#endif
