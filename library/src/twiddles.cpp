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

#include "twiddles.h"
#include "function_pool.h"
#include "rocfft_hip.h"

template <typename T>

gpubuf twiddles_create_pr(size_t              N,
                          size_t              threshold,
                          bool                large,
                          size_t              largeTwdBase,
                          bool                no_radices,
                          bool                attach_2N,
                          std::vector<size_t> radices)
{
    gpubuf twts; // device side
    void*  twtc; // host side
    size_t ns = 0; // table size

    if((N <= threshold) && !large)
    {
        TwiddleTable<T> twTable(N);
        if(no_radices)
        {
            twtc = twTable.GenerateTwiddleTable();
        }
        else
        {
            if(radices.empty())
            {
                throw std::runtime_error("Can't compute twiddle table: missing radices");
            }
            twtc = twTable.GenerateTwiddleTable(radices); // calculate twiddles on host side
        }

        if(attach_2N)
        {
            twTable.Attach2NTable((T*)twtc, twts);
        }
        else
        {
            if(twts.alloc(N * sizeof(T)) != hipSuccess
               || hipMemcpy(twts.data(), twtc, N * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
                twts.free();
        }
    }
    else
    {
        assert(!attach_2N);

        if(no_radices)
        {
            TwiddleTable<T> twTable(N);
            twtc = twTable.GenerateTwiddleTable();
            if(twts.alloc(N * sizeof(T)) != hipSuccess
               || hipMemcpy(twts.data(), twtc, N * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
                twts.free();
        }
        else
        {
            TwiddleTableLarge<T> twTable(N, largeTwdBase); // does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

            if(twts.alloc(ns * sizeof(T)) != hipSuccess
               || hipMemcpy(twts.data(), twtc, ns * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
                twts.free();
        }
    }

    return twts;
}

gpubuf twiddles_create(size_t              N,
                       rocfft_precision    precision,
                       bool                large,
                       size_t              largeTwdBase,
                       bool                no_radices,
                       bool                attach_2N,
                       std::vector<size_t> radices)
{
    if(large)
    {
        if(!largeTwdBase)
            throw std::runtime_error("missing largeTwdBase value for large twiddle");
    }

    if(precision == rocfft_precision_single)
        return twiddles_create_pr<float2>(
            N, LARGE_TWIDDLE_THRESHOLD, large, largeTwdBase, no_radices, attach_2N, radices);
    else if(precision == rocfft_precision_double)
        return twiddles_create_pr<double2>(
            N, LARGE_TWIDDLE_THRESHOLD, large, largeTwdBase, no_radices, attach_2N, radices);
    else
    {
        assert(false);
        return {};
    }
}

template <typename T>
gpubuf twiddles_create_2D_pr(size_t N1, size_t N2, rocfft_precision precision)
{
    auto                kernel = function_pool::get_kernel(fpkey(N1, N2, precision));
    std::vector<size_t> radices1, radices2;

    int    count               = 0;
    size_t cummulative_product = 1;
    while(cummulative_product != N1)
    {
        cummulative_product *= kernel.factors[count++];
    }
    radices1.insert(radices1.cbegin(), kernel.factors.cbegin(), kernel.factors.cbegin() + count);
    radices2.insert(radices2.cbegin(), kernel.factors.cbegin() + count, kernel.factors.cend());
    if(radices1 == radices2)
    {
        N2 = 0;
    }

    TwiddleTable<T> twTable1(N1);
    TwiddleTable<T> twTable2(N2);

    auto twtc1 = twTable1.GenerateTwiddleTable(radices1);
    T*   twtc2 = nullptr;
    if(N2)
    {
        twtc2 = twTable2.GenerateTwiddleTable(radices2);
    }

    // glue those two twiddle tables together in one malloc that we
    // give to the kernel
    gpubuf twts;
    if(twts.alloc((N1 + N2) * sizeof(T)) != hipSuccess)
        return twts;
    auto twts_ptr = static_cast<T*>(twts.data());
    if(hipMemcpy(twts_ptr, twtc1, N1 * sizeof(T), hipMemcpyHostToDevice) != hipSuccess
       || hipMemcpy(twts_ptr + N1, twtc2, N2 * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        twts.free();
    return twts;
}

gpubuf twiddles_create_2D(size_t N1, size_t N2, rocfft_precision precision)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_2D_pr<float2>(N1, N2, precision);
    else if(precision == rocfft_precision_double)
        return twiddles_create_2D_pr<double2>(N1, N2, precision);
    else
    {
        assert(false);
        return {};
    }
}
