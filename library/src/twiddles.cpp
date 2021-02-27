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
#include "radix_table.h"
#include "rocfft_hip.h"

template <typename T>
gpubuf twiddles_create_pr(size_t N, size_t threshold, bool large, bool no_radices, bool attach_2N)
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
            std::vector<size_t> radices;
            radices = GetRadices(N);
            twtc    = twTable.GenerateTwiddleTable(radices); // calculate twiddles on host side
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
            TwiddleTableLarge<T> twTable(N); // does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

            if(twts.alloc(ns * sizeof(T)) != hipSuccess
               || hipMemcpy(twts.data(), twtc, ns * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
                twts.free();
        }
    }

    return twts;
}

gpubuf twiddles_create(
    size_t N, rocfft_precision precision, bool large, bool no_radices, bool attach_2N)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_pr<float2>(
            N, Large1DThreshold(precision), large, no_radices, attach_2N);
    else if(precision == rocfft_precision_double)
        return twiddles_create_pr<double2>(
            N, Large1DThreshold(precision), large, no_radices, attach_2N);
    else
    {
        assert(false);
        return {};
    }
}

template <typename T>
gpubuf twiddles_create_2D_pr(size_t N1, size_t N2)
{
    // create just one twiddle table if we can get away with it
    if(N1 == N2)
        N2 = 0;
    std::vector<size_t> radices;

    TwiddleTable<T> twTable1(N1);
    TwiddleTable<T> twTable2(N2);
    // generate twiddles for each dimension separately
    radices    = GetRadices(N1);
    auto twtc1 = twTable1.GenerateTwiddleTable(radices);
    T*   twtc2 = nullptr;
    if(N2)
    {
        radices = GetRadices(N2);
        twtc2   = twTable2.GenerateTwiddleTable(radices);
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
        return twiddles_create_2D_pr<float2>(N1, N2);
    else if(precision == rocfft_precision_double)
        return twiddles_create_2D_pr<double2>(N1, N2);
    else
    {
        assert(false);
        return {};
    }
}
