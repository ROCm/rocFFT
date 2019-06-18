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
void* twiddles_create_pr(size_t N, size_t threshold, bool large, bool no_radices)
{
    void*  twts; // device side
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

        hipMalloc(&twts, N * sizeof(T));
        hipMemcpy(twts, twtc, N * sizeof(T), hipMemcpyHostToDevice);
    }
    else
    {
        if(no_radices)
        {
            TwiddleTable<T> twTable(N);
            twtc = twTable.GenerateTwiddleTable();
            hipMalloc(&twts, N * sizeof(T));
            hipMemcpy(twts, twtc, N * sizeof(T), hipMemcpyHostToDevice);
        }
        else
        {
            TwiddleTableLarge<T> twTable(N); // does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

            hipMalloc(&twts, ns * sizeof(T));
            hipMemcpy(twts, twtc, ns * sizeof(T), hipMemcpyHostToDevice);
        }
    }

    return twts;
}

void* twiddles_create(size_t N, rocfft_precision precision, bool large, bool no_radices)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_pr<float2>(N, Large1DThreshold(precision), large, no_radices);
    else if(precision == rocfft_precision_double)
        return twiddles_create_pr<double2>(N, Large1DThreshold(precision), large, no_radices);
    else
    {
        assert(false);
        return nullptr;
    }
}

void twiddles_delete(void* twt)
{
    if(twt)
        hipFree(twt);
}
