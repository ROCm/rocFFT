/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "twiddles.h"

std::vector<size_t> get_radices(size_t N)
{
    // Pow of 2 radix table, implemented in pow2.h
    const std::vector<size_t> radices_pow2_2048 = {8, 8, 8, 4};
    const std::vector<size_t> radices_pow2_4096 = {16, 16, 16};

    switch(N)
    {
    case 4096:
        return radices_pow2_4096;
    case 2048:
        return radices_pow2_2048;
    default:
        return radices_pow2_2048;
    }
}

void* twiddles_create(size_t N)
{
    void*  twts; // device side
    void*  twtc; // host side
    size_t ns = 0; // table size

    if(N <= 4096)
    {
        TwiddleTable<float2> twTable(N);
        std::vector<size_t>  radices
            = get_radices(N); // get radices from the radice table based on length N

        twtc = twTable.GenerateTwiddleTable(radices); // calculate twiddles on host side

        hipMalloc(&twts, N * sizeof(float2));
        hipMemcpy(twts, twtc, N * sizeof(float2), hipMemcpyHostToDevice);
    }
    else
    {

        TwiddleTableLarge<float2> twTable(N); // does not generate radices
        std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

        hipMalloc(&twts, ns * sizeof(float2));
        hipMemcpy(twts, twtc, ns * sizeof(float2), hipMemcpyHostToDevice);
    }

    return twts;
}

void twiddles_delete(void* twt)
{
    if(twt)
        hipFree(twt);
}
