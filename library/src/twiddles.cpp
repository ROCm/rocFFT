/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "rocfft_hip.h"
#include "twiddles.h"
#include "radix_table.h"


template <typename T>
void *twiddles_create_pr(size_t N, size_t threshold)
{
    void* twts;//device side
    void* twtc;//host side
    size_t ns = 0; // table size

    std::vector<size_t> radices;

    radices = GetRadices(N);

    if(N <= threshold){
        TwiddleTable<T> twTable(N);
        twtc = twTable.GenerateTwiddleTable(radices); //calculate twiddles on host side

        hipMalloc(&twts, N*sizeof(T) );
        hipMemcpy(twts, twtc, N*sizeof(T), hipMemcpyHostToDevice);
    }
    else{
        TwiddleTableLarge<T> twTable(N); //does not generate radices
        std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); //calculate twiddles on host side

        hipMalloc(&twts, ns*sizeof(T));
        hipMemcpy(twts, twtc, ns*sizeof(T), hipMemcpyHostToDevice);
    }

    return twts;
}

void *twiddles_create(size_t N, rocfft_precision precision)
{
	if(precision == rocfft_precision_single)
		return twiddles_create_pr<float2>(N, Large1DThreshold(precision));
	else if(precision == rocfft_precision_double)
		return twiddles_create_pr<double2>(N, Large1DThreshold(precision));
	else
	{
		assert(false);
		return nullptr;
	}
}

void twiddles_delete(void *twt)
{
    if(twt)
        hipFree(twt);
}
