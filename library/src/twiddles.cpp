/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "rocfft_hip.h"
#include "twiddles.h"
#include "radix_table.h"



void *twiddles_create(size_t N, rocfft_precision precision)
{

    void* twts;//device side
    void* twtc;//host side
    size_t ns = 0; // table size

    std::vector<size_t> radices;

    radices = GetRadices(N);


    if( precision == rocfft_precision_single){
        if(N <= 4096){
            TwiddleTable<float2> twTable(N);

            twtc = twTable.GenerateTwiddleTable(radices); //calculate twiddles on host side

            hipMalloc(&twts, N*sizeof( float2 ) );
            hipMemcpy(twts, twtc, N*sizeof( float2 ), hipMemcpyHostToDevice);
        }
        else{
    printf("large table%d\n",N);
            TwiddleTableLarge<float2> twTable(N); //does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); //calculate twiddles on host side

            hipMalloc(&twts, ns*sizeof(float2));
            hipMemcpy(twts, twtc, ns*sizeof(float2), hipMemcpyHostToDevice);
        }
    }
    else if( precision == rocfft_precision_double){
        if(N <= 2048){
            TwiddleTable<double2> twTable(N);

            twtc = twTable.GenerateTwiddleTable(radices); //calculate twiddles on host side

            hipMalloc(&twts, N*sizeof( double2 ) );
            hipMemcpy(twts, twtc, N*sizeof( double2 ), hipMemcpyHostToDevice);
        }
        else{

            TwiddleTableLarge<double2> twTable(N); //does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); //calculate twiddles on host side

            hipMalloc(&twts, ns*sizeof(double2));
            hipMemcpy(twts, twtc, ns*sizeof(double2), hipMemcpyHostToDevice);
        }
    }

    return twts;

}

void twiddles_delete(void *twt)
{
    if(twt)
        hipFree(twt);
}
