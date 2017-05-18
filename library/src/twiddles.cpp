/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "rocfft_hip.h"
#include "twiddles.h"

std::vector<size_t> get_radices(size_t N)
{
	// Pow of 2 radix table, implemented in pow2.h
	const std::vector<size_t> radices_pow2_1 = {1};
	const std::vector<size_t> radices_pow2_2 = {2};
	const std::vector<size_t> radices_pow2_4 = {2, 2};
	const std::vector<size_t> radices_pow2_8 = {4, 2};
	const std::vector<size_t> radices_pow2_16 = {4, 4};
	const std::vector<size_t> radices_pow2_32 = {8, 4};
	const std::vector<size_t> radices_pow2_64 = {4, 4, 4};
	const std::vector<size_t> radices_pow2_128 = {8, 4, 4};
	const std::vector<size_t> radices_pow2_256 = {4, 4, 4, 4};
	const std::vector<size_t> radices_pow2_512 = {8, 8, 8};
	const std::vector<size_t> radices_pow2_1024 = {8, 8, 4, 4};
	const std::vector<size_t> radices_pow2_2048 = {8, 8, 8, 4};
	const std::vector<size_t> radices_pow2_4096 = {16, 16, 16};


    switch (N)
    {
        case 4096:     return radices_pow2_4096;
        case 2048:     return radices_pow2_2048;
        case 1024:     return radices_pow2_1024;
        case 512:      return radices_pow2_512;
        case 256:      return radices_pow2_256;
        case 128:      return radices_pow2_128;
        case 64:       return radices_pow2_64;
        case 32:       return radices_pow2_32;
        case 16:       return radices_pow2_16;
        case 8:        return radices_pow2_8;
        case 4:        return radices_pow2_4;
        case 2:        return radices_pow2_2;
        case 1:     return radices_pow2_1;
        default:    return radices_pow2_1;
    }
}



void *twiddles_create(size_t N, rocfft_precision precision)
{
    void* twts;//device side
    void* twtc;//host side
    size_t ns = 0; // table size

    if( precision == rocfft_precision_single){
        if(N <= 4096){
            TwiddleTable<float2> twTable(N); 
            std::vector<size_t> radices = get_radices(N); //get radices from the radice table based on length N

            twtc = twTable.GenerateTwiddleTable(radices); //calculate twiddles on host side

            hipMalloc(&twts, N*sizeof( float2 ) );
            hipMemcpy(twts, twtc, N*sizeof( float2 ), hipMemcpyHostToDevice);
        }
        else{
    
            TwiddleTableLarge<float2> twTable(N); //does not generate radices
            std::tie(ns, twtc) = twTable.GenerateTwiddleTable(); //calculate twiddles on host side

            hipMalloc(&twts, ns*sizeof(float2));
            hipMemcpy(twts, twtc, ns*sizeof(float2), hipMemcpyHostToDevice);                        
        }    
    }
    else if( precision == rocfft_precision_double){
        if(N <= 4096){
            TwiddleTable<double2> twTable(N); 
            std::vector<size_t> radices = get_radices(N); //get radices from the radice table based on length N

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
