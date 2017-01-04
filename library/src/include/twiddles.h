/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined( TWIDDLES_H )
#define TWIDDLES_H

#include <math.h>
#include "rocfft.h"

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


// Twiddle factors table
template <typename T>
class TwiddleTable
{
    size_t N; // length
    T *wc; // cosine, sine arrays. T is float2 or double2, wc.x stores cosine, wc.y stores sine

    public:
    TwiddleTable(size_t length) : N(length)
    {
        // Allocate memory for the tables
        wc = new T[N];
    }

    ~TwiddleTable()
    {
        // Free
        delete[] wc;
    }


    T* GenerateTwiddleTable(const std::vector<size_t> &radices)
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Make sure the radices vector multiplication product up to N
        size_t sz = 1;
        for(std::vector<size_t>::const_iterator i = radices.begin();
                  i != radices.end(); i++)
        {
            sz *= (*i);
        }
        assert(sz == N);

        // Generate the table
        size_t L = 1;
        size_t nt = 0;
        for(std::vector<size_t>::const_iterator i = radices.begin();
                  i != radices.end(); i++)
        {
            size_t radix = *i;

            L *= radix;

            // Twiddle factors
            for(size_t k=0; k<(L/radix); k++)
            {
                double theta = TWO_PI * (k)/(L);

                for(size_t j=1; j<radix; j++)
                {
                    double c = cos((j) * theta);
                    double s = sin((j) * theta);

                    //if (fabs(c) < 1.0E-12)    c = 0.0;
                    //if (fabs(s) < 1.0E-12)    s = 0.0;

                    wc[nt].x   = c;
                    wc[nt].y   = s;
                    nt++;
                }
            }
        }//end of for radices

        return wc;
    }
};


void *twiddles_create(size_t N, rocfft_precision precision);
void twiddles_delete(void *twt);


#endif //defined( TWIDDLES_H )
