/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(TWIDDLES_H)
#define TWIDDLES_H

#include "../../../shared/gpubuf.h"
#include "rocfft.h"
#include <cassert>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

static const size_t LTWD_BASE_DEFAULT = 8;

//	help function: Find the smallest power of 2 that is >= n; return its
//  power of 2 factor
//	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
static inline size_t CeilPo2(size_t n)
{
    size_t v = 1, t = 0;
    while(v < n)
    {
        v <<= 1;
        t++;
    }

    return t;
}

template <typename T>
static inline T DivRoundingUp(T a, T b)
{
    return (a + (b - 1)) / b;
}

// Twiddle factors table
template <typename T>
class TwiddleTable
{
    size_t         N; // length
    std::vector<T> wc; // cosine, sine arrays. T is float2 or double2, wc.x stores cosine,
    // wc.y stores sine

public:
    TwiddleTable(size_t length)
        : N(length)
        , wc(N)
    {
    }

    T* GenerateTwiddleTable(const std::vector<size_t>& radices)
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Make sure the radices vector multiplication product up to N
        assert(N
               == std::accumulate(std::begin(radices), std::end(radices), 1, std::multiplies<>()));

        // Generate the table
        size_t L  = 1;
        size_t nt = 0;
        for(auto radix : radices)
        {
            L *= radix;

            // Twiddle factors
            for(size_t k = 0; k < (L / radix); k++)
            {
                double theta = TWO_PI * (k) / (L);

                for(size_t j = 1; j < radix; j++)
                {
                    double c = cos((j)*theta);
                    double s = sin((j)*theta);

                    // if (fabs(c) < 1.0E-12)    c = 0.0;
                    // if (fabs(s) < 1.0E-12)    s = 0.0;

                    wc[nt].x = c;
                    wc[nt].y = s;
                    nt++;
                }
            }
        } // end of for radices

        return wc.data();
    }

    T* GenerateTwiddleTable()
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt = 0;
        for(size_t i = 0; i < N; i++)
        {
            double c = cos(TWO_PI * i / N);
            double s = sin(TWO_PI * i / N);

            wc[nt].x = c;
            wc[nt].y = s;
            nt++;
        }

        return wc.data();
    }

    // Attach 2N table for potential fused even-length real2complex post-processing
    // or complex2real pre-processing.
    void Attach2NTable(const T* twtc, gpubuf& twts)
    {
        std::vector<T> twc_all(3 * N);
        std::copy(twtc, twtc + N, twc_all.begin());

        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt = N;
        for(size_t i = 0; i < 2 * N; i++)
        {
            double c = cos(TWO_PI * i / (2 * N));
            double s = sin(TWO_PI * i / (2 * N));

            twc_all[nt].x = c;
            twc_all[nt].y = s;
            nt++;
        }

        if(twts.alloc(3 * N * sizeof(T)) != hipSuccess
           || hipMemcpy(twts.data(), twc_all.data(), 3 * N * sizeof(T), hipMemcpyHostToDevice)
                  != hipSuccess)
            twts.free();
    }
};

// Twiddle factors table for large N > 4096
// used in 3-step algorithm
template <typename T>
class TwiddleTableLarge
{
    size_t         N; // length
    size_t         largeTwdBase;
    size_t         X, Y;
    size_t         tableSize;
    std::vector<T> wc; // cosine, sine arrays

public:
    TwiddleTableLarge(size_t length, size_t base = LTWD_BASE_DEFAULT)
        : N(length)
        , largeTwdBase(base)
    {
        X         = size_t(1) << largeTwdBase; // ex: 2^8 = 256
        Y         = DivRoundingUp<size_t>(CeilPo2(N), largeTwdBase);
        tableSize = X * Y;

        wc = std::vector<T>(tableSize);
    }

    std::tuple<size_t, T*> GenerateTwiddleTable()
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt  = 0;
        double phi = TWO_PI / double(N);
        for(size_t iY = 0; iY < Y; ++iY)
        {
            size_t i = size_t(1) << (iY * largeTwdBase);
            for(size_t iX = 0; iX < X; ++iX)
            {
                size_t j = i * iX;

                double c = cos(phi * j);
                double s = sin(phi * j);

                // if (fabs(c) < 1.0E-12)	c = 0.0;
                // if (fabs(s) < 1.0E-12)	s = 0.0;

                wc[nt].x = c;
                wc[nt].y = s;
                nt++;
            }
        } // end of for

        return std::make_tuple(tableSize, wc.data());
    }
};

gpubuf twiddles_create(size_t           N,
                       rocfft_precision precision,
                       bool             large,
                       size_t           largeTwdBase,
                       bool             no_radices,
                       bool             attach_2N);
gpubuf twiddles_create_2D(size_t N1, size_t N2, rocfft_precision precision);

#endif // defined( TWIDDLES_H )
