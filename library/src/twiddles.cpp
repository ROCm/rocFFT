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

// Twiddle factors table
template <typename T>
class TwiddleTable
{
    size_t N; // length
    size_t length_limit; // limit number of generated table elements

public:
    TwiddleTable(size_t _N, size_t _length_limit)
        : N(_N)
        , length_limit(_length_limit ? _length_limit : _N)
    {
    }

    std::vector<T> GenerateTwiddleTable(const std::vector<size_t>& radices)
    {
        // cosine, sine arrays. T is float2 or double2, wc.x stores cosine,
        // wc.y stores sine
        std::vector<T> wc(length_limit);
        const double   TWO_PI = -6.283185307179586476925286766559;

        // Make sure the radices vector multiplication product up to N
        assert(N
               == std::accumulate(std::begin(radices),
                                  std::end(radices),
                                  static_cast<size_t>(1),
                                  std::multiplies<>()));

        // Generate the table
        size_t L  = 1;
        size_t nt = 0;
        for(auto radix : radices)
        {
            L *= radix;

            // Twiddle factors
            for(size_t k = 0; k < (L / radix) && nt < length_limit; k++)
            {
                double theta = TWO_PI * (k) / (L);

                for(size_t j = 1; j < radix && nt < length_limit; j++)
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

        return wc;
    }

    std::vector<T> GenerateTwiddleTable()
    {
        // cosine, sine arrays. T is float2 or double2, wc.x stores cosine,
        // wc.y stores sine
        std::vector<T> wc(length_limit);
        const double   TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt = 0;
        for(size_t i = 0; i < N && nt < length_limit; i++)
        {
            double c = cos(TWO_PI * i / N);
            double s = sin(TWO_PI * i / N);

            wc[nt].x = c;
            wc[nt].y = s;
            nt++;
        }

        return wc;
    }

    // Attach half-N table for potential fused even-length
    // real2complex post-processing or complex2real pre-processing.
    // Pre/post processing requires a table that's a quarter of the
    // real length, but N here is our complex length.  So half-N is
    // what we need.
    void AttachHalfNTable(const std::vector<T>& twtc, gpubuf& twts)
    {
        size_t         half_N       = (N + 1) / 2;
        size_t         total_length = N + half_N;
        std::vector<T> twc_all(total_length);
        std::copy(twtc.begin(), twtc.end(), twc_all.begin());

        const double TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt = N;
        for(size_t i = 0; i < half_N; i++)
        {
            double c = cos(TWO_PI * i / (2 * N));
            double s = sin(TWO_PI * i / (2 * N));

            twc_all[nt].x = c;
            twc_all[nt].y = s;
            nt++;
        }

        if(twts.alloc(total_length * sizeof(T)) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_length));
        if(hipMemcpy(twts.data(), twc_all.data(), total_length * sizeof(T), hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to copy twiddle length "
                                     + std::to_string(total_length));
    }
};

// Twiddle factors table for large N > 4096
// used in 3-step algorithm
template <typename T>
class TwiddleTableLarge
{
    size_t N; // length
    size_t largeTwdBase;
    size_t X, Y;
    size_t tableSize;

public:
    TwiddleTableLarge(size_t length, size_t base = LTWD_BASE_DEFAULT)
        : N(length)
        , largeTwdBase(base)
    {
        X         = static_cast<size_t>(1) << largeTwdBase; // ex: 2^8 = 256
        Y         = DivRoundingUp<size_t>(CeilPo2(N), largeTwdBase);
        tableSize = X * Y;
    }

    std::vector<T> GenerateTwiddleTable()
    {
        std::vector<T> wc(tableSize); // cosine, sine arrays
        const double   TWO_PI = -6.283185307179586476925286766559;

        // Generate the table
        size_t nt  = 0;
        double phi = TWO_PI / double(N);
        for(size_t iY = 0; iY < Y; ++iY)
        {
            size_t i = static_cast<size_t>(1) << (iY * largeTwdBase);
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

        return wc;
    }
};

template <typename T>
gpubuf twiddles_create_pr(size_t                     N,
                          size_t                     length_limit,
                          size_t                     largeTwdBase,
                          bool                       attach_halfN,
                          const std::vector<size_t>& radices)
{
    if(largeTwdBase && length_limit)
        throw std::runtime_error("length-limited large twiddles are not supported");

    gpubuf         twts; // device side
    std::vector<T> twtc; // host side
    if((N <= LARGE_TWIDDLE_THRESHOLD) && largeTwdBase == 0)
    {
        TwiddleTable<T> twTable(N, length_limit);
        if(radices.empty())
        {
            twtc = twTable.GenerateTwiddleTable();
        }
        else
        {
            twtc = twTable.GenerateTwiddleTable(radices); // calculate twiddles on host side
        }

        if(attach_halfN)
        {
            twTable.AttachHalfNTable(twtc, twts);
        }
        else
        {
            if(twts.alloc(twtc.size() * sizeof(T)) != hipSuccess)
                throw std::runtime_error("unable to allocate twiddle length "
                                         + std::to_string(twtc.size()));
            if(hipMemcpy(twts.data(), twtc.data(), twtc.size() * sizeof(T), hipMemcpyHostToDevice)
               != hipSuccess)
                throw std::runtime_error("failed to copy twiddle length "
                                         + std::to_string(twtc.size()));
        }
    }
    else
    {
        assert(!attach_halfN);

        if(largeTwdBase == 0)
        {
            TwiddleTable<T> twTable(N, length_limit);
            twtc = twTable.GenerateTwiddleTable();
            if(twts.alloc(twtc.size() * sizeof(T)) != hipSuccess)
                throw std::runtime_error("unable to allocate twiddle length "
                                         + std::to_string(twtc.size()));
            if(hipMemcpy(twts.data(), twtc.data(), twtc.size() * sizeof(T), hipMemcpyHostToDevice)
               != hipSuccess)
                throw std::runtime_error("failed to copy twiddle length "
                                         + std::to_string(twtc.size()));
        }
        else
        {
            TwiddleTableLarge<T> twTable(N, largeTwdBase); // does not generate radices

            twtc = twTable.GenerateTwiddleTable(); // calculate twiddles on host side

            if(twts.alloc(twtc.size() * sizeof(T)) != hipSuccess)
                throw std::runtime_error("unable to allocate twiddle length "
                                         + std::to_string(twtc.size()));
            if(hipMemcpy(twts.data(), twtc.data(), twtc.size() * sizeof(T), hipMemcpyHostToDevice)
               != hipSuccess)
                throw std::runtime_error("failed to copy twiddle length "
                                         + std::to_string(twtc.size()));
        }
    }

    return twts;
}

gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       size_t                     largeTwdBase,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_pr<float2>(N, length_limit, largeTwdBase, attach_halfN, radices);
    else if(precision == rocfft_precision_double)
        return twiddles_create_pr<double2>(N, length_limit, largeTwdBase, attach_halfN, radices);
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

    TwiddleTable<T> twTable1(N1, 0);
    TwiddleTable<T> twTable2(N2, 0);

    auto           twtc1 = twTable1.GenerateTwiddleTable(radices1);
    std::vector<T> twtc2;
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
    if(hipMemcpy(twts_ptr, twtc1.data(), twtc1.size() * sizeof(T), hipMemcpyHostToDevice)
           != hipSuccess
       || hipMemcpy(twts_ptr + twtc1.size(),
                    twtc2.data(),
                    twtc2.size() * sizeof(T),
                    hipMemcpyHostToDevice)
              != hipSuccess)
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
