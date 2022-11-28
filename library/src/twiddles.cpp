/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../shared/arithmetic.h"
#include "device/kernels/twiddle_factors.h"
#include "function_pool.h"
#include "rocfft_hip.h"
#include <cassert>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

// RAII wrapper around hipStream_t
struct hipStream_wrapper_t
{
    hipStream_wrapper_t()
        : stream(nullptr)
    {
    }
    void alloc()
    {
        if(stream == nullptr && hipStreamCreate(&stream) != hipSuccess)
            throw std::runtime_error("hipStreamCreate failure");
    }
    operator hipStream_t()
    {
        return stream;
    }
    ~hipStream_wrapper_t()
    {
        if(stream)
            (void)hipStreamDestroy(stream);
    }

private:
    hipStream_t stream;
};

// this vector stores streams for each device id.  index in the
// vector is device id.  note that this vector needs to be protected
// against concurrent access, but twiddles are always accessed
// through the Repo which guarantees exclusive access.
static std::vector<hipStream_wrapper_t> twiddle_streams;

void twiddle_streams_cleanup()
{
    twiddle_streams.clear();
}

// Twiddle factors table
template <typename T>
class TwiddleTable
{
protected:
    size_t N;
    size_t half_N;
    size_t length_limit; // limit number of generated table elements

    // Attach half-N table for potential fused even-length
    // real2complex post-processing or complex2real pre-processing.
    // Pre/post processing requires a table that's a quarter of the
    // real length, but N here is our complex length.  So half-N is
    // what we need.
    bool attach_halfN;

    void GetKernelParams(const std::vector<size_t>& radices,
                         std::vector<size_t>&       radices_prod,
                         std::vector<size_t>&       radices_sum_prod,
                         size_t&                    max_radix_prod,
                         size_t&                    min_radix,
                         size_t&                    table_sz)
    {
        radices_sum_prod = {0};
        radices_prod     = {};

        size_t prod      = 1;
        size_t prod_next = radices.at(0);
        size_t sum       = 0;

        for(size_t i = 0; i < radices.size() - 1; ++i)
        {
            auto radix      = radices.at(i);
            auto radix_next = radices.at(i + 1);

            prod *= radix;
            prod_next *= radix_next;
            sum += prod * (radix_next - 1);
            radices_sum_prod.push_back(sum);
            radices_prod.push_back(prod_next);
        }

        if(radices_prod.empty())
            radices_prod.push_back(radices.at(0));

        max_radix_prod = *(std::max_element(std::begin(radices_prod), std::end(radices_prod)));
        min_radix      = *(std::min_element(std::begin(radices), std::end(radices)));

        auto M          = radices.size() - 1;
        auto last_radix = radices.at(M);
        table_sz        = M ? radices_sum_prod.at(M - 1)
                           + ((radices_prod.at(M - 1) / last_radix) - 1) * (last_radix - 1)
                           + (last_radix - 1)
                            : radices_sum_prod.at(0);
    }

    void GenerateTable(const std::vector<size_t>& radices, hipStream_t& stream, gpubuf& output)
    {
        if(radices.size() > TWIDDLES_MAX_RADICES)
            throw std::runtime_error("maximum twiddle radices exceeded");

        size_t              table_sz, maxElem, minElem;
        std::vector<size_t> radices_prod, radices_sum_prod;

        GetKernelParams(radices, radices_prod, radices_sum_prod, maxElem, minElem, table_sz);

        table_sz          = std::min(table_sz, length_limit);
        auto total_length = attach_halfN ? table_sz + half_N : table_sz;

        auto table_bytes = total_length * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_length));

        auto num_radices = radices.size();

        auto blockSize  = TWIDDLES_THREADS;
        auto numBlocksX = DivRoundingUp<size_t>(num_radices, blockSize);
        auto numBlocksY = DivRoundingUp<size_t>(maxElem / minElem, blockSize);

        auto device_data_ptr = static_cast<T*>(output.data());

        radices_t radices_device;
        radices_t radices_prod_device;
        radices_t radices_sum_prod_device;
        std::copy(radices.begin(), radices.end(), radices_device.data);
        std::copy(radices_prod.begin(), radices_prod.end(), radices_prod_device.data);
        std::copy(radices_sum_prod.begin(), radices_sum_prod.end(), radices_sum_prod_device.data);

        hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                           dim3(numBlocksX, numBlocksY),
                           dim3(blockSize, blockSize),
                           0, // sharedMemBytes
                           stream,
                           length_limit,
                           num_radices,
                           radices_device,
                           radices_prod_device,
                           radices_sum_prod_device,
                           device_data_ptr);

        if(attach_halfN)
        {
            auto numBlocks_halfN = DivRoundingUp<size_t>(half_N, blockSize);

            hipLaunchKernelGGL(GenerateHalfNTableKernel<T>,
                               dim3(numBlocks_halfN),
                               dim3(blockSize),
                               0, // sharedMemBytes
                               stream,
                               half_N,
                               N,
                               device_data_ptr + table_sz);
        }
    }

    void GenerateTable(hipStream_t& stream, gpubuf& output)
    {
        auto length       = std::min(N, length_limit);
        auto total_length = attach_halfN ? length + half_N : length;
        auto table_bytes  = total_length * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_length));

        auto blockSize = TWIDDLES_THREADS;
        auto numBlocks = DivRoundingUp<size_t>(length, blockSize);

        auto device_data_ptr = static_cast<T*>(output.data());

        hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                           dim3(numBlocks),
                           dim3(blockSize),
                           0, // sharedMemBytes
                           stream,
                           length_limit,
                           N,
                           device_data_ptr);

        if(attach_halfN)
        {
            auto numBlocks_halfN = DivRoundingUp<size_t>(half_N, blockSize);

            hipLaunchKernelGGL(GenerateHalfNTableKernel<T>,
                               dim3(numBlocks_halfN),
                               dim3(blockSize),
                               0, // sharedMemBytes
                               stream,
                               half_N,
                               N,
                               device_data_ptr + length);
        }
    }

public:
    TwiddleTable(size_t _N, size_t _length_limit, bool _attach_halfN)
        : N(_N)
        , length_limit(_length_limit ? _length_limit : _N)
        , attach_halfN(_attach_halfN)
    {
        half_N = attach_halfN ? (N + 1) / 2 : 0;
    }

    void GenerateTwiddleTable(const std::vector<size_t>& radices, hipStream_t& stream, gpubuf& twts)
    {
        auto use_radices = !radices.empty();
        use_radices ? GenerateTable(radices, stream, twts) : GenerateTable(stream, twts);
    }
};

template <typename T>
class TwiddleTable2D : public TwiddleTable<T>
{
private:
    size_t N1;
    size_t N2;

public:
    TwiddleTable2D(size_t _N1, size_t _N2)
        : TwiddleTable<T>(0, 0, false)
        , N1(_N1)
        , N2(_N2)
    {
    }

    void GenerateTwiddleTable(const std::vector<size_t>& radices1,
                              const std::vector<size_t>& radices2,
                              hipStream_t&               stream,
                              gpubuf&                    output)
    {
        if(radices1 == radices2)
            N2 = 0;

        auto blockSize = TWIDDLES_THREADS;

        size_t              table_sz_1, maxElem_1, minElem_1;
        std::vector<size_t> radices_prod_1, radices_sum_prod_1;

        TwiddleTable<T>::GetKernelParams(
            radices1, radices_prod_1, radices_sum_prod_1, maxElem_1, minElem_1, table_sz_1);

        size_t              table_sz_2, maxElem_2, minElem_2;
        std::vector<size_t> radices_prod_2, radices_sum_prod_2;

        if(N2)
            TwiddleTable<T>::GetKernelParams(
                radices2, radices_prod_2, radices_sum_prod_2, maxElem_2, minElem_2, table_sz_2);
        else
            table_sz_2 = N2;

        auto table_sz    = (table_sz_1 + table_sz_2);
        auto table_bytes = table_sz * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(table_sz));

        auto device_data_ptr = static_cast<T*>(output.data());

        auto num_radices_1 = radices1.size();

        auto numBlocksX_1 = DivRoundingUp<size_t>(num_radices_1, blockSize);
        auto numBlocksY_1 = DivRoundingUp<size_t>(maxElem_1 / minElem_1, blockSize);

        radices_t radices1_device;
        radices_t radices_prod_device_1;
        radices_t radices_sum_prod_device_1;
        std::copy(radices1.begin(), radices1.end(), radices1_device.data);
        std::copy(radices_prod_1.begin(), radices_prod_1.end(), radices_prod_device_1.data);
        std::copy(
            radices_sum_prod_1.begin(), radices_sum_prod_1.end(), radices_sum_prod_device_1.data);

        hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                           dim3(numBlocksX_1, numBlocksY_1),
                           dim3(blockSize, blockSize),
                           0, // sharedMemBytes
                           stream,
                           N1,
                           num_radices_1,
                           radices1_device,
                           radices_prod_device_1,
                           radices_sum_prod_device_1,
                           device_data_ptr);

        if(N2)
        {
            auto num_radices_2 = radices2.size();

            auto numBlocksX_2 = DivRoundingUp<size_t>(num_radices_2, blockSize);
            auto numBlocksY_2 = DivRoundingUp<size_t>(maxElem_2 / minElem_2, blockSize);

            radices_t radices2_device;
            radices_t radices_prod_device_2;
            radices_t radices_sum_prod_device_2;
            std::copy(radices2.begin(), radices2.end(), radices2_device.data);
            std::copy(radices_prod_2.begin(), radices_prod_2.end(), radices_prod_device_2.data);
            std::copy(radices_sum_prod_2.begin(),
                      radices_sum_prod_2.end(),
                      radices_sum_prod_device_2.data);

            hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                               dim3(numBlocksX_2, numBlocksY_2),
                               dim3(blockSize, blockSize),
                               0, // sharedMemBytes
                               stream,
                               N2,
                               num_radices_2,
                               radices2_device,
                               radices_prod_device_2,
                               radices_sum_prod_device_2,
                               device_data_ptr + table_sz_1);
        }
    }
};

// Twiddle factors table for large N > 4096
// used in 3-step algorithm
template <typename T>
class TwiddleTableLarge
{
private:
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

    void GenerateTwiddleTable(hipStream_t& stream, gpubuf& output)
    {
        auto table_bytes = tableSize * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(tableSize));

        auto blockSize = TWIDDLES_THREADS;

        double phi = TWO_PI / double(N);

        auto numBlocksX = DivRoundingUp<size_t>(X, blockSize);
        auto numBlocksY = DivRoundingUp<size_t>(Y, blockSize);

        hipLaunchKernelGGL(GenerateTwiddleTableLargeKernel<T>,
                           dim3(numBlocksX, numBlocksY),
                           dim3(blockSize, blockSize),
                           0, // sharedMemBytes
                           stream,
                           phi,
                           largeTwdBase,
                           X,
                           Y,
                           static_cast<T*>(output.data()));
    }
};

template <typename T>
gpubuf twiddles_create_pr(size_t                     N,
                          size_t                     length_limit,
                          size_t                     largeTwdBase,
                          bool                       attach_halfN,
                          const std::vector<size_t>& radices,
                          unsigned int               deviceId)
{
    if(largeTwdBase && length_limit)
        throw std::runtime_error("length-limited large twiddles are not supported");

    gpubuf twts;
    if(deviceId >= twiddle_streams.size())
        twiddle_streams.resize(deviceId + 1);
    if(twiddle_streams[deviceId] == nullptr)
        twiddle_streams[deviceId].alloc();
    hipStream_t stream = twiddle_streams[deviceId];

    if(stream == nullptr)
    {
        if(hipStreamCreate(&stream) != hipSuccess)
            throw std::runtime_error("hipStreamCreate failure");
    }

    if((N <= LARGE_TWIDDLE_THRESHOLD) && largeTwdBase == 0)
    {
        TwiddleTable<T> twTable(N, length_limit, attach_halfN);
        twTable.GenerateTwiddleTable(radices, stream, twts);
    }
    else
    {
        assert(!attach_halfN);

        if(largeTwdBase == 0)
        {
            TwiddleTable<T> twTable(N, length_limit, attach_halfN);
            twTable.GenerateTwiddleTable(radices, stream, twts);
        }
        else
        {
            TwiddleTableLarge<T> twTable(N, largeTwdBase); // does not generate radices
            twTable.GenerateTwiddleTable(stream, twts);
        }
    }

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return twts;
}

gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       size_t                     largeTwdBase,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices,
                       unsigned int               deviceId)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_pr<float2>(
            N, length_limit, largeTwdBase, attach_halfN, radices, deviceId);
    else if(precision == rocfft_precision_double)
        return twiddles_create_pr<double2>(
            N, length_limit, largeTwdBase, attach_halfN, radices, deviceId);
    else
    {
        assert(false);
        return {};
    }
}

template <typename T>
gpubuf
    twiddles_create_2D_pr(size_t N1, size_t N2, rocfft_precision precision, unsigned int deviceId)
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

    gpubuf twts;
    if(deviceId >= twiddle_streams.size())
        twiddle_streams.resize(deviceId + 1);
    if(twiddle_streams[deviceId] == nullptr)
        twiddle_streams[deviceId].alloc();
    hipStream_t stream = twiddle_streams[deviceId];

    if(stream == nullptr)
    {
        if(hipStreamCreate(&stream) != hipSuccess)
            throw std::runtime_error("hipStreamCreate failure");
    }

    TwiddleTable2D<T> twTable(N1, N2);
    twTable.GenerateTwiddleTable(radices1, radices2, stream, twts);

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return twts;
}

gpubuf twiddles_create_2D(size_t N1, size_t N2, rocfft_precision precision, unsigned int deviceId)
{
    if(precision == rocfft_precision_single)
        return twiddles_create_2D_pr<float2>(N1, N2, precision, deviceId);
    else if(precision == rocfft_precision_double)
        return twiddles_create_2D_pr<double2>(N1, N2, precision, deviceId);
    else
    {
        assert(false);
        return {};
    }
}
