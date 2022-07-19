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

#ifndef FUNCTION_POOL_H
#define FUNCTION_POOL_H

#include "../device/kernels/common.h"
#include "tree_node.h"
#include <sstream>
#include <unordered_map>

using FMKey
    = std::tuple<std::array<size_t, 2>, rocfft_precision, ComputeScheme, SBRC_TRANSPOSE_TYPE>;

static inline FMKey fpkey(size_t              length,
                          rocfft_precision    precision,
                          ComputeScheme       scheme    = CS_KERNEL_STOCKHAM,
                          SBRC_TRANSPOSE_TYPE transpose = NONE)
{
    return {{length, 0}, precision, scheme, transpose};
}

static inline FMKey fpkey(size_t              length1,
                          size_t              length2,
                          rocfft_precision    precision,
                          ComputeScheme       scheme    = CS_KERNEL_2D_SINGLE,
                          SBRC_TRANSPOSE_TYPE transpose = NONE)
{
    return {{length1, length2}, precision, scheme, transpose};
}

inline std::string PrintMissingKernelInfo(const FMKey& key)
{
    const auto&               lengthVec = std::get<0>(key);
    const rocfft_precision    precision = std::get<1>(key);
    const ComputeScheme       scheme    = std::get<2>(key);
    const SBRC_TRANSPOSE_TYPE trans     = std::get<3>(key);
    std::stringstream         msg;
    msg << "Kernel not found: \n"
        << "\tlength: " << lengthVec[0] << "," << lengthVec[1] << "\n"
        << "\tprecision: " << precision << "\n"
        << "\tscheme: " << PrintScheme(scheme) << "\n"
        << "\tSBRC Transpose type: " << PrintSBRCTransposeType(trans) << std::endl;

    return msg.str();
}

struct SimpleHash
{
    size_t operator()(const FMKey& p) const noexcept
    {
        size_t h = 0;
        for(auto& v : std::get<0>(p))
            h ^= std::hash<int>{}(v);
        h ^= std::hash<rocfft_precision>{}(std::get<1>(p));
        h ^= std::hash<ComputeScheme>{}(std::get<2>(p));
        h ^= std::hash<SBRC_TRANSPOSE_TYPE>{}(std::get<3>(p));
        return h;
    }
};

struct FFTKernel
{

    DevFnCall           device_function = nullptr;
    std::vector<size_t> factors;
    // NB:
    //    Some abbrevs for namings that we can follow (tpb/wgs/tpt)
    // number of transforms performed by one threadblock (tpb)
    unsigned int transforms_per_block = 0;
    // workgroup size： number of threads per block (wgs) = tpt * tpb
    int workgroup_size = 0;
    // number of threads to perform single transform (tpt)
    // 2D_SINGLE specifies separate threads for each dimension;
    // otherwise second dim's threads will be 0
    std::array<int, 2> threads_per_transform = {0, 0};
    bool               use_3steps_large_twd  = false;
    bool               half_lds              = false;
    bool               direct_to_from_reg    = false;

    FFTKernel() = delete;

    FFTKernel(const FFTKernel&) = default;
    FFTKernel& operator=(const FFTKernel&) = default;

    FFTKernel(DevFnCall             fn,
              bool                  use_3steps,
              std::vector<size_t>&& factors,
              int                   tpb,
              int                   wgs,
              std::array<int, 2>&&  tpt,
              bool                  half_lds           = false,
              bool                  direct_to_from_reg = false)
        : device_function(fn)
        , factors(factors)
        , transforms_per_block(tpb)
        , workgroup_size(wgs)
        , threads_per_transform(tpt)
        , use_3steps_large_twd(use_3steps)
        , half_lds(half_lds)
        , direct_to_from_reg(direct_to_from_reg)
    {
    }
};

class function_pool
{
    std::unordered_map<FMKey, FFTKernel, SimpleHash> function_map;

    ROCFFT_DEVICE_EXPORT function_pool();

public:
    function_pool(const function_pool&) = delete;
    function_pool& operator=(const function_pool&) = delete;

    static function_pool& get_function_pool()
    {
        static function_pool func_pool;
        return func_pool;
    }

    ~function_pool() {}

    static bool has_function(const FMKey& key)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map.count(key) > 0;
    }

    static size_t get_largest_length(rocfft_precision precision)
    {
        auto supported = function_pool::get_lengths(precision, CS_KERNEL_STOCKHAM);
        auto itr       = std::max_element(supported.cbegin(), supported.cend());
        if(itr != supported.cend())
            return *itr;
        return 0;
    }

    static std::vector<size_t> get_lengths(rocfft_precision precision, ComputeScheme scheme)
    {
        const function_pool& func_pool = get_function_pool();
        std::vector<size_t>  lengths;
        for(auto const& kv : func_pool.function_map)
        {
            if(std::get<0>(kv.first)[1] == 0 && std::get<1>(kv.first) == precision
               && std::get<2>(kv.first) == scheme && std::get<3>(kv.first) == NONE)
            {
                lengths.push_back(std::get<0>(kv.first)[0]);
            }
        }

        return lengths;
    }

    static DevFnCall get_function(const FMKey& key)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map.at(key).device_function;
    }

    static FFTKernel get_kernel(const FMKey& key)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map.at(key);
    }

    // helper for common used
    static bool has_SBCC_kernel(size_t length, rocfft_precision precision)
    {
        return has_function(fpkey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CC));
    }

    static bool has_SBRC_kernel(size_t length, rocfft_precision precision)
    {
        return has_function(fpkey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
    }

    static bool has_SBCR_kernel(size_t length, rocfft_precision precision)
    {
        return has_function(fpkey(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CR));
    }
};

#endif // FUNCTION_POOL_H
