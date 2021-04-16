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

#ifndef FUNCTION_POOL_H
#define FUNCTION_POOL_H

#include "../device/kernels/common.h"
#include "tree_node.h"
#include <unordered_map>

struct SimpleHash
{
    using Key   = std::pair<size_t, ComputeScheme>;
    using Key2D = std::tuple<size_t, size_t, ComputeScheme>;

    size_t operator()(const Key& p) const noexcept
    {
        size_t h1 = std::hash<size_t>{}(std::get<0>(p));
        size_t h2 = std::hash<ComputeScheme>{}(std::get<1>(p));
        return h1 ^ h2;
    }

    size_t operator()(const Key2D& p) const noexcept
    {
        size_t h1 = std::hash<size_t>{}(std::get<0>(p));
        size_t h2 = std::hash<size_t>{}(std::get<1>(p));
        size_t h3 = std::hash<ComputeScheme>{}(std::get<2>(p));
        return h1 ^ h2 ^ h3;
    }
};

struct FFTKernel
{

    DevFnCall           device_function = nullptr;
    std::vector<size_t> factors;
    int                 batches_per_block    = 0;
    int                 threads_per_block    = 0;
    bool                use_3steps_large_twd = false;

    FFTKernel() = delete;
    FFTKernel(DevFnCall fn)
        : device_function(fn)
    {
    }

    FFTKernel(DevFnCall fn, bool use_3steps)
        : device_function(fn)
        , use_3steps_large_twd(use_3steps)
    {
    }

    // used by new generator
    FFTKernel(DevFnCall fn, bool use_3steps, std::vector<size_t> factors, int bpb, int tpb)
        : device_function(fn)
        , factors(factors)
        , batches_per_block(bpb)
        , threads_per_block(tpb)
        , use_3steps_large_twd(use_3steps)
    {
    }
};

class function_pool
{
    using Key   = std::pair<size_t, ComputeScheme>;
    using Key2D = std::tuple<size_t, size_t, ComputeScheme>;

    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_single;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_double;

    // fused transpose kernels can transpose an even multiple of the
    // tiled rows (faster), or the number of required rows is not an
    // even multiple (slower).  diagonal transpose is even better,
    // but requires pow2 cube sizes.
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_single_transpose_diagonal;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_single_transpose_tile_aligned;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_single_transpose_tile_unaligned;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_double_transpose_diagonal;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_double_transpose_tile_aligned;
    std::unordered_map<Key, FFTKernel, SimpleHash> function_map_double_transpose_tile_unaligned;

    std::unordered_map<Key2D, FFTKernel, SimpleHash> function_map_single_2D;
    std::unordered_map<Key2D, FFTKernel, SimpleHash> function_map_double_2D;

    function_pool();

public:
    function_pool(const function_pool&)
        = delete; // delete is a c++11 feature, prohibit copy constructor
    function_pool& operator=(const function_pool&) = delete; // prohibit assignment operator

    static function_pool& get_function_pool()
    {
        static function_pool func_pool;
        return func_pool;
    }

    ~function_pool() {}

    static bool has_function(rocfft_precision precision, const Key k)
    {
        try
        {
            switch(precision)
            {
            case rocfft_precision_single:
                function_pool::get_function_single(k);
                return true;
            case rocfft_precision_double:
                function_pool::get_function_double(k);
                return true;
            }
        }
        catch(std::exception&)
        {
            return false;
        }
    }

    static std::vector<size_t> get_lengths(rocfft_precision precision, ComputeScheme scheme)
    {
        function_pool&      func_pool = get_function_pool();
        std::vector<size_t> lengths;
        switch(precision)
        {
        case rocfft_precision_single:
            for(auto const& kv : func_pool.function_map_single)
                if(kv.first.second == scheme)
                    lengths.push_back(kv.first.first);
            break;
        case rocfft_precision_double:
            for(auto const& kv : func_pool.function_map_double)
                if(kv.first.second == scheme)
                    lengths.push_back(kv.first.first);
            break;
        }
        return lengths;
    }

    static DevFnCall get_function_single(const Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_single.at(mykey).device_function; // return an reference to
        // the value of the key, if
        // not found throw an
        // exception
    }

    static FFTKernel get_kernel(rocfft_precision precision, const Key k)
    {
        switch(precision)
        {
        case rocfft_precision_single:
            return function_pool::get_kernel_single(k);
        case rocfft_precision_double:
            return function_pool::get_kernel_double(k);
        }
    }

    static FFTKernel get_kernel_single(const Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_single.at(mykey);
    }

    static FFTKernel get_kernel_double(const Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_double.at(mykey);
    }

    static DevFnCall get_function_double(const Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_double.at(mykey).device_function;
    }

    static DevFnCall get_function_single_transpose(const Key mykey, SBRC_TRANSPOSE_TYPE type)
    {
        function_pool& func_pool = get_function_pool();
        switch(type)
        {
        case DIAGONAL:
            return func_pool.function_map_single_transpose_diagonal.at(mykey).device_function;
        case TILE_ALIGNED:
            return func_pool.function_map_single_transpose_tile_aligned.at(mykey).device_function;
        case TILE_UNALIGNED:
            return func_pool.function_map_single_transpose_tile_unaligned.at(mykey).device_function;
        }
    }

    static DevFnCall get_function_double_transpose(const Key mykey, SBRC_TRANSPOSE_TYPE type)
    {
        function_pool& func_pool = get_function_pool();
        switch(type)
        {
        case DIAGONAL:
            return func_pool.function_map_double_transpose_diagonal.at(mykey).device_function;
        case TILE_ALIGNED:
            return func_pool.function_map_double_transpose_tile_aligned.at(mykey).device_function;
        case TILE_UNALIGNED:
            return func_pool.function_map_double_transpose_tile_unaligned.at(mykey).device_function;
        }
    }

    static DevFnCall get_function_single_2D(const Key2D mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_single_2D.at(mykey).device_function;
    }

    static DevFnCall get_function_double_2D(const Key2D mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_double_2D.at(mykey).device_function;
    }

    template <class funcmap>
    static void verify_map(const funcmap& fm, const char* description)
    {
        for(auto& f : fm)
        {
            if(!f.second.device_function)
            {
                rocfft_cerr << "null ptr registered in " << description << std::endl;
                abort();
            }
        }
    }
    static void verify_no_null_functions()
    {
        function_pool& func_pool = get_function_pool();

        verify_map(func_pool.function_map_single, "function_map_single");
        verify_map(func_pool.function_map_double, "function_map_double");
        verify_map(func_pool.function_map_single_transpose_tile_aligned,
                   "function_map_single_transpose_tile_aligned");
        verify_map(func_pool.function_map_double_transpose_tile_aligned,
                   "function_map_double_transpose_tile_aligned");
        verify_map(func_pool.function_map_single_transpose_tile_unaligned,
                   "function_map_single_transpose_tile_unaligned");
        verify_map(func_pool.function_map_double_transpose_tile_unaligned,
                   "function_map_double_transpose_tile_unaligned");
        verify_map(func_pool.function_map_single_transpose_diagonal,
                   "function_map_single_transpose_diagonal");
        verify_map(func_pool.function_map_double_transpose_diagonal,
                   "function_map_double_transpose_diagonal");
        verify_map(func_pool.function_map_single_2D, "function_map_single_2D");
        verify_map(func_pool.function_map_double_2D, "function_map_double_2D");
    }
};

#endif // FUNCTION_POOL_H
