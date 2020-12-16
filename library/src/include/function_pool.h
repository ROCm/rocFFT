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

#include "tree_node.h"
#include <unordered_map>

struct SimpleHash
{
    size_t operator()(const std::pair<size_t, ComputeScheme>& p) const
    {
        using std::hash;
        size_t hash_in = 0;
        hash_in |= ((size_t)(p.first));
        hash_in |= ((size_t)(p.second) << 32);
        return hash<size_t>()(hash_in);
    }

    std::size_t operator()(const std::tuple<size_t, size_t, ComputeScheme>& p) const noexcept
    {
        std::size_t h1 = std::hash<size_t>{}(std::get<0>(p));
        std::size_t h2 = std::hash<size_t>{}(std::get<1>(p));
        std::size_t h3 = std::hash<ComputeScheme>{}(std::get<2>(p));
        return h1 ^ h2 ^ h3;
    }

    // example usage:  function_map_single[std::make_pair(64,CS_KERNEL_STOCKHAM)]
    // = &rocfft_internal_dfn_sp_ci_ci_stoc_1_64;
};

class function_pool
{
    using Key   = std::pair<size_t, ComputeScheme>;
    using Key2D = std::tuple<size_t, size_t, ComputeScheme>;

    std::unordered_map<Key, DevFnCall, SimpleHash>   function_map_single;
    std::unordered_map<Key, DevFnCall, SimpleHash>   function_map_double;
    std::unordered_map<Key2D, DevFnCall, SimpleHash> function_map_single_2D;
    std::unordered_map<Key2D, DevFnCall, SimpleHash> function_map_double_2D;

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

    static bool has_function(rocfft_precision precision, Key k)
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

    static DevFnCall get_function_single(Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_single.at(mykey); // return an reference to
        // the value of the key, if
        // not found throw an
        // exception
    }

    static DevFnCall get_function_double(Key mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_double.at(mykey);
    }

    static DevFnCall get_function_single_2D(Key2D mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_single_2D.at(mykey);
    }

    static DevFnCall get_function_double_2D(Key2D mykey)
    {
        function_pool& func_pool = get_function_pool();
        return func_pool.function_map_double_2D.at(mykey);
    }

    static void verify_no_null_functions()
    {
        function_pool& func_pool = get_function_pool();

        for(auto it = func_pool.function_map_single.begin();
            it != func_pool.function_map_single.end();
            ++it)
        {
            if(it->second == nullptr)
            {
                rocfft_cout << "null ptr registered in function_map_single" << std::endl;
            }
        }

        for(auto it = func_pool.function_map_double.begin();
            it != func_pool.function_map_double.end();
            ++it)
        {
            if(it->second == nullptr)
            {
                rocfft_cout << "null ptr registered in function_map_double" << std::endl;
            }
        }
    }
};

#endif // FUNCTION_POOL_H
