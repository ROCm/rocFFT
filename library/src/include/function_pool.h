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

    // exampel usage:  function_map_single[std::make_pair(64,CS_KERNEL_STOCKHAM)]
    // = &rocfft_internal_dfn_sp_ci_ci_stoc_1_64;
};

class function_pool
{
    using Key = std::pair<size_t, ComputeScheme>;
    std::unordered_map<Key, DevFnCall, SimpleHash> function_map_single;
    std::unordered_map<Key, DevFnCall, SimpleHash> function_map_double;

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

    static void verify_no_null_functions()
    {
        function_pool& func_pool = get_function_pool();

        for(auto it = func_pool.function_map_single.begin();
            it != func_pool.function_map_single.end();
            ++it)
        {
            if(it->second == nullptr)
            {
                std::cout << "null ptr registered in function_map_single" << std::endl;
            }
        }

        for(auto it = func_pool.function_map_double.begin();
            it != func_pool.function_map_double.end();
            ++it)
        {
            if(it->second == nullptr)
            {
                std::cout << "null ptr registered in function_map_double" << std::endl;
            }
        }
    }
};

#endif // FUNCTION_POOL_H
