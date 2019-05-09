/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
