/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#ifndef FUNCTION_POOL_H
#define FUNCTION_POOL_H

#include <unordered_map>
#include "tree_node.h"

struct SimpleHash
{
    size_t operator()(const std::pair<size_t, ComputeScheme>& p) const
    {
        using std::hash;
        return (hash<size_t>()(p.first) ^ hash<int8_t>()((int8_t)p.second));// or 
    }

	//exampel usage:  function_map_single[std::make_pair(64,CS_KERNEL_STOCKHAM)] = &rocfft_internal_dfn_sp_ci_ci_stoc_1_64;
};

class function_pool
{
    using Key = std::pair<size_t, ComputeScheme>; 
    std::unordered_map<Key, DevFnCall, SimpleHash> function_map_single;
    std::unordered_map<Key, DevFnCall, SimpleHash> function_map_double;

    function_pool();

public:
    function_pool(const function_pool &) = delete; // delete is a c++11 feature, prohibit copy constructor 
    function_pool &operator=(const function_pool &) = delete; //prohibit assignment operator


    static function_pool &get_function_pool()
    {
        static function_pool func_pool;
        return func_pool;
    }

    ~function_pool()
    {
    }

    static DevFnCall get_function_single(Key mykey)
    {
        function_pool &func_pool = get_function_pool();
        return func_pool.function_map_single.at(mykey);//return an reference to the value of the key, if not found throw an exception
    }

    static DevFnCall get_function_double(Key mykey)
    {
        function_pool &func_pool = get_function_pool();
        return func_pool.function_map_double.at(mykey);
    }


};


#endif // FUNCTION_POOL_H

