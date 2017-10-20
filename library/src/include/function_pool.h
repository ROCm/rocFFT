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

public:
    //function_pool(const function_pool &) = delete; // delete is a c++11 feature, prohibit copy constructor 
    //function_pool &operator=(const function_pool &) = delete; //prohibit assignment operator

    function_pool();

    ~function_pool()
    {
    }

    DevFnCall get_function_single(Key mykey)
    {
        return function_map_single.at(mykey);//return an reference to the value of the key, if not found throw an exception
/*
        std::unordered_map<size_t, DevFnCall>::const_iterator iter = function_map_single.find (length);

        if ( iter == function_map_single.end() ){
            std::cout << "no implementation is found" << std::endl;
            return nullptr;
        }
        else{
            return iter->second;
        }
*/
    }

    DevFnCall get_function_double(Key mykey)
    {
        return function_map_double.at(mykey);
    }


};


#endif // FUNCTION_POOL_H

