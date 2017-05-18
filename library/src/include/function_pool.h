/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#ifndef FUNCTION_POOL_H
#define FUNCTION_POOL_H

#include <unordered_map>
#include "tree_node.h"

class function_pool
{
	std::unordered_map<size_t, DevFnCall> function_map_single;
	std::unordered_map<size_t, DevFnCall> function_map_double;

public:
	function_pool(const function_pool &) = delete; // delete is a c++11 feature, prohibit copy constructor 
	function_pool &operator=(const function_pool &) = delete; //prohibit assignment operator

	function_pool();

	~function_pool()
	{
	}

	DevFnCall get_function_single(const size_t length)
    {
        std::unordered_map<size_t, DevFnCall>::const_iterator iter = function_map_single.find (length);

        if ( iter == function_map_single.end() ){
            std::cout << "no implementation is found";
            return nullptr;
        }
        else{
            return iter->second;
        }
    }

	DevFnCall get_function_double(const size_t length)
    {
        std::unordered_map<size_t, DevFnCall>::const_iterator iter = function_map_double.find (length);

        if ( iter == function_map_single.end() ){
            std::cout << "no implementation is found";
            return nullptr;
        }
        else{
            return iter->second;
        }
    }


};


#endif // FUNCTION_POOL_H

