/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(KARGS_H)
#define KARGS_H

#include <cstddef>
#include <vector>

#define KERN_ARGS_ARRAY_WIDTH 16

size_t* kargs_create(std::vector<size_t> length,
                     std::vector<size_t> inStride,
                     std::vector<size_t> outStride,
                     size_t              iDist,
                     size_t              oDist);
void    kargs_delete(void* devk);

#endif // defined( KARGS_H )
