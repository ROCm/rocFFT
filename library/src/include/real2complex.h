/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef REAL_TO_COMPLEX_H
#define REAL_TO_COMPLEX_H

#include "tree_node.h"
#include <unordered_map>

void real2complex(const void* data, void* back);

void complex2hermitian(const void* data, void* back);

void complex2real(const void* data, void* back);

void hermitian2complex(const void* data, void* back);

void real2complex_post(const void* data, void* back)
{
    // FIXME: implement
}

void complex2real_pre(const void* data, void* back)
{
    // FIXME: implement
}

#endif // REAL_TO_COMPLEX_H
