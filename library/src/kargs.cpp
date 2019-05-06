/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "kargs.h"
#include "rocfft_hip.h"

// malloc device buffer; copy host buffer to device buffer
size_t* kargs_create(std::vector<size_t> length,
                     std::vector<size_t> inStride,
                     std::vector<size_t> outStride,
                     size_t              iDist,
                     size_t              oDist)
{
    void* devk;
    hipMalloc(&devk, 3 * KERN_ARGS_ARRAY_WIDTH * sizeof(size_t));

    size_t devkHost[3 * KERN_ARGS_ARRAY_WIDTH];

    size_t i = 0;
    while(i < 3 * KERN_ARGS_ARRAY_WIDTH)
        devkHost[i++] = 0;

    i = 0;
    while(i < length.size())
    {
        devkHost[i + 0 * KERN_ARGS_ARRAY_WIDTH] = length[i];
        devkHost[i + 1 * KERN_ARGS_ARRAY_WIDTH] = inStride[i];
        devkHost[i + 2 * KERN_ARGS_ARRAY_WIDTH] = outStride[i];
        i++;
    }

    devkHost[i + 1 * KERN_ARGS_ARRAY_WIDTH] = iDist;
    devkHost[i + 2 * KERN_ARGS_ARRAY_WIDTH] = oDist;

    hipMemcpy(devk, devkHost, 3 * KERN_ARGS_ARRAY_WIDTH * sizeof(size_t), hipMemcpyHostToDevice);
    return (size_t*)devk;
}

void kargs_delete(void* devk)
{
    if(devk)
        hipFree(devk);
}
