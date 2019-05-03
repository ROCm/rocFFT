/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <stdexcept>

#include "./misc.h"
#include "rocfft.h"

void setupBuffers(std::vector<int> devices,
                  const size_t     bufferSizeBytesIn,
                  const unsigned   numBuffersIn,
                  void*            buffersIn[],
                  const size_t     bufferSizeBytesOut,
                  const unsigned   numBuffersOut,
                  void*            buffersOut[])
{
    for(unsigned i = 0; i < numBuffersIn; i++)
        HIP_V_THROW(hipMalloc(&buffersIn[i], bufferSizeBytesIn), "hipMalloc failed");

    for(unsigned i = 0; i < numBuffersOut; i++)
        HIP_V_THROW(hipMalloc(&buffersOut[i], bufferSizeBytesOut), "hipMalloc failed");
}

void clearBuffers(void* buffersIn[], void* buffersOut[])
{

    for(unsigned i = 0; i < 2; i++)
    {
        if(buffersIn[i] != NULL)
            HIP_V_THROW(hipFree(buffersIn[i]), "hipFree failed");
    }

    for(unsigned i = 0; i < 2; i++)
    {
        if(buffersOut[i] != NULL)
            HIP_V_THROW(hipFree(buffersOut[i]), "hipFree failed");
    }
}
