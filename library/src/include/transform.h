/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "rocfft_hip.h"

struct rocfft_execution_info_t
{
    void*       workBuffer;
    size_t      workBufferSize;
    hipStream_t rocfft_stream = 0; // by default it is stream 0
    rocfft_execution_info_t()
        : workBuffer(nullptr)
        , workBufferSize(0)
    {
    }
};

void TransformPowX(const ExecPlan&       execPlan,
                   void*                 in_buffer[],
                   void*                 out_buffer[],
                   rocfft_execution_info info);

#endif // TRANSFORM_H
