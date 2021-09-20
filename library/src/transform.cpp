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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../shared/array_predicate.h"
#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "rocfft.h"
#include "transform.h"

rocfft_status rocfft_execution_info_create(rocfft_execution_info* info)
{
    rocfft_execution_info einfo = new rocfft_execution_info_t;
    *info                       = einfo;
    log_trace(__func__, "info", *info);

    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info)
{
    log_trace(__func__, "info", info);
    if(info != nullptr)
        delete info;

    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info,
                                                    void*                 work_buffer,
                                                    size_t                work_buffer_size)
{
    log_trace(
        __func__, "info", info, "work_buffer", work_buffer, "work_buffer_size", work_buffer_size);
    if(!work_buffer)
        return rocfft_status_invalid_work_buffer;
    info->workBufferSize = work_buffer_size;
    info->workBuffer     = work_buffer;

    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, void* stream)
{
    log_trace(__func__, "info", info, "stream", stream);
    info->rocfft_stream = (hipStream_t)stream;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_load_callback(rocfft_execution_info info,
                                                      void**                cb_functions,
                                                      void**                cb_data,
                                                      size_t                shared_mem_size)
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_size)
        return rocfft_status_invalid_arg_value;

    info->callbacks.load_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.load_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.load_cb_lds_bytes = shared_mem_size;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info,
                                                       void**                cb_functions,
                                                       void**                cb_data,
                                                       size_t                shared_mem_size)
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_size)
        return rocfft_status_invalid_arg_value;

    info->callbacks.store_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.store_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.store_cb_lds_bytes = shared_mem_size;
    return rocfft_status_success;
}

rocfft_status rocfft_execute(const rocfft_plan     plan,
                             void*                 in_buffer[],
                             void*                 out_buffer[],
                             rocfft_execution_info user_info)
{
    log_trace(__func__,
              "plan",
              plan,
              "in_buffer",
              in_buffer,
              "out_buffer",
              out_buffer,
              "info",
              user_info);

    Repo&     repo     = Repo::GetRepo();
    ExecPlan* execPlan = repo.GetPlan(plan);
    if(!execPlan)
        return rocfft_status_failure;

    if(LOG_PLAN_ENABLED())
        PrintNode(*LogSingleton::GetInstance().GetPlanOS(), *execPlan);

    // tolerate user not providing an execution_info
    rocfft_execution_info_t info;
    if(user_info)
        info = *user_info;

    gpubuf autoAllocWorkBuf;

    if(execPlan->workBufSize > 0)
    {
        auto requiredWorkBufBytes = execPlan->WorkBufBytes(plan->base_type_size);
        if(!info.workBuffer)
        {
            // user didn't provide a buffer, alloc one now
            if(autoAllocWorkBuf.alloc(requiredWorkBufBytes) != hipSuccess)
                return rocfft_status_failure;
            info.workBufferSize = requiredWorkBufBytes;
            info.workBuffer     = autoAllocWorkBuf.data();
        }
        // otherwise user provided a buffer, but complain if it's too small
        else if(info.workBufferSize < requiredWorkBufBytes)
            return rocfft_status_invalid_work_buffer;
    }

    // Callbacks do not currently support planar format
    if((array_type_is_planar(execPlan->rootPlan->inArrayType)
        || array_type_is_planar(execPlan->rootPlan->outArrayType))
       && (info.callbacks.load_cb_fn || info.callbacks.store_cb_fn))
        return rocfft_status_failure;

    try
    {
        TransformPowX(*execPlan,
                      in_buffer,
                      (plan->placement == rocfft_placement_inplace) ? in_buffer : out_buffer,
                      &info);
    }
    catch(std::exception&)
    {
        return rocfft_status_failure;
    }

    return rocfft_status_success;
}
