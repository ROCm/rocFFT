/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../shared/precision_type.h"
#include "logging.h"
#include "plan.h"
#include "rocfft/rocfft.h"
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
                                                    const size_t          size_in_bytes)
{
    log_trace(__func__, "info", info, "work_buffer", work_buffer, "size_in_bytes", size_in_bytes);
    if(!work_buffer)
        return rocfft_status_invalid_work_buffer;
    info->workBufferSize = size_in_bytes;
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
                                                      size_t                shared_mem_bytes)
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_bytes)
        return rocfft_status_invalid_arg_value;

    info->callbacks.load_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.load_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.load_cb_lds_bytes = shared_mem_bytes;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info,
                                                       void**                cb_functions,
                                                       void**                cb_data,
                                                       size_t                shared_mem_bytes)
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_bytes)
        return rocfft_status_invalid_arg_value;

    info->callbacks.store_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.store_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.store_cb_lds_bytes = shared_mem_bytes;
    return rocfft_status_success;
}

std::vector<size_t> rocfft_plan_t::MultiPlanTopologicalSort() const
{
    std::vector<size_t> ret;
    std::vector<bool>   visited(multiPlan.size());

    for(size_t i = 0; i < multiPlan.size(); ++i)
    {
        if(!visited[i])
            TopologicalSortDFS(i, visited, ret);
    }
    return ret;
}

void rocfft_plan_t::TopologicalSortDFS(size_t               idx,
                                       std::vector<bool>&   visited,
                                       std::vector<size_t>& sorted) const
{
    visited[idx] = true;
    for(auto adjacent : multiPlanAdjacency[idx])
    {
        if(!visited[adjacent])
        {
            TopologicalSortDFS(adjacent, visited, sorted);
        }
    }
    sorted.push_back(idx);
}

void rocfft_plan_t::LogFields(const char* description, const std::vector<rocfft_field_t>& fields)
{
    if(!LOG_PLAN_ENABLED())
        return;

    auto& os = *LogSingleton::GetInstance().GetPlanOS();

    for(size_t fieldIdx = 0; fieldIdx < fields.size(); ++fieldIdx)
    {
        const auto& f = fields[fieldIdx];

        os << description << " field " << fieldIdx << ":" << std::endl;
        for(size_t brickIdx = 0; brickIdx < f.bricks.size(); ++brickIdx)
        {
            const auto& b = f.bricks[brickIdx];
            os << "  brick " << brickIdx << ":" << std::endl;
            os << "    rank: " << b.rank << std::endl;
            os << "    device: " << b.device << std::endl;
            os << "    lower bound:";
            for(auto i : b.lower)
                os << " " << i;
            os << std::endl;
            os << "    upper bound:";
            for(auto i : b.upper)
                os << " " << i;
            os << std::endl;

            os << "    stride:";
            for(auto i : b.stride)
                os << " " << i;
            os << std::endl;

            auto len = b.length();
            os << "    length:";
            for(auto i : len)
                os << " " << i;
            os << std::endl;

            os << "    elements: " << b.count_elems() << std::endl;
        }
    }
}

void rocfft_plan_t::LogSortedPlan(const std::vector<size_t>& sortedIdx) const
{
    if(!LOG_PLAN_ENABLED())
        return;

    auto& os = *LogSingleton::GetInstance().GetPlanOS();

    // if we have a single-node plan, just log that without any extra
    // indenting
    if(multiPlan.size() == 1)
    {
        multiPlan.front()->Print(os, 0);
        return;
    }

    for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
    {
        auto idx = *i;
        os << "multiPlan idx " << idx;

        const auto& antecedents = multiPlanAdjacency[idx];
        if(!antecedents.empty())
        {
            os << "(depends on";
            for(auto antecedentIdx : antecedents)
            {
                os << " " << antecedentIdx;
            }
            os << ")";
        }
        os << std::endl;

        if(!multiPlan[idx])
            os << "  (null)" << std::endl;
        else
            multiPlan[idx]->Print(os, 1);
        os << std::endl;
    }
}

void rocfft_plan_t::Execute(rocfft_rank_t         currentRank,
                            void*                 in_buffer[],
                            void*                 out_buffer[],
                            rocfft_execution_info info)
{
    // vector of topologically sorted indexes to the items in multiPlan
    auto sortedIdx = MultiPlanTopologicalSort();

    LogFields("input", desc.inFields);
    LogFields("output", desc.outFields);

    LogSortedPlan(sortedIdx);

    for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
    {
        auto idx = *i;

        if(!multiPlan[idx])
            continue;

        auto& item = *multiPlan[idx];
        if(!item.RunsOnRank(currentRank))
            continue;

        // so now we have an item that involves this rank somehow

        for(auto antecedentIdx : multiPlanAdjacency[idx])
        {
            if(!multiPlan[antecedentIdx])
                continue;

            // check if antecedent involved us
            auto& antecedent = *multiPlan[antecedentIdx];
            if(!antecedent.RunsOnRank(currentRank))
                continue;

            // the antecedent involved us somehow, wait for it
            antecedent.Wait();
        }

        // done waiting for all our antecedents, so this item can now proceed

        // launch this item async
        item.ExecuteAsync(this, currentRank, in_buffer, out_buffer, info);
    }

    // finished executing all items, wait for outstanding work to complete
    for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
    {
        auto idx = *i;

        if(!multiPlan[idx])
            continue;

        auto& item = *multiPlan[idx];
        item.Wait();
    }
}

rocfft_status rocfft_execute(const rocfft_plan     plan,
                             void*                 in_buffer[],
                             void*                 out_buffer[],
                             rocfft_execution_info info)
{
    log_trace(
        __func__, "plan", plan, "in_buffer", in_buffer, "out_buffer", out_buffer, "info", info);

    if(!plan)
        return rocfft_status_failure;

    try
    {
        plan->Execute(0, in_buffer, out_buffer, info);
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }
    catch(rocfft_status e)
    {
        return e;
    }
    return rocfft_status_success;
}

void ExecPlan::ExecuteAsync(const rocfft_plan     plan,
                            const rocfft_rank_t   rank,
                            void*                 in_buffer[],
                            void*                 out_buffer[],
                            rocfft_execution_info info)
{
    rocfft_scoped_device dev(deviceID);

    // tolerate user not providing an execution_info
    rocfft_execution_info_t exec_info;
    if(info)
        exec_info = *info;

    // allocate stream for async operations if necessary
    if(!exec_info.rocfft_stream)
    {
        this->stream.alloc();
        exec_info.rocfft_stream = this->stream;
    }
    event.alloc();

    // TransformPowX below needs in_buffer, out_buffer to work with.
    // But we need to potentially override pointers in those arrays.
    // So copy them to temporary vectors.
    std::vector<void*> in_buffer_copy;
    std::copy_n(in_buffer,
                plan->desc.count_pointers(plan->desc.inFields, plan->desc.inArrayType),
                std::back_inserter(in_buffer_copy));

    // if input/output are overridden, override now
    if(inputPtr)
        in_buffer_copy[0] = inputPtr;

    std::vector<void*> out_buffer_copy = in_buffer_copy;
    if(rootPlan->placement == rocfft_placement_notinplace)
    {
        out_buffer_copy.clear();
        std::copy_n(out_buffer,
                    plan->desc.count_pointers(plan->desc.outFields, plan->desc.outArrayType),
                    std::back_inserter(out_buffer_copy));
    }

    if(outputPtr)
        out_buffer_copy[0] = outputPtr;

    gpubuf autoAllocWorkBuf;

    if(workBufSize > 0)
    {
        auto requiredWorkBufBytes = WorkBufBytes(real_type_size(rootPlan->precision));
        if(!exec_info.workBuffer)
        {
            // user didn't provide a buffer, alloc one now
            if(autoAllocWorkBuf.alloc(requiredWorkBufBytes) != hipSuccess)
                throw std::runtime_error("work buffer allocation failure");
            exec_info.workBufferSize = requiredWorkBufBytes;
            exec_info.workBuffer     = autoAllocWorkBuf.data();
        }
        // otherwise user provided a buffer, but complain if it's too small
        else if(exec_info.workBufferSize < requiredWorkBufBytes)
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "user work buffer too small" << std::endl;
            throw rocfft_status_invalid_work_buffer;
        }
    }

    // Callbacks do not currently support planar format
    if((array_type_is_planar(rootPlan->inArrayType) || array_type_is_planar(rootPlan->outArrayType))
       && (exec_info.callbacks.load_cb_fn || exec_info.callbacks.store_cb_fn))
        throw std::runtime_error("callbacks not supported with planar format");

    try
    {
        TransformPowX(*this,
                      in_buffer_copy.data(),
                      (rootPlan->placement == rocfft_placement_inplace) ? in_buffer_copy.data()
                                                                        : out_buffer_copy.data(),
                      &exec_info);
        // all work is enqueued to the stream, record the event on
        // the stream
        if(hipEventRecord(event, exec_info.rocfft_stream) != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        throw;
    }
}

void ExecPlan::Wait()
{
    if(hipEventSynchronize(event) != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");
}
