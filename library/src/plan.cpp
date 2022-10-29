// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "plan.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "../../shared/environment.h"
#include "../../shared/ptrdiff.h"
#include "assignment_policy.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "node_factory.h"
#include "rocfft-version.h"
#include "rocfft.h"
#include "rocfft_ostream.hpp"
#include "rtc_kernel.h"

#include <algorithm>
#include <assert.h>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)

// clang-format off
#define ROCFFT_VERSION_STRING (TO_STR(rocfft_version_major) "." \
                               TO_STR(rocfft_version_minor) "." \
                               TO_STR(rocfft_version_patch) "." \
                               TO_STR(rocfft_version_tweak) )
// clang-format on

std::string PrintOperatingBuffer(const OperatingBuffer ob)
{
    const std::map<OperatingBuffer, const char*> BuffertoString
        = {{ENUMSTR(OB_UNINIT)},
           {ENUMSTR(OB_USER_IN)},
           {ENUMSTR(OB_USER_OUT)},
           {ENUMSTR(OB_TEMP)},
           {ENUMSTR(OB_TEMP_CMPLX_FOR_REAL)},
           {ENUMSTR(OB_TEMP_BLUESTEIN)}};
    return BuffertoString.at(ob);
}

std::string PrintOperatingBufferCode(const OperatingBuffer ob)
{
    const std::map<OperatingBuffer, const char*> BuffertoString = {{OB_UNINIT, "ERR"},
                                                                   {OB_USER_IN, "A"},
                                                                   {OB_USER_OUT, "B"},
                                                                   {OB_TEMP, "T"},
                                                                   {OB_TEMP_CMPLX_FOR_REAL, "C"},
                                                                   {OB_TEMP_BLUESTEIN, "S"}};
    return BuffertoString.at(ob);
}

std::string PrintOptimizeStrategy(const rocfft_optimize_strategy ros)
{
    const std::map<rocfft_optimize_strategy, const char*> StrategytoString
        = {{rocfft_optimize_min_buffer, "MINIMIZE_BUFFER"},
           {rocfft_optimize_balance, "BALANCE_BUFFER_FUSION"},
           {rocfft_optimize_max_fusion, "MAXIMIZE_FUSION"}};
    return StrategytoString.at(ros);
}

std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty)
{
    const std::map<SBRC_TRANSPOSE_TYPE, const char*> TypetoString = {
        {ENUMSTR(NONE)}, {ENUMSTR(DIAGONAL)}, {ENUMSTR(TILE_ALIGNED)}, {ENUMSTR(TILE_UNALIGNED)}};
    return TypetoString.at(ty);
}

std::string PrintDirectToFromRegMode(const DirectRegType ty)
{
    const std::map<DirectRegType, const char*> TypetoString
        = {{ENUMSTR(FORCE_OFF_OR_NOT_SUPPORT)}, {ENUMSTR(TRY_ENABLE_IF_SUPPORT)}};
    return TypetoString.at(ty);
}

rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description,
                                                       const double            scale_factor)
{
    log_trace(__func__, "description", description, "scale", scale_factor);
    if(!std::isfinite(scale_factor))
        return rocfft_status_invalid_arg_value;
    description->scale_factor = scale_factor;
    return rocfft_status_success;
}

static size_t offset_count(rocfft_array_type type)
{
    // planar data has 2 sets of offsets, otherwise we have one
    return type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar
               ? 2
               : 1;
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type,
                                                      const size_t*           in_offsets,
                                                      const size_t*           out_offsets,
                                                      const size_t            in_strides_size,
                                                      const size_t*           in_strides,
                                                      const size_t            in_distance,
                                                      const size_t            out_strides_size,
                                                      const size_t*           out_strides,
                                                      const size_t            out_distance)
{
    log_trace(__func__,
              "description",
              description,
              "in_array_type",
              in_array_type,
              "out_array_type",
              out_array_type,
              "in_offsets",
              std::make_pair(in_offsets, offset_count(in_array_type)),
              "out_offsets",
              std::make_pair(out_offsets, offset_count(out_array_type)),
              "in_strides",
              std::make_pair(in_strides, in_strides_size),
              "in_distance",
              in_distance,
              "out_strides",
              std::make_pair(out_strides, out_strides_size),
              "out_distance",
              out_distance);

    description->inArrayType  = in_array_type;
    description->outArrayType = out_array_type;

    if(in_offsets != nullptr)
    {
        description->inOffset[0] = in_offsets[0];
        if((in_array_type == rocfft_array_type_complex_planar)
           || (in_array_type == rocfft_array_type_hermitian_planar))
            description->inOffset[1] = in_offsets[1];
    }

    if(out_offsets != nullptr)
    {
        description->outOffset[0] = out_offsets[0];
        if((out_array_type == rocfft_array_type_complex_planar)
           || (out_array_type == rocfft_array_type_hermitian_planar))
            description->outOffset[1] = out_offsets[1];
    }

    if(in_strides != nullptr)
    {
        for(size_t i = 0; i < std::min((size_t)3, in_strides_size); i++)
            description->inStrides[i] = in_strides[i];
    }

    if(in_distance != 0)
        description->inDist = in_distance;

    if(out_strides != nullptr)
    {
        for(size_t i = 0; i < std::min((size_t)3, out_strides_size); i++)
            description->outStrides[i] = out_strides[i];
    }

    if(out_distance != 0)
        description->outDist = out_distance;

    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_create(rocfft_plan_description* description)
{
    rocfft_plan_description desc = new rocfft_plan_description_t;
    *description                 = desc;
    log_trace(__func__, "description", *description);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description)
{
    log_trace(__func__, "description", description);
    if(description != nullptr)
        delete description;
    return rocfft_status_success;
}

std::string rocfft_rider_command(rocfft_plan plan)
{
    std::stringstream rider;
    rider << "rocfft-rider --length ";
    std::ostream_iterator<size_t> rider_iter(rider, " ");
    std::copy(plan->lengths.rbegin() + (3 - plan->rank), plan->lengths.rend(), rider_iter);
    rider << "-b " << plan->batch << " ";

    if(plan->placement == rocfft_placement_notinplace)
        rider << "-o ";

    rider << "-t " << plan->transformType << " ";

    if(plan->precision == rocfft_precision_double)
        rider << "--double ";
    rider << "--itype " << plan->desc.inArrayType << " ";
    rider << "--otype " << plan->desc.outArrayType << " ";
    rider << "--istride ";
    std::copy(
        plan->desc.inStrides.rbegin() + (3 - plan->rank), plan->desc.inStrides.rend(), rider_iter);
    rider << "--ostride ";
    std::copy(plan->desc.outStrides.rbegin() + (3 - plan->rank),
              plan->desc.outStrides.rend(),
              rider_iter);
    rider << "--idist " << plan->desc.inDist << " ";
    rider << "--odist " << plan->desc.outDist << " ";
    rider << "--ioffset ";
    std::copy(plan->desc.inOffset.begin(), plan->desc.inOffset.end(), rider_iter);
    rider << "--ooffset ";
    std::copy(plan->desc.outOffset.begin(), plan->desc.outOffset.end(), rider_iter);
    return rider.str();
}

rocfft_status rocfft_plan_create_internal(rocfft_plan                   plan,
                                          const rocfft_result_placement placement,
                                          const rocfft_transform_type   transform_type,
                                          const rocfft_precision        precision,
                                          const size_t                  dimensions,
                                          const size_t*                 lengths,
                                          const size_t                  number_of_transforms,
                                          const rocfft_plan_description description)
{
    // Check plan validity
    if(description != nullptr)
    {
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            // We need complex input data
            if(!((description->inArrayType == rocfft_array_type_complex_interleaved)
                 || (description->inArrayType == rocfft_array_type_complex_planar)))
                return rocfft_status_invalid_array_type;
            // We need complex output data
            if(!((description->outArrayType == rocfft_array_type_complex_interleaved)
                 || (description->outArrayType == rocfft_array_type_complex_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform requires that the input and output
            // format be identical
            if(placement == rocfft_placement_inplace)
            {
                if(description->inArrayType != description->outArrayType)
                    return rocfft_status_invalid_array_type;
            }
            break;
        case rocfft_transform_type_real_forward:
            // Input must be real
            if(description->inArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;
            // Output must be Hermitian
            if(!((description->outArrayType == rocfft_array_type_hermitian_interleaved)
                 || (description->outArrayType == rocfft_array_type_hermitian_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform must output to interleaved format
            if((placement == rocfft_placement_inplace)
               && (description->outArrayType != rocfft_array_type_hermitian_interleaved))
                return rocfft_status_invalid_array_type;
            break;
        case rocfft_transform_type_real_inverse:
            // Output must be real
            if(description->outArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;
            // Intput must be Hermitian
            if(!((description->inArrayType == rocfft_array_type_hermitian_interleaved)
                 || (description->inArrayType == rocfft_array_type_hermitian_planar)))
                return rocfft_status_invalid_array_type;
            // In-place transform must have interleaved input
            if((placement == rocfft_placement_inplace)
               && (description->inArrayType != rocfft_array_type_hermitian_interleaved))
                return rocfft_status_invalid_array_type;
            break;
        }
    }

    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;

    rocfft_plan p = plan;
    p->rank       = dimensions;
    p->lengths[0] = 1;
    p->lengths[1] = 1;
    p->lengths[2] = 1;
    for(size_t ilength = 0; ilength < dimensions; ++ilength)
    {
        p->lengths[ilength] = lengths[ilength];
    }
    p->batch          = number_of_transforms;
    p->placement      = placement;
    p->precision      = precision;
    p->base_type_size = (precision == rocfft_precision_double) ? sizeof(double) : sizeof(float);
    p->transformType  = transform_type;

    if(description != nullptr)
    {
        p->desc = *description;
    }
    else
    {
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            p->desc.inArrayType  = rocfft_array_type_complex_interleaved;
            p->desc.outArrayType = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            p->desc.inArrayType  = rocfft_array_type_real;
            p->desc.outArrayType = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            p->desc.inArrayType  = rocfft_array_type_hermitian_interleaved;
            p->desc.outArrayType = rocfft_array_type_real;
            break;
        }
    }

    // Set inStrides, if not specified
    if(p->desc.inStrides[0] == 0)
    {
        p->desc.inStrides[0] = 1;

        if((p->transformType == rocfft_transform_type_real_forward)
           && (p->placement == rocfft_placement_inplace))
        {
            // real-to-complex in-place
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.inStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.inDist == 0)
                p->desc.inDist = dist;
        }
        else if(p->transformType == rocfft_transform_type_real_inverse)
        {
            // complex-to-real
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.inStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.inDist == 0)
                p->desc.inDist = dist;
        }

        else
        {
            // Set the inStrides to deal with contiguous data
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.inStrides[i] = p->lengths[i - 1] * p->desc.inStrides[i - 1];
        }
    }

    // Set outStrides, if not specified
    if(p->desc.outStrides[0] == 0)
    {
        p->desc.outStrides[0] = 1;

        if((p->transformType == rocfft_transform_type_real_inverse)
           && (p->placement == rocfft_placement_inplace))
        {
            // complex-to-real in-place
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.outStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.outDist == 0)
                p->desc.outDist = dist;
        }
        else if(p->transformType == rocfft_transform_type_real_forward)
        {
            // real-co-complex
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.outStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.outDist == 0)
                p->desc.outDist = dist;
        }
        else
        {
            // Set the outStrides to deal with contiguous data
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.outStrides[i] = p->lengths[i - 1] * p->desc.outStrides[i - 1];
        }
    }

    // Set in and out Distances, if not specified
    if(p->desc.inDist == 0)
    {
        p->desc.inDist = p->lengths[p->rank - 1] * p->desc.inStrides[p->rank - 1];
    }
    if(p->desc.outDist == 0)
    {
        p->desc.outDist = p->lengths[p->rank - 1] * p->desc.outStrides[p->rank - 1];
    }

    log_bench(rocfft_rider_command(p));

    // construct the plan
    try
    {
        NodeMetaData rootPlanData(nullptr);

        rootPlanData.dimension = plan->rank;
        rootPlanData.batch     = plan->batch;
        for(size_t i = 0; i < plan->rank; i++)
        {
            rootPlanData.length.push_back(plan->lengths[i]);

            rootPlanData.inStride.push_back(plan->desc.inStrides[i]);
            rootPlanData.outStride.push_back(plan->desc.outStrides[i]);
        }
        rootPlanData.iDist = plan->desc.inDist;
        rootPlanData.oDist = plan->desc.outDist;

        rootPlanData.placement = plan->placement;
        rootPlanData.precision = plan->precision;
        if((plan->transformType == rocfft_transform_type_complex_forward)
           || (plan->transformType == rocfft_transform_type_real_forward))
            rootPlanData.direction = -1;
        else
            rootPlanData.direction = 1;

        rootPlanData.inArrayType  = plan->desc.inArrayType;
        rootPlanData.outArrayType = plan->desc.outArrayType;
        rootPlanData.rootIsC2C    = (rootPlanData.inArrayType != rocfft_array_type_real)
                                 && (rootPlanData.outArrayType != rocfft_array_type_real);

        ExecPlan& execPlan = plan->execPlan;
        int       deviceId = 0;
        if(hipGetDevice(&deviceId) != hipSuccess)
        {
            throw std::runtime_error("hipGetDevice failed.");
        }
        if(hipGetDeviceProperties(&(execPlan.deviceProp), deviceId) != hipSuccess)
        {
            throw std::runtime_error("hipGetDeviceProperties failed for deviceId "
                                     + std::to_string(deviceId));
        }
        rootPlanData.deviceProp = execPlan.deviceProp;
        execPlan.rootPlan       = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);

        std::copy(plan->lengths.begin(),
                  plan->lengths.begin() + plan->rank,
                  std::back_inserter(execPlan.iLength));
        std::copy(plan->lengths.begin(),
                  plan->lengths.begin() + plan->rank,
                  std::back_inserter(execPlan.oLength));

        if(plan->transformType == rocfft_transform_type_real_inverse)
        {
            execPlan.iLength.front() = execPlan.iLength.front() / 2 + 1;
            if(plan->placement == rocfft_placement_inplace)
                execPlan.oLength.front() = execPlan.iLength.front() * 2;
        }
        if(plan->transformType == rocfft_transform_type_real_forward)
        {
            execPlan.oLength.front() = execPlan.oLength.front() / 2 + 1;
            if(plan->placement == rocfft_placement_inplace)
                execPlan.iLength.front() = execPlan.oLength.front() * 2;
        }

        // set scaling on the root plan
        execPlan.rootPlan->scale_factor = p->desc.scale_factor;

        try
        {
            ProcessNode(execPlan); // TODO: more descriptions are needed
        }
        catch(std::exception&)
        {
            if(LOG_PLAN_ENABLED())
                PrintNode(*LogSingleton::GetInstance().GetPlanOS(), execPlan);
            throw;
        }

        // plan is compiled, no need to alloc twiddles + kargs etc
        if(rocfft_getenv("ROCFFT_INTERNAL_COMPILE_ONLY") == "1")
            return rocfft_status_success;

        if(!PlanPowX(execPlan)) // PlanPowX enqueues the GPU kernels by function
        {

            throw std::runtime_error("Unable to create execution plan.");
        }
        return rocfft_status_success;
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }
}

rocfft_status rocfft_plan_allocate(rocfft_plan* plan)
{
    *plan = new rocfft_plan_t;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                 const rocfft_result_placement placement,
                                 const rocfft_transform_type   transform_type,
                                 const rocfft_precision        precision,
                                 const size_t                  dimensions,
                                 const size_t*                 lengths,
                                 const size_t                  number_of_transforms,
                                 const rocfft_plan_description description)
{
    rocfft_plan_allocate(plan);

    size_t log_len[3] = {1, 1, 1};
    if(dimensions > 0)
        log_len[0] = lengths[0];
    if(dimensions > 1)
        log_len[1] = lengths[1];
    if(dimensions > 2)
        log_len[2] = lengths[2];

    log_trace(__func__,
              "plan",
              *plan,
              "placement",
              placement,
              "transform_type",
              transform_type,
              "precision",
              precision,
              "dimensions",
              dimensions,
              "lengths",
              std::make_pair(lengths, dimensions),
              "number_of_transforms",
              number_of_transforms,
              "description",
              description);

    return rocfft_plan_create_internal(*plan,
                                       placement,
                                       transform_type,
                                       precision,
                                       dimensions,
                                       lengths,
                                       number_of_transforms,
                                       description);
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
{
    delete plan;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
{
    if(!plan)
        return rocfft_status_failure;

    *size_in_bytes = plan->execPlan.WorkBufBytes(plan->base_type_size);
    log_trace(__func__, "plan", plan, "size_in_bytes ptr", size_in_bytes, "val", *size_in_bytes);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    rocfft_cout << std::endl;
    rocfft_cout << "precision: "
                << ((plan->precision == rocfft_precision_single) ? "single" : "double")
                << std::endl;

    rocfft_cout << "transform type: ";
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
        rocfft_cout << "complex forward";
        break;
    case rocfft_transform_type_complex_inverse:
        rocfft_cout << "complex inverse";
        break;
    case rocfft_transform_type_real_forward:
        rocfft_cout << "real forward";
        break;
    case rocfft_transform_type_real_inverse:
        rocfft_cout << "real inverse";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "result placement: ";
    switch(plan->placement)
    {
    case rocfft_placement_inplace:
        rocfft_cout << "in-place";
        break;
    case rocfft_placement_notinplace:
        rocfft_cout << "not in-place";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input array type: ";
    switch(plan->desc.inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "complex planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;

    rocfft_cout << "output array type: ";
    switch(plan->desc.outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        rocfft_cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        rocfft_cout << "comple planar";
        break;
    case rocfft_array_type_real:
        rocfft_cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        rocfft_cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        rocfft_cout << "hermitian planar";
        break;
    default:
        rocfft_cout << "unset";
        break;
    }
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "dimensions: " << plan->rank << std::endl;

    rocfft_cout << "lengths: " << plan->lengths[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->lengths[i];
    rocfft_cout << std::endl;
    rocfft_cout << "batch size: " << plan->batch << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input offset: " << plan->desc.inOffset[0];
    if((plan->desc.inArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.inOffset[1];
    rocfft_cout << std::endl;

    rocfft_cout << "output offset: " << plan->desc.outOffset[0];
    if((plan->desc.outArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar))
        rocfft_cout << ", " << plan->desc.outOffset[1];
    rocfft_cout << std::endl;
    rocfft_cout << std::endl;

    rocfft_cout << "input strides: " << plan->desc.inStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.inStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "output strides: " << plan->desc.outStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        rocfft_cout << ", " << plan->desc.outStrides[i];
    rocfft_cout << std::endl;

    rocfft_cout << "input distance: " << plan->desc.inDist << std::endl;
    rocfft_cout << "output distance: " << plan->desc.outDist << std::endl;
    rocfft_cout << std::endl;

    if(plan->desc.scale_factor != 1.0)
        rocfft_cout << "scale factor: " << plan->desc.scale_factor << std::endl;
    rocfft_cout << std::endl;

    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
{
    log_trace(__func__, "buf", static_cast<void*>(buf), "len", len);
    static constexpr char v[] = ROCFFT_VERSION_STRING;
    if(!buf)
        return rocfft_status_failure;
    if(len < sizeof(v))
        return rocfft_status_invalid_arg_value;
    memcpy(buf, v, sizeof(v));
    return rocfft_status_success;
}

void TreeNode::CopyNodeData(const TreeNode& srcNode)
{
    dimension       = srcNode.dimension;
    batch           = srcNode.batch;
    length          = srcNode.length;
    inStride        = srcNode.inStride;
    outStride       = srcNode.outStride;
    iDist           = srcNode.iDist;
    oDist           = srcNode.oDist;
    iOffset         = srcNode.iOffset;
    oOffset         = srcNode.oOffset;
    placement       = srcNode.placement;
    precision       = srcNode.precision;
    direction       = srcNode.direction;
    inArrayType     = srcNode.inArrayType;
    outArrayType    = srcNode.outArrayType;
    allowInplace    = srcNode.allowInplace;
    allowOutofplace = srcNode.allowOutofplace;
    deviceProp      = srcNode.deviceProp;

    // conditional
    large1D        = srcNode.large1D;
    largeTwd3Steps = srcNode.largeTwd3Steps;
    largeTwdBase   = srcNode.largeTwdBase;
    lengthBlue     = srcNode.lengthBlue;

    //
    obIn  = srcNode.obIn;
    obOut = srcNode.obOut;

    // NB:
    //   we don't copy these since it's possible we're copying
    //   a node to another one that is different scheme/derived class
    //   (for example, when doing fusion).
    //   The src ebtype could be incorrect in the new node
    //   same as lds_padding, lds_padding is initialized for each derived class
    //   so we don't copy this value, the target node already sets its value
    // ebtype      = srcNode.ebtype;
    // lds_padding = srcNode.lds_padding;
}

void TreeNode::CopyNodeData(const NodeMetaData& data)
{
    dimension    = data.dimension;
    batch        = data.batch;
    length       = data.length;
    inStride     = data.inStride;
    outStride    = data.outStride;
    iDist        = data.iDist;
    oDist        = data.oDist;
    iOffset      = data.iOffset;
    oOffset      = data.oOffset;
    placement    = data.placement;
    precision    = data.precision;
    direction    = data.direction;
    inArrayType  = data.inArrayType;
    outArrayType = data.outArrayType;
    deviceProp   = data.deviceProp;
}

bool TreeNode::isPlacementAllowed(rocfft_result_placement test_placement) const
{
    return (test_placement == rocfft_placement_inplace) ? allowInplace : allowOutofplace;
}

bool TreeNode::isOutBufAllowed(OperatingBuffer oB) const
{
    return (oB & allowedOutBuf) != 0;
}

bool TreeNode::isOutArrayTypeAllowed(rocfft_array_type oArrayType) const
{
    return allowedOutArrayTypes.count(oArrayType) > 0;
}

bool TreeNode::isRootNode() const
{
    return parent == nullptr;
}

bool TreeNode::isLeafNode() const
{
    return nodeType == NT_LEAF;
}

// Tree node builders

// NB:
// Don't assign inArrayType and outArrayType when building any tree node.
// That should be done in buffer assignment stage or
// TraverseTreeAssignPlacementsLogicA().

void TreeNode::RecursiveBuildTree()
{
    // Some-Common-Work...
    // We must follow the placement of RootPlan, so needs to make it explicit
    if(isRootNode())
    {
        allowInplace    = (placement == rocfft_placement_inplace);
        allowOutofplace = !allowInplace;
    }

    // overriden by each derived class
    BuildTree_internal();
}

void TreeNode::SanityCheck()
{
    // no un-defined node is allowed in the tree
    if(nodeType == NT_UNDEFINED)
        throw std::runtime_error("NT_UNDEFINED node");

    // Check buffer: all operating buffers have been assigned
    if(obIn == OB_UNINIT)
        throw std::runtime_error("obIn un-init");
    if(obOut == OB_UNINIT)
        throw std::runtime_error("obOut un-init");
    if((obIn == obOut) && (placement != rocfft_placement_inplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement inplace");
    if((obIn != obOut) && (placement != rocfft_placement_notinplace))
        throw std::runtime_error("[obIn,obOut] mismatch placement out-of-place");

    // Check length and stride and dimension:
    if(length.size() != inStride.size())
        throw std::runtime_error("length.size() mismatch inStride.size()");
    if(length.size() != outStride.size())
        throw std::runtime_error("length.size() mismatch outStride.size()");
    if(length.size() < dimension)
        throw std::runtime_error("not enough length[] for dimension");

    for(unsigned int i = 0; i < childNodes.size(); ++i)
    {
        // 1. Recursively check child
        childNodes[i]->SanityCheck();

        // 2. Assert that the kernel chain is connected
        // The Bluestein algorithm uses a separate buffer which is
        // convoluted with the input; the chain assumption isn't true here.
        // NB: we assume that the CS_KERNEL_CHIRP is first in the chain.
        if((i > 0) && (childNodes[i - 1]->scheme != CS_KERNEL_CHIRP))
        {
            if(childNodes[i - 1]->obOut != childNodes[i]->obIn)
                throw std::runtime_error("Sanity Check failed: buffers mismatch");
        }
    }
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_Z_XY()
{
    if(function_pool::has_SBRC_kernel(length[0], precision))
    {
        auto kernel
            = function_pool::get_kernel(fpkey(length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC));
        size_t bwd = kernel.transforms_per_block;
        if((length[1] >= bwd) && (length[2] >= bwd) && (length[1] * length[2] % bwd == 0))
            return true;
    }

    return false;
}

bool TreeNode::fuse_CS_KERNEL_TRANSPOSE_XY_Z()
{
    if(function_pool::has_SBRC_kernel(length[0], precision))
    {
        if((length[0] == length[2]) // limit to original "cubic" case
           && (length[0] / 2 + 1 == length[1])
           && !IsPo2(length[0]) // Need more investigation for diagonal transpose
        )
            return true;
    }
    return false;
}

bool TreeNode::fuse_CS_KERNEL_STK_R2C_TRANSPOSE()
{
    if(function_pool::has_SBRC_kernel(length[0], precision)) // kernel available
    {
        if((length[0] * 2 == length[1]) // limit to original "cubic" case
           && (length.size() == 2 || length[1] == length[2]) // 2D or 3D
        )
            return true;
    }
    return false;
}

// Compute the large twd decomposition base
// 2-Steps:
//  e.g., ( CeilPo2(10000)+ 1 ) / 2 , returns 7 : (2^7)*(2^7) = 16384 >= 10000
// 3-Steps:
//  e.g., ( CeilPo2(10000)+ 2 ) / 3 , returns 5 : (2^5)*(2^5)*(2^5) = 32768 >= 10000
void TreeNode::set_large_twd_base_steps(size_t largeTWDLength)
{
    // if is largeTwd3Steps, then 16^3 ~ 64^3, basically enough for 262144
    // else, base is 8 (2^8 = 256), could be 2-steps 256^2 = 65536, if exceed, then is 256^3, and so on..
    largeTwdBase = this->largeTwd3Steps
                       ? std::min((size_t)6, std::max((size_t)4, (CeilPo2(largeTWDLength) + 2) / 3))
                       : 8;

    // but we still want to know the exact steps we will loop
    ltwdSteps              = 0;
    size_t lenLargeTwdBase = pow(2, largeTwdBase);
    while(pow(lenLargeTwdBase, ltwdSteps) < largeTWDLength)
        ltwdSteps++;

    if(largeTwdBase == 8 && ltwdSteps > 3)
        throw std::runtime_error(
            "large-twd-base 8 could be 2,3 steps, but not supported for 4-steps yet");
    if(largeTwdBase < 8 && ltwdSteps != 3)
        throw std::runtime_error("large-twd-base for 4,5,6 must be 3-steps");
}

void TreeNode::ApplyFusion()
{
    // Do the final fusion after the buffer assign is completed
    for(auto& fuse : fuseShims)
    {
        // the flag was overwritten by execPlan (according to the arch for some specical cases)
        if(!fuse->IsSchemeFusable())
            continue;

        auto fused = fuse->FuseKernels();
        if(fused)
        {
            auto firstFusedNode = fuse->FirstFuseNode();
            this->RecursiveInsertNode(firstFusedNode, fused);

            // iterate from first to last to remove old nodes
            fuse->ForEachNode([=](TreeNode* node) { this->RecursiveRemoveNode(node); });
        }
    }

    for(auto& child : childNodes)
        child->ApplyFusion();
}

void TreeNode::RefreshTree()
{
    if(childNodes.empty())
        return;

    for(auto& child : childNodes)
        child->RefreshTree();

    auto& first = childNodes.front();
    auto& last  = childNodes.back();

    // the obIn of chirp is always set to S buffer, which is not really a input buffer
    // the parent's obIn should be the next node's obIn, instead of the S buffer
    // this avoid error when finding the callback-load-fn kernel
    this->obIn         = (first->scheme == CS_KERNEL_CHIRP) ? childNodes[1]->obIn : first->obIn;
    this->obOut        = last->obOut;
    this->placement    = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;
    this->inArrayType  = first->inArrayType;
    this->outArrayType = last->outArrayType;
}

void TreeNode::AssignParams()
{
    if((length.size() != inStride.size()) || (length.size() != outStride.size()))
        throw std::runtime_error("length size mismatches stride size");

    for(auto& child : childNodes)
    {
        child->inStride.clear();
        child->outStride.clear();
    }

    AssignParams_internal();
}

///////////////////////////////////////////////////////////////////////////////
/// Collect leaf node
void TreeNode::CollectLeaves(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    // re-collect after kernel fusion, so clear the previous collected elements
    if(isRootNode())
    {
        seq.clear();
        fuseSeq.clear();
    }

    if(nodeType == NT_LEAF)
    {
        seq.push_back(this);
    }
    else
    {
        for(auto& child : childNodes)
            child->CollectLeaves(seq, fuseSeq);

        for(auto& fuse : fuseShims)
            fuseSeq.push_back(fuse.get());
    }
}

// Important: Make sure the order of the fuse-shim is consistent with the execSeq
// This is essential for BackTracking in BufferAssignment
void OrderFuseShims(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq)
{
    std::vector<FuseShim*> reordered;
    for(auto node : seq)
    {
        for(size_t fuseID = 0; fuseID < fuseSeq.size(); ++fuseID)
        {
            if(node == fuseSeq[fuseID]->FirstFuseNode())
            {
                reordered.emplace_back(fuseSeq[fuseID]);
                break;
            }
        }
    }

    if(reordered.size() != fuseSeq.size())
        throw std::runtime_error("reorder fuse shim list error");

    fuseSeq.swap(reordered);
}

void CheckFuseShimForArch(ExecPlan& execPlan)
{
    // for gfx906...
    if(is_device_gcn_arch(execPlan.deviceProp, "gfx906"))
    {
        auto& fusions = execPlan.fuseShims;
        for(auto& fusion : fusions)
        {
            if(fusion->fuseType == FT_STOCKHAM_WITH_TRANS
               && fusion->FirstFuseNode()->length[0] == 168)
            {
                fusion->OverwriteFusableFlag(false);

                // remove it from the execPlan list
                fusions.erase(std::remove(fusions.begin(), fusions.end(), fusion), fusions.end());
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Calculate work memory requirements,
/// note this should be done after buffer assignment and deciding oDist
void TreeNode::DetermineBufferMemory(size_t& tmpBufSize,
                                     size_t& cmplxForRealSize,
                                     size_t& blueSize,
                                     size_t& chirpSize)
{
    if(nodeType == NT_LEAF)
    {
        auto outputPtrDiff = compute_ptrdiff(
            UseOutputLengthForPadding() ? GetOutputLength() : length, outStride, batch, oDist);

        if(scheme == CS_KERNEL_CHIRP)
            chirpSize = std::max(lengthBlue, chirpSize);

        if(obOut == OB_TEMP_BLUESTEIN)
            blueSize = std::max(outputPtrDiff, blueSize);

        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
            cmplxForRealSize = std::max(outputPtrDiff, cmplxForRealSize);

        if(obOut == OB_TEMP)
            tmpBufSize = std::max(outputPtrDiff, tmpBufSize);
    }

    for(auto& child : childNodes)
        child->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
}

void TreeNode::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << "\n" << indentStr.c_str() << "scheme: " << PrintScheme(scheme).c_str();
    os << "\n" << indentStr.c_str();
    os << "dimension: " << dimension;
    os << "\n" << indentStr.c_str();
    os << "batch: " << batch;
    os << "\n" << indentStr.c_str();
    os << "length: ";
    for(size_t i = 0; i < length.size(); i++)
    {
        os << length[i] << " ";
    }
    if(!outputLength.empty())
    {
        os << "\n" << indentStr.c_str();
        os << "outputLength: ";
        for(size_t i = 0; i < outputLength.size(); i++)
        {
            os << outputLength[i] << " ";
        }
    }

    os << "\n" << indentStr.c_str() << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << " ";

    os << "\n" << indentStr.c_str() << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << " ";

    if(iOffset)
    {
        os << "\n" << indentStr.c_str();
        os << "iOffset: " << iOffset;
    }
    if(oOffset)
    {
        os << "\n" << indentStr.c_str();
        os << "oOffset: " << oOffset;
    }

    os << "\n" << indentStr.c_str();
    os << "iDist: " << iDist;
    os << "\n" << indentStr.c_str();
    os << "oDist: " << oDist;

    os << "\n" << indentStr.c_str();
    os << "direction: " << direction;

    os << "\n" << indentStr.c_str();
    os << ((placement == rocfft_placement_inplace) ? "inplace" : "not inplace");

    os << "\n" << indentStr.c_str();

    os << ((precision == rocfft_precision_single) ? "single-precision" : "double-precision");

    os << std::endl << indentStr.c_str();
    os << "array type: ";
    switch(inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        os << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        os << "complex planar";
        break;
    case rocfft_array_type_real:
        os << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        os << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        os << "hermitian planar";
        break;
    default:
        os << "unset";
        break;
    }
    os << " -> ";
    switch(outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        os << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        os << "complex planar";
        break;
    case rocfft_array_type_real:
        os << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        os << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        os << "hermitian planar";
        break;
    default:
        os << "unset";
        break;
    }
    if(large1D)
    {
        os << "\n" << indentStr.c_str() << "large1D: " << large1D;
        os << "\n" << indentStr.c_str() << "largeTwdBase: " << largeTwdBase;
        os << "\n" << indentStr.c_str() << "largeTwdSteps: " << ltwdSteps;
    }
    if(twiddles)
    {
        os << "\n"
           << indentStr.c_str()
           << "twiddle table length: " << twiddles_size / sizeof_precision(precision);
    }
    if(twiddles_large)
    {
        os << "\n"
           << indentStr.c_str()
           << "large twiddle table length: " << twiddles_large_size / sizeof_precision(precision);
    }
    if(lengthBlue)
        os << "\n" << indentStr.c_str() << "lengthBlue: " << lengthBlue;
    os << "\n";
    switch(ebtype)
    {
    case EmbeddedType::NONE:
        break;
    case EmbeddedType::C2Real_PRE:
        os << indentStr.c_str() << "EmbeddedType: C2Real_PRE\n";
        break;
    case EmbeddedType::Real2C_POST:
        os << indentStr.c_str() << "EmbeddedType: Real2C_POST\n";
        break;
    }

    os << indentStr.c_str() << "SBRC_Trans_Type: " << PrintSBRCTransposeType(sbrcTranstype).c_str();
    os << "\n";

    switch(intrinsicMode)
    {
    case IntrinsicAccessType::DISABLE_BOTH:
        break;
    case IntrinsicAccessType::ENABLE_LOAD_ONLY:
        os << indentStr.c_str() << "Intrinsic Mode: LOAD_ONLY\n";
        break;
    case IntrinsicAccessType::ENABLE_BOTH:
        os << indentStr.c_str() << "Intrinsic Mode: LOAD_AND_STORE\n";
        break;
    }

    os << indentStr.c_str()
       << "Direct_to_from_Reg: " << PrintDirectToFromRegMode(dir2regMode).c_str();
    os << "\n";
    if(IsScalingEnabled())
        os << indentStr << "scale factor: " << scale_factor << "\n";

    os << indentStr << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n";
    os << indentStr << PrintOperatingBufferCode(obIn) << " -> " << PrintOperatingBufferCode(obOut)
       << "\n";
    for(const auto& c : comments)
    {
        os << indentStr << "comment: " << c << "\n";
    }

    if(childNodes.size())
    {
        for(auto& children_p : childNodes)
        {
            children_p->Print(os, indent + 1);
        }
    }
    std::cout << std::flush;
}

void TreeNode::RecursiveRemoveNode(TreeNode* node)
{
    for(auto& child : childNodes)
        child->RecursiveRemoveNode(node);
    childNodes.erase(std::remove_if(childNodes.begin(),
                                    childNodes.end(),
                                    [node](const std::unique_ptr<TreeNode>& child) {
                                        return child.get() == node;
                                    }),
                     childNodes.end());
}

void TreeNode::RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto found = std::find_if(
        childNodes.begin(), childNodes.end(), [pos](const std::unique_ptr<TreeNode>& child) {
            return child.get() == pos;
        });
    if(found != childNodes.end())
    {
        childNodes.insert(found, std::move(newNode));
    }
    else
    {
        for(auto& child : childNodes)
            child->RecursiveInsertNode(pos, newNode);
    }
}

TreeNode* TreeNode::GetPlanRoot()
{
    if(isRootNode())
        return this;

    return parent->GetPlanRoot();
}

TreeNode* TreeNode::GetFirstLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.front()->GetFirstLeaf();
}

TreeNode* TreeNode::GetLastLeaf()
{
    return (nodeType == NT_LEAF) ? this : childNodes.back()->GetLastLeaf();
}

// if this is in one of the 7 bluestein "componenet nodes", return that component node
TreeNode* TreeNode::GetBluesteinComponentParent()
{
    if(isRootNode())
        return nullptr;

    return (parent->scheme == CS_BLUESTEIN) ? this : parent->GetBluesteinComponentParent();
}

bool TreeNode::IsLastLeafNodeOfBluesteinComponent()
{
    // Note: blueComp is one of the 7 "component nodes", not bluestein node itself
    TreeNode* blueComp = GetBluesteinComponentParent();

    // if in bluestein tree, test if this is the last leaf, else return false
    // for example, if this the last leaf of the FFT-scheme ?
    return (blueComp) ? (blueComp->GetLastLeaf() == this) : false;
}

bool TreeNode::IsRootPlanC2CTransform()
{
    auto root = GetPlanRoot();
    return (root->inArrayType != rocfft_array_type_real)
           && (root->outArrayType != rocfft_array_type_real);
}

// remove a leaf node from the plan completely - plan optimization
// can remove unnecessary nodes to skip unnecessary work.
void RemoveNode(ExecPlan& execPlan, TreeNode* node)
{
    auto& execSeq = execPlan.execSeq;
    // remove it from the non-owning leaf nodes
    execSeq.erase(std::remove(execSeq.begin(), execSeq.end(), node), execSeq.end());

    // remove it from the tree structure
    execPlan.rootPlan->RecursiveRemoveNode(node);
}

// insert a leaf node to the plan, bot execSeq and tree - plan optimization
void InsertNode(ExecPlan& execPlan, TreeNode* pos, std::unique_ptr<TreeNode>& newNode)
{
    auto& execSeq = execPlan.execSeq;
    // insert it to execSeq, before pos
    execSeq.insert(std::find(execSeq.begin(), execSeq.end(), pos), newNode.get());

    // insert it before pos in the tree structure
    execPlan.rootPlan->RecursiveInsertNode(pos, newNode);
}

std::pair<TreeNode*, TreeNode*> ExecPlan::get_load_store_nodes() const
{
    const auto& seq = execSeq;

    // look forward for the first node that reads from input
    auto load_it = std::find_if(
        seq.begin(), seq.end(), [&](const TreeNode* n) { return n->obIn == rootPlan->obIn; });
    TreeNode* load = load_it == seq.end() ? nullptr : *load_it;

    // look backward for the last node that writes to output
    auto store_it = std::find_if(
        seq.rbegin(), seq.rend(), [&](const TreeNode* n) { return n->obOut == rootPlan->obOut; });
    TreeNode* store = store_it == seq.rend() ? nullptr : *store_it;

    assert(load && store);
    return std::make_pair(load, store);
}

void RuntimeCompilePlan(ExecPlan& execPlan)
{
    for(auto& node : execPlan.execSeq)
        node->compiledKernel = RTCKernel::runtime_compile(*node, execPlan.deviceProp.gcnArchName);
    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = execPlan.get_load_store_nodes();

    // callbacks are only possible on plans that don't use planar format for input or output
    bool need_callbacks = !array_type_is_planar(load_node->inArrayType)
                          && !array_type_is_planar(store_node->outArrayType);

    if(need_callbacks)
    {
        load_node->compiledKernelWithCallbacks
            = RTCKernel::runtime_compile(*load_node, execPlan.deviceProp.gcnArchName, true);

        if(store_node != load_node)
        {
            store_node->compiledKernelWithCallbacks
                = RTCKernel::runtime_compile(*store_node, execPlan.deviceProp.gcnArchName, true);
        }
    }

    // All of the compilations are started in parallel (via futures),
    // so resolve the futures now.  That ensures that the plan is
    // ready to run as soon as the caller gets the plan back.
    for(auto& node : execPlan.execSeq)
    {
        if(node->compiledKernel.valid())
            node->compiledKernel.get();
        if(node->compiledKernelWithCallbacks.valid())
            node->compiledKernelWithCallbacks.get();
    }
}

void ProcessNode(ExecPlan& execPlan)
{
    execPlan.rootPlan->RecursiveBuildTree();

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->dimension);
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    // collect leaf-nodes to execSeq and fuseShims
    execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);
    CheckFuseShimForArch(execPlan);
    OrderFuseShims(execPlan.execSeq, execPlan.fuseShims);

    // initialize root plan input/output location if not already done
    if(execPlan.rootPlan->obOut == OB_UNINIT)
        execPlan.rootPlan->obOut = OB_USER_OUT;
    if(execPlan.rootPlan->obIn == OB_UNINIT)
        execPlan.rootPlan->obIn
            = execPlan.rootPlan->placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;

    // guarantee min buffers but possible less fusions
    // execPlan.assignOptStrategy = rocfft_optimize_min_buffer;
    // starting from ABT
    execPlan.assignOptStrategy = rocfft_optimize_balance;
    // try to use all buffer to get most fusion
    //execPlan.assignOptStrategy = rocfft_optimize_max_fusion;
    AssignmentPolicy policy;
    policy.AssignBuffers(execPlan);

    // Apply the fusion after buffer, strides are assigned
    execPlan.rootPlan->ApplyFusion();

    // collect the execSeq since we've fused some kernels
    execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);

    // So we also need to update the whole tree including internal nodes
    // NB: The order matters: assign param -> fusion -> refresh internal node param
    execPlan.rootPlan->RefreshTree();

    // add padding if necessary
    policy.PadPlan(execPlan);

    // Collapse high dims on leaf nodes where possible
    execPlan.rootPlan->CollapseContiguousDims();

    // Check the buffer, param and tree integrity, Note we do this after fusion
    execPlan.rootPlan->SanityCheck();

    // get workBufSize..
    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

    // Set scale factor on final leaf node prior to RTC, since we
    // force RTC on Stockham kernels that need scaling
    //
    // But scaling happens before callback (if you want both), so we
    // need to get the last one that's not APPLY_CALLBACK
    auto scale_node
        = std::find_if(execPlan.execSeq.rbegin(), execPlan.execSeq.rend(), [](TreeNode* node) {
              return node->scheme != CS_KERNEL_APPLY_CALLBACK;
          });
    (*scale_node)->scale_factor = execPlan.rootPlan->scale_factor;

    // compile kernels for applicable nodes
    RuntimeCompilePlan(execPlan);

    execPlan.workBufSize      = tmpBufSize + cmplxForRealSize + blueSize + chirpSize;
    execPlan.tmpWorkBufSize   = tmpBufSize;
    execPlan.copyWorkBufSize  = cmplxForRealSize;
    execPlan.blueWorkBufSize  = blueSize;
    execPlan.chirpWorkBufSize = chirpSize;
}

void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan)
{
    os << "**********************************************************************"
          "*********"
       << std::endl;

    const size_t N = std::accumulate(execPlan.rootPlan->length.begin(),
                                     execPlan.rootPlan->length.end(),
                                     execPlan.rootPlan->batch,
                                     std::multiplies<size_t>());
    os << "Work buffer size: " << execPlan.workBufSize << std::endl;
    os << "Work buffer ratio: " << (double)execPlan.workBufSize / (double)N << std::endl;
    os << "Assignment strategy: " << PrintOptimizeStrategy(execPlan.assignOptStrategy) << std::endl;

    if(execPlan.execSeq.size() > 1)
    {
        std::vector<TreeNode*>::const_iterator prev_p = execPlan.execSeq.begin();
        std::vector<TreeNode*>::const_iterator curr_p = prev_p + 1;
        while(curr_p != execPlan.execSeq.end())
        {
            if((*curr_p)->placement == rocfft_placement_inplace)
            {
                for(size_t i = 0; i < (*curr_p)->inStride.size(); i++)
                {
                    const int infact  = (*curr_p)->inArrayType == rocfft_array_type_real ? 1 : 2;
                    const int outfact = (*curr_p)->outArrayType == rocfft_array_type_real ? 1 : 2;
                    if(outfact * (*curr_p)->inStride[i] != infact * (*curr_p)->outStride[i])
                    {
                        os << "error in stride assignments" << std::endl;
                    }
                    if(((*curr_p)->batch > 1)
                       && (outfact * (*curr_p)->iDist != infact * (*curr_p)->oDist))
                    {
                        os << "error in dist assignments" << std::endl;
                    }
                }
            }

            if((*prev_p)->scheme != CS_KERNEL_CHIRP && (*curr_p)->scheme != CS_KERNEL_CHIRP)
            {
                if((*prev_p)->obOut != (*curr_p)->obIn)
                {
                    os << "error in buffer assignments" << std::endl;
                }
            }

            prev_p = curr_p;
            ++curr_p;
        }
    }

    execPlan.rootPlan->Print(os, 0);

    os << "GridParams\n";
    for(const auto& gp : execPlan.gridParam)
    {
        os << "  b[" << gp.b_x << "," << gp.b_y << "," << gp.b_z << "] wgs[" << gp.wgs_x << ","
           << gp.wgs_y << "," << gp.wgs_z << "], dy_lds bytes " << gp.lds_bytes << "\n";
    }
    os << "End GridParams\n";

    os << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
