// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include "assignment_policy.h"
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "node_factory.h"
#include "private.h"
#include "repo.h"
#include "rocfft-version.h"
#include "rocfft.h"
#include "rocfft_ostream.hpp"
#include "rtc.h"

#include <algorithm>
#include <assert.h>
#include <functional>
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

std::string PrintScheme(ComputeScheme cs)
{
    const std::map<ComputeScheme, const char*> ComputeSchemetoString
        = {{ENUMSTR(CS_NONE)},
           {ENUMSTR(CS_KERNEL_STOCKHAM)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_RC)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_BLOCK_CR)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_USING_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_HERM)},
           {ENUMSTR(CS_KERNEL_COPY_HERM_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_R)},

           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY)},
           {ENUMSTR(CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_EVEN)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_CMPLX_TO_R)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_CMPLX_TO_R)},
           {ENUMSTR(CS_REAL_2D_EVEN)},
           {ENUMSTR(CS_REAL_3D_EVEN)},
           {ENUMSTR(CS_KERNEL_APPLY_CALLBACK)},

           {ENUMSTR(CS_BLUESTEIN)},
           {ENUMSTR(CS_KERNEL_CHIRP)},
           {ENUMSTR(CS_KERNEL_PAD_MUL)},
           {ENUMSTR(CS_KERNEL_FFT_MUL)},
           {ENUMSTR(CS_KERNEL_RES_MUL)},

           {ENUMSTR(CS_L1D_TRTRT)},
           {ENUMSTR(CS_L1D_CC)},
           {ENUMSTR(CS_L1D_CRT)},

           {ENUMSTR(CS_2D_STRAIGHT)},
           {ENUMSTR(CS_2D_RTRT)},
           {ENUMSTR(CS_2D_RC)},
           {ENUMSTR(CS_KERNEL_2D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_2D_SINGLE)},

           {ENUMSTR(CS_3D_STRAIGHT)},
           {ENUMSTR(CS_3D_TRTRTR)},
           {ENUMSTR(CS_3D_RTRT)},
           {ENUMSTR(CS_3D_BLOCK_RC)},
           {ENUMSTR(CS_3D_BLOCK_CR)},
           {ENUMSTR(CS_3D_RC)},
           {ENUMSTR(CS_KERNEL_3D_STOCKHAM_BLOCK_CC)},
           {ENUMSTR(CS_KERNEL_3D_SINGLE)}};

    return ComputeSchemetoString.at(cs);
}

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

rocfft_status rocfft_plan_description_set_scale_float(rocfft_plan_description description,
                                                      const float             scale)
{
    description->scale = scale;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_scale_double(rocfft_plan_description description,
                                                       const double            scale)
{
    description->scale = scale;
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

    // size_t prodLength = 1;
    // for(size_t i = 0; i < (p->rank); i++)
    // {
    //     prodLength *= lengths[i];
    // }
    // if(!SupportedLength(prodLength))
    // {
    //     printf("This size %zu is not supported in rocFFT, will return;\n",
    //            prodLength);
    //     return rocfft_status_invalid_dimensions;
    // }

    // add this plan into repo, incurs computation, see repo.cpp
    try
    {
        Repo::GetRepo().CreatePlan(p);
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

    std::stringstream ss;
    ss << "./rocfft-rider"
       << " -t " << transform_type << " -x " << log_len[0] << " -y " << log_len[1] << " -z "
       << log_len[2] << " -b " << number_of_transforms;
    if(placement == rocfft_placement_notinplace)
        ss << " -o ";
    if(precision == rocfft_precision_double)
        ss << " --double ";
    if(description != NULL)
        ss << " --isX " << description->inStrides[0] << " --isY " << description->inStrides[1]
           << " --isZ " << description->inStrides[2] << " --osX " << description->outStrides[0]
           << " --osY " << description->outStrides[1] << " --osZ " << description->outStrides[2]
           << " --scale " << description->scale << " --iOff0 " << description->inOffset[0]
           << " --iOff1 " << description->inOffset[1] << " --oOff0 " << description->outOffset[0]
           << " --oOff1 " << description->outOffset[1] << " --inArrType "
           << description->inArrayType << " --outArrType " << description->outArrayType;

    log_bench(ss.str());

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
    log_trace(__func__, "plan", plan);
    // Remove itself from Repo first, and then delete itself
    Repo& repo = Repo::GetRepo();
    repo.DeletePlan(plan);
    if(plan != nullptr)
    {
        delete plan;
        plan = nullptr;
    }
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
{
    Repo&     repo     = Repo::GetRepo();
    ExecPlan* execPlan = repo.GetPlan(plan);
    if(!execPlan)
        return rocfft_status_failure;

    *size_in_bytes = execPlan->WorkBufBytes(plan->base_type_size);
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

    rocfft_cout << "scale: " << plan->desc.scale << std::endl;
    rocfft_cout << std::endl;

    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
{
    log_trace(__func__, "buf", buf, "len", len);
    static constexpr char v[] = ROCFFT_VERSION_STRING;
    if(!buf)
        return rocfft_status_failure;
    if(len < sizeof(v))
        return rocfft_status_invalid_arg_value;
    memcpy(buf, v, sizeof(v));
    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_repo_get_unique_plan_count(size_t* count)
{
    Repo& repo = Repo::GetRepo();
    *count     = repo.GetUniquePlanCount();
    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_repo_get_total_plan_count(size_t* count)
{
    Repo& repo = Repo::GetRepo();
    *count     = repo.GetTotalPlanCount();
    return rocfft_status_success;
}

void TreeNode::CopyNodeData(const TreeNode& srcNode)
{
    dimension        = srcNode.dimension;
    batch            = srcNode.batch;
    length           = srcNode.length;
    inStride         = srcNode.inStride;
    outStride        = srcNode.outStride;
    iDist            = srcNode.iDist;
    oDist            = srcNode.oDist;
    iOffset          = srcNode.iOffset;
    oOffset          = srcNode.oOffset;
    placement        = srcNode.placement;
    precision        = srcNode.precision;
    direction        = srcNode.direction;
    inArrayType      = srcNode.inArrayType;
    outArrayType     = srcNode.outArrayType;
    allowInplace     = srcNode.allowInplace;
    allowOutofplace  = srcNode.allowOutofplace;
    outputHasPadding = srcNode.outputHasPadding;

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
}

bool TreeNode::isPlacementAllowed(rocfft_result_placement test_placement)
{
    return (test_placement == rocfft_placement_inplace) ? allowInplace : allowOutofplace;
}

bool TreeNode::isOutBufAllowed(OperatingBuffer oB)
{
    return (oB & allowedOutBuf) != 0;
}

bool TreeNode::isOutArrayTypeAllowed(rocfft_array_type oArrayType)
{
    return allowedOutArrayTypes.count(oArrayType) > 0;
}

bool TreeNode::isRootNode()
{
    return parent == nullptr;
}

bool TreeNode::isLeafNode()
{
    return nodeType == NT_LEAF;
}

bool TreeNode::isInplacePreferable()
{
    // see comments in header
    return true;
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

    for(int i = 0; i < childNodes.size(); ++i)
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
        size_t bwd = kernel.block_width;
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
size_t TreeNode::large_twiddle_base(size_t length, bool use3Steps)
{
    if(use3Steps)
        // 16^3 ~ 64^3, basically enough for 262144
        return std::min((size_t)6, std::max((size_t)4, (CeilPo2(length) + 2) / 3));
    else
        // always return 8, 256^2, if exceed 65536, it becomes 256^3...
        return 8;
}

#if !GENERIC_BUF_ASSIGMENT
struct TreeNode::TraverseState
{
    TraverseState(const ExecPlan& execPlan)
        : rootPlan(execPlan.rootPlan.get())
    {
        TraverseFullSequence(rootPlan);
    }
    const TreeNode* rootPlan;
    // All nodes in the plan (leaf + non-leaf), ordered by how they
    // would be executed
    std::vector<const TreeNode*> fullSeq;

private:
    // recursively fill fullSeq
    void TraverseFullSequence(const TreeNode* node)
    {
        fullSeq.push_back(node);
        for(auto& child : node->childNodes)
            TraverseFullSequence(child.get());
    }
};

/// Buffer assignment
void TreeNode::SetInputBuffer(TraverseState& state)
{
    // find the given node in the full sequence
    auto it = std::find(state.fullSeq.begin(), state.fullSeq.end(), this);
    if(it == state.fullSeq.end())
    {
        // How did we get a node that wasn't in sequence?
        // Trigger an error in buffer assignment.
        assert(false);
        obIn = OB_UNINIT;
    }
    // Looking backwards from this node, find the closest leaf
    // node.  Exclude CS_KERNEL_CHIRP, since those effectively take
    // no inputs and output to a separate out-of-band buffer that
    // is not part of the chain.
    auto rev_begin = std::make_reverse_iterator(it);
    auto rev_end   = std::make_reverse_iterator(state.fullSeq.begin());
    auto prevLeaf  = std::find_if(rev_begin, rev_end, [](const TreeNode* n) {
        return n->childNodes.empty() && n->scheme != CS_KERNEL_CHIRP;
    });
    if(prevLeaf == rev_end)
    {
        // There is no earlier leaf node, so we should use the user's input for this node.
        obIn = state.rootPlan->obIn;
    }
    else
    {
        // There is an earlier leaf node, so we have to use its output as this node's input.
        obIn = (*prevLeaf)->obOut;
    }
}

static rocfft_result_placement EffectivePlacement(OperatingBuffer         obIn,
                                                  OperatingBuffer         obOut,
                                                  rocfft_result_placement rootPlacement)
{
    if(rootPlacement == rocfft_placement_inplace)
    {
        // in == out
        if((obIn == OB_USER_IN || obIn == OB_USER_OUT)
           && (obOut == OB_USER_IN || obOut == OB_USER_OUT))
            return rocfft_placement_inplace;
    }
    // otherwise just check if the buffers look different
    return obIn == obOut ? rocfft_placement_inplace : rocfft_placement_notinplace;
}

// Assign buffers, taking into account out-of-place transposes and
// padded buffers.
// NB: this recursive function modifies the parameters in the parent call.
void TreeNode::AssignBuffers(TraverseState&   state,
                             OperatingBuffer& flipIn,
                             OperatingBuffer& flipOut,
                             OperatingBuffer& obOutBuf)
{
    // Input buffer for 'this' is dictated by our traversal state.
    // Either we're the first node, which means we use the input the
    // user said to use, or we use the output of the last traversed
    // node.
    //
    // obIn might have already been set in special cases during plan
    // building, so only set it if it's not already set.
    if(obIn == OB_UNINIT)
        SetInputBuffer(state);

    // Set flipIn, flipOut, and oboutBuf for the root node.
    // Note: CS_REAL_TRANSFORM_USING_CMPLX and CS_BLUESTEIN would modify it
    //       Real-1D-Even, -2D-Even, -3D-Even would possibly modify it
    if(isRootNode())
    {
        flipIn   = OB_USER_OUT;
        flipOut  = OB_TEMP;
        obOutBuf = OB_USER_OUT;
    }

    AssignBuffers_internal(state, flipIn, flipOut, obOutBuf);

    if(obOut == OB_UNINIT)
    {
        obOut = obOutBuf; // assign output
    }

    TreeNode* rootnode = this;
    while(!rootnode->isRootNode())
    {
        rootnode = rootnode->parent;
    }

    // Verify that nodes that need to be out-of-place are indeed out-of-place.
    for(const auto& node : childNodes)
    {
        // TODO: this list can be expanded, though it may depend on other parameters.
        if(node->scheme == CS_KERNEL_TRANSPOSE || node->scheme == CS_KERNEL_TRANSPOSE_XY_Z
           || node->scheme == CS_KERNEL_TRANSPOSE_Z_XY
           || node->scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
           || node->scheme == CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
           || node->scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
           || node->scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        {
            //if(EffectivePlacement(node->obIn, node->obOut, placement) == rocfft_placement_inplace)
            if(node->obIn == node->obOut)
            {
                throw std::invalid_argument("Transpose must be out-of-place.");
            }
        }

        if(rootnode->placement == rocfft_placement_inplace)
        {
            if(node->obIn == OB_USER_IN || node->obOut == OB_USER_IN)
            {
                throw std::invalid_argument("In-place transforms cannot touch the input buffer; "
                                            "they are output-to-output.");
            }
        }

        if(rootnode->inArrayType != rocfft_array_type_real
           && rootnode->outArrayType != rocfft_array_type_real)
        {
            if(node->obOut == OB_USER_IN)
            {
                throw std::invalid_argument(
                    "Complex-to-complex transforms cannot write to the input buffer.");
            }
        }
    }

#if 0
    auto        here = this;
    auto        up   = parent;
    std::string tabs;
    while(up != nullptr && here != up)
    {
        here = up;
        up   = parent->parent;
        tabs += "\t";
    }
    rocfft_cout << "TraverseTreeAssignBuffersLogicA: " << PrintScheme(scheme) << ": "
                << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << "\n"
                << tabs << "\tobIn: " << PrintOperatingBuffer(obIn) << "\n"
                << tabs << "\tobOut: " << PrintOperatingBuffer(obOut) << "\n"
                << tabs << "\tflipIn: " << PrintOperatingBuffer(flipIn) << "\n"
                << tabs << "\tflipOut: " << PrintOperatingBuffer(flipOut) << "\n"
                << tabs << "\tobOutBuf: " << PrintOperatingBuffer(obOutBuf) << std::endl;
#endif
}

///////////////////////////////////////////////////////////////////////////////
/// Set placement variable and in/out array types, if not already set.
void TreeNode::TraverseTreeAssignPlacementsLogicA(const rocfft_array_type rootIn,
                                                  const rocfft_array_type rootOut)
{
    if(parent != nullptr)
    {
        placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;
        // if (this->scheme == CS_KERNEL_TRANSPOSE)
        // {
        //     rocfft_cout << " obIn " << obIn << ", obOut " << obOut << " rootIn " << rootIn
        //               << ", rootOut " << rootOut << " inArrayType " << inArrayType
        //               << ", outArrayType " << outArrayType << std::endl;
        // }

        if(inArrayType == rocfft_array_type_unset)
        {
            switch(obIn)
            {
            case OB_USER_IN:
                // NB:
                // There are some cases that 2D/3D even length r2c with
                // child node ***BLOCK_CC. The child node can not detect
                // the correct array type from its direct parent, which
                // has to get the info from root node.
                // On the other hand, some cases that 1D even length r2c
                // with children should use the array type from the parent
                // instead of root node.
                inArrayType = (rootIn == rocfft_array_type_complex_planar
                               || rootIn == rocfft_array_type_hermitian_planar)
                                  ? rootIn
                                  : parent->inArrayType;
                break;
            case OB_USER_OUT:
                inArrayType = (rootOut == rocfft_array_type_complex_planar
                               || rootOut == rocfft_array_type_hermitian_planar)
                                  ? rootOut
                                  : parent->outArrayType;
                break;
            case OB_TEMP:
                inArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                inArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_BLUESTEIN:
                inArrayType = rocfft_array_type_complex_interleaved;
                if(parent->iOffset != 0)
                    iOffset = parent->iOffset;
                break;
            default:
                inArrayType = rocfft_array_type_complex_interleaved;
            }
        }

        if(outArrayType == rocfft_array_type_unset)
        {
            switch(obOut)
            {
            case OB_USER_IN:
                outArrayType = (rootIn == rocfft_array_type_complex_planar
                                || rootIn == rocfft_array_type_hermitian_planar)
                                   ? rootIn
                                   : parent->inArrayType;
                break;
            case OB_USER_OUT:
                outArrayType = (rootOut == rocfft_array_type_complex_planar
                                || rootOut == rocfft_array_type_hermitian_planar)
                                   ? rootOut
                                   : parent->outArrayType;
                break;
            case OB_TEMP:
                outArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                outArrayType = rocfft_array_type_complex_interleaved;
                break;
            case OB_TEMP_BLUESTEIN:
                outArrayType = rocfft_array_type_complex_interleaved;
                if(parent->oOffset != 0)
                    oOffset = parent->oOffset;
                break;
            default:
                outArrayType = rocfft_array_type_complex_interleaved;
            }
        }
    }

    for(auto children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
    {
        (*children_p)->TraverseTreeAssignPlacementsLogicA(rootIn, rootOut);
    }
}
#endif

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
    std::string arch(execPlan.deviceProp.gcnArchName);
    // for gfx906...
    if(arch.compare(0, 6, "gfx906") == 0)
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
        if(scheme == CS_KERNEL_CHIRP)
            chirpSize = std::max(2 * lengthBlue, chirpSize);

        if(obOut == OB_TEMP_BLUESTEIN)
            blueSize = std::max(oDist * batch, blueSize);

        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
            cmplxForRealSize = std::max(oDist * batch, cmplxForRealSize);

        if(obOut == OB_TEMP)
            tmpBufSize = std::max(oDist * batch, tmpBufSize);
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

    if(outputHasPadding)
    {
        os << "\n" << indentStr.c_str();
        os << "outputHasPadding: " << outputHasPadding;
    }

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
#ifdef ROCFFT_RUNTIME_COMPILE
    for(auto& node : execPlan.execSeq)
        node->compiledKernel = RTCKernel::runtime_compile(*node, execPlan.deviceProp.gcnArchName);
    TreeNode* load_node             = nullptr;
    TreeNode* store_node            = nullptr;
    std::tie(load_node, store_node) = execPlan.get_load_store_nodes();
    load_node->compiledKernelWithCallbacks
        = RTCKernel::runtime_compile(*load_node, execPlan.deviceProp.gcnArchName, true);
    if(store_node != load_node)
    {
        store_node->compiledKernelWithCallbacks
            = RTCKernel::runtime_compile(*store_node, execPlan.deviceProp.gcnArchName, true);
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
#endif
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

#if GENERIC_BUF_ASSIGMENT
    // guarantee min buffers but possible less fusions
    // execPlan.assignOptStrategy = rocfft_optimize_min_buffer;
    // starting from ABT
    execPlan.assignOptStrategy = rocfft_optimize_balance;
    // try to use all buffer to get most fusion
    //execPlan.assignOptStrategy = rocfft_optimize_max_fusion;
    AssignmentPolicy policy;
    policy.AssignBuffers(execPlan);
#else
    // initialize traverse state so we can initialize obIn + obOut for all nodes
    TreeNode::TraverseState state(execPlan);
    OperatingBuffer         flipIn = OB_UNINIT, flipOut = OB_UNINIT, obOutBuf = OB_UNINIT;
    execPlan.rootPlan->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    execPlan.rootPlan->TraverseTreeAssignPlacementsLogicA(execPlan.rootPlan->inArrayType,
                                                          execPlan.rootPlan->outArrayType);
    execPlan.rootPlan->AssignParams();
#endif

    // Apply the fusion after buffer, strides are assigned
    execPlan.rootPlan->ApplyFusion();

    // collect the execSeq since we've fused some kernels
    execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);

    // So we also need to update the whole tree including internal nodes
    // NB: The order matters: assign param -> fusion -> refresh internal node param
    execPlan.rootPlan->RefreshTree();

    // Check the buffer, param and tree integrity, Note we do this after fusion
    execPlan.rootPlan->SanityCheck();

    // get workBufSize..
    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->DetermineBufferMemory(tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

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
            curr_p++;
        }
    }

    execPlan.rootPlan->Print(os, 0);

    os << "GridParams\n";
    for(const auto& gp : execPlan.gridParam)
    {
        os << "  b[" << gp.b_x << "," << gp.b_y << "," << gp.b_z << "] tpb[" << gp.tpb_x << ","
           << gp.tpb_y << "," << gp.tpb_z << "], dy_lds bytes " << gp.lds_bytes << "\n";
    }
    os << "End GridParams\n";

    os << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
