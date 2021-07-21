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
#include "function_pool.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "node_factory.h"
#include "private.h"
#include "radix_table.h"
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
        return Repo::GetRepo().CreatePlan(p);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
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
    dimension    = srcNode.dimension;
    batch        = srcNode.batch;
    length       = srcNode.length;
    inStride     = srcNode.inStride;
    outStride    = srcNode.outStride;
    iDist        = srcNode.iDist;
    oDist        = srcNode.oDist;
    iOffset      = srcNode.iOffset;
    oOffset      = srcNode.oOffset;
    placement    = srcNode.placement;
    precision    = srcNode.precision;
    direction    = srcNode.direction;
    inArrayType  = srcNode.inArrayType;
    outArrayType = srcNode.outArrayType;
    // conditional
    large1D        = srcNode.large1D;
    largeTwd3Steps = srcNode.largeTwd3Steps;
    largeTwdBase   = srcNode.largeTwdBase;
    lengthBlue     = srcNode.lengthBlue;
    //
    obIn  = srcNode.obIn;
    obOut = srcNode.obOut;
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

bool TreeNode::isRootNode()
{
    return parent == nullptr;
}

bool TreeNode::isLeafNode()
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
    // keep the flexibilty to add any other "common action" here
    // Some-Common-Work...

    // overriden by each derived class
    BuildTree_internal();
}

void TreeNode::SanityCheck()
{
    // no un-defined node is allowed in the tree
    if(nodeType == NT_UNDEFINED)
        throw std::runtime_error("NT_UNDEFINED node");

    // Check buffer: all operating buffers have been assigned
    assert(obIn != OB_UNINIT);
    assert(obOut != OB_UNINIT);

    // Check length and stride and dimension:
    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());
    assert(length.size() >= dimension);

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
        size_t bwd, wgs, lds;
        GetBlockComputeTable(length[0], bwd, wgs, lds);

        if((length[1] >= bwd) && (length[2] >= bwd) && (length[1] * length[2] % bwd == 0))
        {
            return true;
        }
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
        obOut = obOutBuf; // assign output

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

void TreeNode::AssignParams()
{
    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());

    AssignParams_internal();
}

///////////////////////////////////////////////////////////////////////////////
/// Collect leaf node and calculate work memory requirements

void TreeNode::TraverseTreeCollectLeafsLogicA(std::vector<TreeNode*>& seq,
                                              size_t&                 tmpBufSize,
                                              size_t&                 cmplxForRealSize,
                                              size_t&                 blueSize,
                                              size_t&                 chirpSize)
{
    if(childNodes.size() == 0)
    {
        if(scheme == CS_KERNEL_CHIRP)
        {
            chirpSize = std::max(2 * lengthBlue, chirpSize);
        }
        if(obOut == OB_TEMP_BLUESTEIN)
        {
            blueSize = std::max(oDist * batch, blueSize);
        }
        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
        {
            cmplxForRealSize = std::max(oDist * batch, cmplxForRealSize);
        }
        if(obOut == OB_TEMP)
        {
            tmpBufSize = std::max(oDist * batch, tmpBufSize);
        }
        seq.push_back(this);
    }
    else
    {
        for(auto children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
        {
            (*children_p)
                ->TraverseTreeCollectLeafsLogicA(
                    seq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
        }
    }
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

static size_t TransformsPerThreadblock(const size_t len, rocfft_precision precision)
{
    // look in function pool first to see if it knows
    auto k = function_pool::get_kernel(fpkey(len, precision));
    if(k.batches_per_block)
        return k.batches_per_block;
    // otherwise fall back to old generator
    size_t wgs = 0, numTrans = 0;
    DetermineSizes(len, wgs, numTrans);
    return numTrans;
}

void Optimize_Transpose_With_Strides(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    auto canOptimizeWithStrides = [](TreeNode* stockham) {
        // for 3D pow2 sizes, manipulating strides looks like it loses to
        // diagonal transpose
        if(IsPo2(stockham->length[0]) && stockham->length.size() >= 3)
            return false;
        size_t numTrans = TransformsPerThreadblock(stockham->length[0], stockham->precision);

        // ensure we are doing enough rows to coalesce properly. 4
        // seems to be enough for double-precision, whereas some
        // sizes that do 7 rows seem to be slower for single.
        size_t minRows = stockham->precision == rocfft_precision_single ? 8 : 4;
        return numTrans >= minRows;
    };

    // If this is a stockham fft that does multiple rows in one
    // kernel, followed by a transpose, adjust output strides to
    // replace the transpose.  Multiple rows will ensure that the
    // transposed column writes are coalesced.
    for(auto it = execSeq.begin(); it != execSeq.end(); ++it)
    {
        auto stockham = *it;
        if(stockham->scheme != CS_KERNEL_STOCKHAM && stockham->scheme != CS_KERNEL_2D_SINGLE)
            continue;

        // if we're a child of a plan that we know is doing TR
        // instead of RT, we don't want to combine the wrong pairs.
        auto parent = stockham->parent;
        if(parent != nullptr)
        {
            if(parent->scheme == CS_3D_TRTRTR
               || (parent->scheme == CS_REAL_3D_EVEN && parent->direction == 1)
               || (parent->scheme == CS_REAL_2D_EVEN && parent->direction == 1))
                continue;
        }

        if(!canOptimizeWithStrides(stockham))
            continue;

        auto next = it + 1;
        if(next == execSeq.end())
            break;
        auto transpose = *next;
        if(transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY
           && transpose->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
            continue;
        if(stockham->length == transpose->length
           && stockham->outStride == transpose->inStride
           // can't get rid of a transpose that also does twiddle multiplication
           && transpose->large1D == 0
           // This is a transpose, which must be out-of-place
           && EffectivePlacement(stockham->obIn, transpose->obOut, execPlan.rootPlan->placement)
                  == rocfft_placement_notinplace)
        {
            stockham->outStride = transpose->outStride;
            if(transpose->scheme == CS_KERNEL_TRANSPOSE)
            {
                std::swap(stockham->outStride[0], stockham->outStride[1]);
            }
            else if(transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
            {
                // make stockham write Z_XY-transposed outputs
                stockham->outStride[0] = transpose->outStride[2];
                stockham->outStride[1] = transpose->outStride[0];
                stockham->outStride[2] = transpose->outStride[1];
            }
            else
            {
                // make stockham write XY_Z-transposed outputs
                stockham->outStride[0] = transpose->outStride[1];
                stockham->outStride[1] = transpose->outStride[2];
                stockham->outStride[2] = transpose->outStride[0];
            }
            stockham->obOut        = transpose->obOut;
            stockham->outArrayType = transpose->outArrayType;
            stockham->placement    = rocfft_placement_notinplace;
            stockham->oDist        = transpose->oDist;

            stockham->comments.push_back("removed following " + PrintScheme(transpose->scheme)
                                         + " using strides");
            RemoveNode(execPlan, transpose);
        }
    }

    // In case something was removed, reset our local execSeq so we
    // don't try to combine an already-removed node with anything
    execSeq = execPlan.execSeq;

    // Similarly, if we have a transpose followed by a stockham
    // fft that does multiple rows in one kernel, adjust input
    // strides to replace the transpose.  Multiple rows will ensure
    // that the transposed column reads are coalesced.
    for(auto it = execSeq.begin(); it != execSeq.end(); ++it)
    {
        auto transpose = *it;
        // can't get rid of a transpose that also does twiddle multiplication
        if((transpose->scheme != CS_KERNEL_TRANSPOSE
            && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY
            && transpose->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
           || transpose->large1D != 0)
            continue;

        auto next = it + 1;
        if(next == execSeq.end())
            break;
        auto stockham = *next;

        if(stockham->scheme != CS_KERNEL_STOCKHAM)
            continue;

        if(!canOptimizeWithStrides(stockham))
            continue;

        // verify that the transpose output lengths match the FFT input lengths
        auto transposeOutputLengths = transpose->length;
        if(transpose->scheme == CS_KERNEL_TRANSPOSE)
            std::swap(transposeOutputLengths[0], transposeOutputLengths[1]);
        else if(transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
        {
            std::swap(transposeOutputLengths[0], transposeOutputLengths[1]);
            std::swap(transposeOutputLengths[1], transposeOutputLengths[2]);
        }
        else
        {
            // must be XY_Z
            std::swap(transposeOutputLengths[1], transposeOutputLengths[2]);
            std::swap(transposeOutputLengths[0], transposeOutputLengths[1]);
        }
        if(transposeOutputLengths != stockham->length)
            continue;

        // This is a transpose, which must be out-of-place
        if(EffectivePlacement(transpose->obIn, stockham->obOut, execPlan.rootPlan->placement)
           == rocfft_placement_notinplace)
        {
            stockham->inStride = transpose->inStride;
            if(transpose->scheme == CS_KERNEL_TRANSPOSE)
                std::swap(stockham->inStride[0], stockham->inStride[1]);
            else if(transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
            {
                // give stockham kernel Z_XY-transposed inputs and outputs
                stockham->inStride[0] = transpose->inStride[1];
                stockham->inStride[1] = transpose->inStride[0];
                stockham->inStride[2] = transpose->inStride[2];

                std::swap(stockham->outStride[1], stockham->outStride[2]);
                std::swap(stockham->length[1], stockham->length[2]);
            }
            else
            {
                // give stockham kernel XY_Z-transposed inputs
                stockham->inStride[0] = transpose->inStride[2];
                stockham->inStride[1] = transpose->inStride[0];
                stockham->inStride[2] = transpose->inStride[1];
            }

            stockham->obIn        = transpose->obIn;
            stockham->inArrayType = transpose->inArrayType;
            stockham->placement   = rocfft_placement_notinplace;
            stockham->iDist       = transpose->iDist;

            stockham->comments.push_back("removed preceding " + PrintScheme(transpose->scheme)
                                         + " using strides");
            RemoveNode(execPlan, transpose);
        }
    }
}

void Optimize_R_TO_CMPLX_TRANSPOSE(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    // combine R_TO_CMPLX and following transpose
    auto it = std::find_if(execSeq.begin(), execSeq.end(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_R_TO_CMPLX;
    });
    if(it != execSeq.end())
    {
        auto r_to_cmplx = *it;
        // check that next node is transpose
        ++it;
        if(it == execSeq.end())
            return;
        auto transpose = *it;

        // check placement
        if(EffectivePlacement(r_to_cmplx->obIn, transpose->obOut, execPlan.rootPlan->placement)
           != rocfft_placement_notinplace)
            return;

        if(transpose->scheme == CS_KERNEL_TRANSPOSE
           || transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
        {
            // NOTE: changing the scheme directly is dangerous, it's OK here because
            // CS_KERNEL_R_TO_CMPLX and CS_KERNEL_R_TO_CMPLX_TRANSPOSE is same class
            r_to_cmplx->obOut        = transpose->obOut;
            r_to_cmplx->scheme       = CS_KERNEL_R_TO_CMPLX_TRANSPOSE;
            r_to_cmplx->outArrayType = transpose->outArrayType;
            r_to_cmplx->placement    = EffectivePlacement(
                r_to_cmplx->obIn, r_to_cmplx->obOut, execPlan.rootPlan->placement);
            r_to_cmplx->outStride = transpose->outStride;
            r_to_cmplx->oDist     = transpose->oDist;
            r_to_cmplx->comments.push_back("fused " + PrintScheme(CS_KERNEL_R_TO_CMPLX)
                                           + " and following " + PrintScheme(transpose->scheme));
            RemoveNode(execPlan, transpose);
        }
    }
}

void Optimize_TRANSPOSE_CMPLX_TO_R(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    // combine CMPLX_TO_R with preceding transpose
    auto cmplx_to_r = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_CMPLX_TO_R;
    });
    // should be a stockham or bluestein kernel following the CMPLX_TO_R, so
    // CMPLX_TO_R can't be the last node either
    if(cmplx_to_r != execSeq.rend() && cmplx_to_r != execSeq.rbegin())
    {
        auto following = cmplx_to_r - 1;
        if((*following)->scheme == CS_KERNEL_CHIRP)
            following = following - 1; // skip CHIRP
        auto transpose = cmplx_to_r + 1;
        if(transpose != execSeq.rend()
           && ((*transpose)->scheme == CS_KERNEL_TRANSPOSE
               || (*transpose)->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
           && cmplx_to_r != execSeq.rbegin())
        {
            // but transpose needs to be out-of-place, so bring the
            // temp buffer in if the operation would be effectively
            // in-place.
            if(EffectivePlacement(
                   (*transpose)->obIn, (*cmplx_to_r)->obOut, execPlan.rootPlan->placement)
               == rocfft_placement_inplace)
            {
                auto followingScheme    = (*following)->scheme;
                bool canChangeNextInput = (followingScheme == CS_KERNEL_STOCKHAM
                                           || followingScheme == CS_KERNEL_2D_SINGLE
                                           || followingScheme == CS_KERNEL_STOCKHAM_BLOCK_CC
                                           || followingScheme == CS_KERNEL_PAD_MUL);
                // if the following node's input can't be changed (ex, a Transpose)
                // then we simply give up this fusion, otherwise it leads to a inplace Transpose
                // a case is: 80 84 312 -t 3, in which the whole following FFT is TRTRT
                if(!canChangeNextInput)
                    return;

                // connect the transpose operation to the following
                // transform and bring in the temp buffer
                (*cmplx_to_r)->obIn     = (*transpose)->obIn;
                (*cmplx_to_r)->obOut    = OB_TEMP;
                (*following)->obIn      = OB_TEMP;
                (*following)->placement = EffectivePlacement(
                    (*following)->obIn, (*following)->obOut, execPlan.rootPlan->placement);
            }
            else
            {
                // connect the transpose operation to the following
                // transform by default
                (*cmplx_to_r)->obIn  = (*transpose)->obIn;
                (*cmplx_to_r)->obOut = (*following)->obIn;
            }

            (*cmplx_to_r)->placement = rocfft_placement_notinplace;

            // NOTE: changing the scheme directly is dangerous, it's OK here because
            // CS_KERNEL_CMPLX_TO_R and CS_KERNEL_TRANSPOSE_CMPLX_TO_R is same class
            (*cmplx_to_r)->scheme      = CS_KERNEL_TRANSPOSE_CMPLX_TO_R;
            (*cmplx_to_r)->inArrayType = (*transpose)->inArrayType;
            (*cmplx_to_r)->inStride    = (*transpose)->inStride;
            (*cmplx_to_r)->length      = (*transpose)->length;
            (*cmplx_to_r)->iDist       = (*transpose)->iDist;
            (*cmplx_to_r)
                ->comments.push_back("fused " + PrintScheme(CS_KERNEL_CMPLX_TO_R)
                                     + " and preceding " + PrintScheme((*transpose)->scheme));
            RemoveNode(execPlan, *transpose);
        }
    }
}

// combine CS_KERNEL_STOCKHAM and following CS_KERNEL_TRANSPOSE_Z_XY to CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY if possible
void Optimize_STOCKHAM_TRANSPOSE_Z_XY(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    for(auto it = execSeq.rbegin() + 1; it != execSeq.rend(); ++it)
    {

        if((it != execSeq.rend()) && ((*it)->scheme == CS_KERNEL_STOCKHAM)
           && ((*(it - 1))->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
           && ((*(it - 1))->fuse_CS_KERNEL_TRANSPOSE_Z_XY()) // kernel available
           && ((*it)->obIn
               != ((*(it - 1))->obOut)) // "in-place" doesn't work without manipulating buffers
           && (!(((it + 1) != execSeq.rend())
                 && (*(it + 1))->scheme
                        == CS_KERNEL_TRANSPOSE_XY_Z)) // don't touch case XY_Z -> FFT -> Z_XY
        )
        {
            auto stockham  = it;
            auto transpose = it - 1;

            (*stockham)->obOut        = (*transpose)->obOut;
            (*stockham)->scheme       = CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY;
            (*stockham)->outArrayType = (*transpose)->outArrayType;
            (*stockham)->placement    = EffectivePlacement(
                (*stockham)->obIn, (*stockham)->obOut, execPlan.rootPlan->placement);
            // transpose must be out-of-place
            assert((*stockham)->placement == rocfft_placement_notinplace);
            (*stockham)->outStride = (*transpose)->outStride;
            (*stockham)->oDist     = (*transpose)->oDist;

            // Note: can't simply change the scheme, need to re-create the node (if different class)
            // TODO: any safer and elegant code design ?
            auto replaceNode = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
                                                                 (*stockham)->parent);
            replaceNode->CopyNodeData(*(*stockham));
            replaceNode->comments.push_back("fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                                            + " and following "
                                            + PrintScheme((*transpose)->scheme));

            InsertNode(execPlan, *stockham, replaceNode);
            RemoveNode(execPlan, *stockham);
            RemoveNode(execPlan, *transpose);
        }
    }
}

// combine one CS_KERNEL_STOCKHAM and following CS_KERNEL_TRANSPOSE_XY_Z in 3D complex to real
// NB: this should be replaced by combining CS_KERNEL_TRANSPOSE_XY_Z and the following
//     CS_KERNEL_STOCKHAM eventually, in which we might fuse 2 pairs of TR.
void Optimize_STOCKHAM_TRANSPOSE_XY_Z(ExecPlan& execPlan, std::vector<TreeNode*>& execSeq)
{
    auto trans_cmplx_to_r = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R;
    });
    if(trans_cmplx_to_r != execSeq.rend() && trans_cmplx_to_r != execSeq.rbegin())
    {
        auto stockham2  = trans_cmplx_to_r + 1;
        auto transpose2 = trans_cmplx_to_r + 2;
        auto stockham1  = trans_cmplx_to_r + 3;
        if(stockham1 != execSeq.rend() && (*stockham2)->scheme == CS_KERNEL_STOCKHAM
           && (*transpose2)->scheme == CS_KERNEL_TRANSPOSE_XY_Z
           && (*stockham1)->scheme == CS_KERNEL_STOCKHAM
           && (function_pool::has_SBRC_kernel((*transpose2)->length[0],
                                              (*transpose2)->precision)) // kernel available
           && ((*transpose2)->length[0]
               == (*transpose2)->length[2]) // limit to original "cubic" case
           && ((*transpose2)->length[0] / 2 + 1 == (*transpose2)->length[1])
           && (!IsPo2((*transpose2)->length[0]))) // Need more investigation for diagonal transpose
        {
            (*transpose2)->scheme       = CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            (*transpose2)->obIn         = (*stockham1)->obIn;
            (*transpose2)->obOut        = (*stockham2)->obOut;
            (*transpose2)->inArrayType  = (*stockham1)->inArrayType;
            (*transpose2)->outArrayType = (*stockham2)->outArrayType;

            (*stockham2)->obIn        = (*stockham2)->obOut;
            (*stockham2)->inArrayType = (*stockham2)->outArrayType;
            (*stockham2)->placement   = rocfft_placement_inplace;

            // Note: can't simply change the scheme, need to re-create the node (if different class)
            // TODO: any safer and elegant code design ?
            auto replaceNode = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
                                                                 (*transpose2)->parent);
            replaceNode->CopyNodeData(*(*transpose2));
            replaceNode->comments.push_back("fused " + PrintScheme(CS_KERNEL_TRANSPOSE_XY_Z)
                                            + " and preceding "
                                            + PrintScheme((*stockham1)->scheme));

            InsertNode(execPlan, *transpose2, replaceNode);
            RemoveNode(execPlan, *transpose2);
            RemoveNode(execPlan, *stockham1);
        }
    }
}

// combine CS_KERNEL_R_TO_CMPLX_TRANSPOSE with preceding CS_KERNEL_STOCKHAM
//         to CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
void Optimize_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY(ExecPlan&               execPlan,
                                                 std::vector<TreeNode*>& execSeq)
{
    auto r_to_cmplx_transpose = std::find_if(execSeq.rbegin(), execSeq.rend(), [](TreeNode* n) {
        return n->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE;
    });
    if(r_to_cmplx_transpose != execSeq.rend() && r_to_cmplx_transpose != execSeq.rbegin())
    {
        auto stockham = r_to_cmplx_transpose + 1;
        if((*stockham)->scheme == CS_KERNEL_STOCKHAM
           && (function_pool::has_SBRC_kernel((*stockham)->length[0],
                                              (*stockham)->precision)) // kernel available
           && ((*stockham)->length[0] * 2
               == (*stockham)->length[1]) // limit to original "cubic" case
           && (((*stockham)->length.size() == 2)
               || ((*stockham)->length[1] == (*stockham)->length[2]))
           // FIXME: need more investigate:
           //        "128 128 -t2 'ip'" and "100 100 -t 2 'ip'" will fail the validations.
           && (execPlan.rootPlan->placement != rocfft_placement_inplace))
        {
            (*stockham)->scheme       = CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY;
            (*stockham)->obOut        = (*r_to_cmplx_transpose)->obOut;
            (*stockham)->outArrayType = (*r_to_cmplx_transpose)->outArrayType;
            (*stockham)->placement    = rocfft_placement_notinplace;
            (*stockham)->outStride    = (*r_to_cmplx_transpose)->outStride;
            (*stockham)->oDist        = (*r_to_cmplx_transpose)->oDist;

            // NB:
            //    The generated CS_KERNEL_R_TO_CMPLX_TRANSPOSE kernel is in 3D fashion.
            //    We just need extend length and strides to make it work for 2D case.
            if((*stockham)->length.size() == 2)
            {
                (*stockham)->length.push_back(1);
                (*stockham)->inStride.push_back((*stockham)->inStride[1]);
                (*stockham)->outStride.push_back((*stockham)->outStride[1]);
            }

            // Note: can't simply change the scheme, need to re-create the node (if different class)
            // TODO: any safer and elegant code design ?
            auto replaceNode = NodeFactory::CreateNodeFromScheme(
                CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY, (*stockham)->parent);
            replaceNode->CopyNodeData(*(*stockham));
            replaceNode->comments.push_back("fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                                            + " and following "
                                            + PrintScheme((*r_to_cmplx_transpose)->scheme));

            InsertNode(execPlan, *stockham, replaceNode);
            RemoveNode(execPlan, *stockham);
            RemoveNode(execPlan, *r_to_cmplx_transpose);
        }
    }
}

static void OptimizePlan(ExecPlan& execPlan)
{
    auto passes = {
        Optimize_R_TO_CMPLX_TRANSPOSE,
        Optimize_TRANSPOSE_CMPLX_TO_R,
        Optimize_STOCKHAM_TRANSPOSE_Z_XY,
        Optimize_STOCKHAM_TRANSPOSE_XY_Z,
        Optimize_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,
        Optimize_Transpose_With_Strides,
    };
    for(auto pass : passes)
    {
        // Give each optimization pass its own copy of execSeq, so
        // that it can call RemoveNode without worrying about
        // invalidating iterators it might be using.  But it can also
        // modify nodes willy-nilly via the TreeNode*'s in the vector.
        auto localExecSeq = execPlan.execSeq;
        pass(execPlan, localExecSeq);
    }
}

void RuntimeCompilePlan(ExecPlan& execPlan)
{
    for(auto& node : execPlan.execSeq)
        node->compiledKernel = RTCKernel::runtime_compile(*node, execPlan.deviceProp.gcnArchName);
}

void ProcessNode(ExecPlan& execPlan)
{
    execPlan.rootPlan->RecursiveBuildTree();

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->dimension);
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    // initialize root plan input/output location if not already done
    if(execPlan.rootPlan->obOut == OB_UNINIT)
        execPlan.rootPlan->obOut = OB_USER_OUT;
    if(execPlan.rootPlan->obIn == OB_UNINIT)
        execPlan.rootPlan->obIn
            = execPlan.rootPlan->placement == rocfft_placement_inplace ? OB_USER_OUT : OB_USER_IN;
    // initialize traverse state so we can initialize obIn + obOut for all nodes
    TreeNode::TraverseState state(execPlan);
    OperatingBuffer         flipIn = OB_UNINIT, flipOut = OB_UNINIT, obOutBuf = OB_UNINIT;
    execPlan.rootPlan->AssignBuffers(state, flipIn, flipOut, obOutBuf);

    execPlan.rootPlan->TraverseTreeAssignPlacementsLogicA(execPlan.rootPlan->inArrayType,
                                                          execPlan.rootPlan->outArrayType);
    execPlan.rootPlan->AssignParams();

    // Check the buffer, param and tree integrity
    execPlan.rootPlan->SanityCheck();

    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->TraverseTreeCollectLeafsLogicA(
        execPlan.execSeq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);

    OptimizePlan(execPlan);

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
                    if(outfact * (*curr_p)->iDist != infact * (*curr_p)->oDist)
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
           << gp.tpb_y << "," << gp.tpb_z << "]\n";
    }
    os << "End GridParams\n";

    os << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
