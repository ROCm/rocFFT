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
#include "logging.h"
#include "private.h"
#include "radix_table.h"
#include "repo.h"
#include "rocfft.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)

// clang-format off
#define VERSION_STRING (TO_STR(rocfft_version_major) "." \
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
           {ENUMSTR(CS_KERNEL_TRANSPOSE)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_XY_Z)},
           {ENUMSTR(CS_KERNEL_TRANSPOSE_Z_XY)},

           {ENUMSTR(CS_REAL_TRANSFORM_USING_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_HERM)},
           {ENUMSTR(CS_KERNEL_COPY_HERM_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_COPY_CMPLX_TO_R)},

           {ENUMSTR(CS_REAL_TRANSFORM_EVEN)},
           {ENUMSTR(CS_KERNEL_R_TO_CMPLX)},
           {ENUMSTR(CS_KERNEL_CMPLX_TO_R)},
           {ENUMSTR(CS_REAL_2D_EVEN)},

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
           {ENUMSTR(CS_3D_RTRT)},
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

std::string PrintArrayType(const rocfft_array_type x)
{
    const std::map<rocfft_array_type, const char*> array_type_to_string
        = {{ENUMSTR(rocfft_array_type_complex_interleaved)},
           {ENUMSTR(rocfft_array_type_complex_planar)},
           {ENUMSTR(rocfft_array_type_real)},
           {ENUMSTR(rocfft_array_type_hermitian_interleaved)},
           {ENUMSTR(rocfft_array_type_hermitian_planar)},
           {ENUMSTR(rocfft_array_type_unset)}};
    return array_type_to_string.at(x);
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
              in_offsets,
              "out_offsets",
              out_offsets,
              "in_strides_size",
              in_strides_size,
              "in_strides",
              in_strides,
              "in_distance",
              in_distance,
              "out_strides_size",
              out_strides_size,
              "out_strides",
              out_strides,
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
                                          const rocfft_plan_description description,
                                          const bool                    dry_run)
{
    // Check plan validity
    if(description != nullptr)
    {
        // We do not currently support planar formats.
        // TODO: remove these checks when complex planar format is enabled.
        if(description->inArrayType == rocfft_array_type_complex_planar
           || description->outArrayType == rocfft_array_type_complex_planar)
            return rocfft_status_invalid_array_type;
        if(description->inArrayType == rocfft_array_type_hermitian_planar
           || description->outArrayType == rocfft_array_type_hermitian_planar)
            return rocfft_status_invalid_array_type;

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

    if(!dry_run)
    {
        Repo& repo = Repo::GetRepo();
        repo.CreatePlan(p); // add this plan into repo, incurs computation, see repo.cpp
    }
    return rocfft_status_success;
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
              "placment",
              placement,
              "transform_type",
              transform_type,
              "precision",
              precision,
              "dimensions",
              dimensions,
              "lengths",
              log_len[0],
              log_len[1],
              log_len[2],
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
                                       description,
                                       false);
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    // Remove itself from Repo first, and then delete itself
    Repo& repo = Repo::GetRepo();
    repo.DeletePlan(plan);
    if(plan != nullptr)
        delete plan;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t* size_in_bytes)
{
    Repo&    repo = Repo::GetRepo();
    ExecPlan execPlan;
    repo.GetPlan(plan, execPlan);
    *size_in_bytes = execPlan.workBufSize * 2 * plan->base_type_size;
    log_trace(__func__, "plan", plan, "size_in_bytes ptr", size_in_bytes, "val", *size_in_bytes);
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
{
    log_trace(__func__, "plan", plan);
    std::cout << std::endl;
    std::cout << "precision: "
              << ((plan->precision == rocfft_precision_single) ? "single" : "double") << std::endl;

    std::cout << "transform type: ";
    switch(plan->transformType)
    {
    case rocfft_transform_type_complex_forward:
        std::cout << "complex forward";
        break;
    case rocfft_transform_type_complex_inverse:
        std::cout << "complex inverse";
        break;
    case rocfft_transform_type_real_forward:
        std::cout << "real forward";
        break;
    case rocfft_transform_type_real_inverse:
        std::cout << "real inverse";
        break;
    }
    std::cout << std::endl;

    std::cout << "result placement: ";
    switch(plan->placement)
    {
    case rocfft_placement_inplace:
        std::cout << "in-place";
        break;
    case rocfft_placement_notinplace:
        std::cout << "not in-place";
        break;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "input array type: ";
    switch(plan->desc.inArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "complex planar";
        break;
    case rocfft_array_type_real:
        std::cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "hermitian planar";
        break;
    default:
        std::cout << "unset";
        break;
    }
    std::cout << std::endl;

    std::cout << "output array type: ";
    switch(plan->desc.outArrayType)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "complex interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "comple planar";
        break;
    case rocfft_array_type_real:
        std::cout << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "hermitian interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "hermitian planar";
        break;
    default:
        std::cout << "unset";
        break;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "dimensions: " << plan->rank << std::endl;

    std::cout << "lengths: " << plan->lengths[0];
    for(size_t i = 1; i < plan->rank; i++)
        std::cout << ", " << plan->lengths[i];
    std::cout << std::endl;
    std::cout << "batch size: " << plan->batch << std::endl;
    std::cout << std::endl;

    std::cout << "input offset: " << plan->desc.inOffset[0];
    if((plan->desc.inArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.inArrayType == rocfft_array_type_hermitian_planar))
        std::cout << ", " << plan->desc.inOffset[1];
    std::cout << std::endl;

    std::cout << "output offset: " << plan->desc.outOffset[0];
    if((plan->desc.outArrayType == rocfft_array_type_complex_planar)
       || (plan->desc.outArrayType == rocfft_array_type_hermitian_planar))
        std::cout << ", " << plan->desc.outOffset[1];
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "input strides: " << plan->desc.inStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        std::cout << ", " << plan->desc.inStrides[i];
    std::cout << std::endl;

    std::cout << "output strides: " << plan->desc.outStrides[0];
    for(size_t i = 1; i < plan->rank; i++)
        std::cout << ", " << plan->desc.outStrides[i];
    std::cout << std::endl;

    std::cout << "input distance: " << plan->desc.inDist << std::endl;
    std::cout << "output distance: " << plan->desc.outDist << std::endl;
    std::cout << std::endl;

    std::cout << "scale: " << plan->desc.scale << std::endl;
    std::cout << std::endl;

    return rocfft_status_success;
}

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, const size_t len)
{
    log_trace(__func__, "buf", buf, "len", len);
    static constexpr char v[] = VERSION_STRING;
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

void TreeNode::build_real_even_1D()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    scheme = CS_REAL_TRANSFORM_EVEN;

    TreeNode* cfftPlan  = TreeNode::CreateNode(this);
    cfftPlan->dimension = dimension;
    cfftPlan->length    = length;
    cfftPlan->length[0] = cfftPlan->length[0] / 2;

    cfftPlan->inArrayType  = rocfft_array_type_complex_interleaved;
    cfftPlan->outArrayType = rocfft_array_type_complex_interleaved;
    cfftPlan->placement    = rocfft_placement_inplace;

    switch(direction)
    {
    case -1:
    {
        // complex-to-real transform: in-place complex transform then post-process

        // cfftPlan works in-place on the input buffer.
        // NB: the input buffer is real, but we treat it as complex
        cfftPlan->obOut = obIn;
        cfftPlan->RecursiveBuildTree();
        childNodes.push_back(cfftPlan);

        TreeNode* postPlan  = TreeNode::CreateNode(this);
        postPlan->scheme    = CS_KERNEL_R_TO_CMPLX;
        postPlan->dimension = 1;
        postPlan->length    = length;
        postPlan->length[0] /= 2;

        postPlan->inArrayType  = rocfft_array_type_complex_interleaved;
        postPlan->outArrayType = rocfft_array_type_hermitian_interleaved;

        childNodes.push_back(postPlan);
        break;
    }
    case 1:
    {
        // complex-to-real transform: pre-process followed by in-place complex transform

        TreeNode* prePlan  = TreeNode::CreateNode(this);
        prePlan->scheme    = CS_KERNEL_CMPLX_TO_R;
        prePlan->dimension = 1;
        prePlan->length    = length;
        prePlan->length[0] /= 2;

        prePlan->inArrayType  = rocfft_array_type_hermitian_interleaved;
        prePlan->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes.push_back(prePlan);

        // cfftPlan works in-place on the output buffer.
        // NB: the output buffer is real, but we treat it as complex
        cfftPlan->obIn = obOut;
        cfftPlan->RecursiveBuildTree();
        childNodes.push_back(cfftPlan);
        break;
    }
    default:
    {
        std::cerr << "invalid direction: plan creation failed!\n";
    }
    }
}

void TreeNode::build_real_even_2D()
{
    // Fastest moving dimension must be even:
    assert(length[0] % 2 == 0);

    assert(inArrayType == rocfft_array_type_real || outArrayType == rocfft_array_type_real);

    const auto complex_type = (inArrayType == rocfft_array_type_real) ? outArrayType : inArrayType;

    scheme = CS_REAL_2D_EVEN;

    const bool forward = inArrayType == rocfft_array_type_real;

    if(forward)
    {
        // RTRT
        {
            // first row fft
            TreeNode* row1Plan = TreeNode::CreateNode(this);
            row1Plan->length.push_back(length[0]);
            row1Plan->dimension = 1;
            row1Plan->length.push_back(length[1]);
            row1Plan->inArrayType  = inArrayType;
            row1Plan->outArrayType = complex_type;
            for(size_t index = 2; index < length.size(); index++)
            {
                row1Plan->length.push_back(length[index]);
            }
            row1Plan->build_real_even_1D();
            childNodes.push_back(row1Plan);
        }

        {
            // first transpose
            TreeNode* trans1Plan = TreeNode::CreateNode(this);
            trans1Plan->length.push_back(length[0] / 2 + 1);
            trans1Plan->length.push_back(length[1]);
            trans1Plan->scheme       = CS_KERNEL_TRANSPOSE;
            trans1Plan->dimension    = 2;
            trans1Plan->inArrayType  = complex_type;
            trans1Plan->outArrayType = complex_type;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans1Plan->length.push_back(length[index]);
            }
            childNodes.push_back(trans1Plan);
        }

        {
            // second row fft
            TreeNode* row2Plan = TreeNode::CreateNode(this);
            row2Plan->length.push_back(length[1]);
            row2Plan->dimension = 1;
            row2Plan->length.push_back(length[0] / 2 + 1);
            row2Plan->inArrayType  = complex_type;
            row2Plan->outArrayType = outArrayType;
            for(size_t index = 2; index < length.size(); index++)
            {
                row2Plan->length.push_back(length[index]);
            }
            row2Plan->RecursiveBuildTree();
            childNodes.push_back(row2Plan);
        }

        {
            // second transpose
            TreeNode* trans2Plan = TreeNode::CreateNode(this);
            trans2Plan->length.push_back(length[1]);
            trans2Plan->length.push_back(length[0] / 2 + 1);
            trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
            trans2Plan->dimension = 2;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans2Plan->length.push_back(length[index]);
            }

            childNodes.push_back(trans2Plan);
        }
    }
    else
    {
        // TRTR

        // first transpose
        {
            TreeNode* trans1Plan = TreeNode::CreateNode(this);
            trans1Plan->length.push_back(length[0] / 2 + 1);
            trans1Plan->length.push_back(length[1]);
            trans1Plan->scheme       = CS_KERNEL_TRANSPOSE;
            trans1Plan->dimension    = 2;
            trans1Plan->inArrayType  = inArrayType;
            trans1Plan->outArrayType = complex_type;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans1Plan->length.push_back(length[index]);
            }
            childNodes.push_back(trans1Plan);
        }

        // c2c row transform
        {
            TreeNode* c2cPlan  = TreeNode::CreateNode(this);
            c2cPlan->dimension = 1;
            c2cPlan->length.push_back(length[1]);
            c2cPlan->length.push_back(length[0] / 2 + 1);
            c2cPlan->inArrayType  = complex_type;
            c2cPlan->outArrayType = complex_type;
            for(size_t index = 2; index < length.size(); index++)
            {
                c2cPlan->length.push_back(length[index]);
            }
            c2cPlan->RecursiveBuildTree();
            childNodes.push_back(c2cPlan);
        }

        // second transpose
        {
            TreeNode* trans2plan = TreeNode::CreateNode(this);
            trans2plan->length.push_back(length[1]);
            trans2plan->length.push_back(length[0] / 2 + 1);
            trans2plan->scheme       = CS_KERNEL_TRANSPOSE;
            trans2plan->dimension    = 2;
            trans2plan->inArrayType  = complex_type;
            trans2plan->outArrayType = complex_type;
            for(size_t index = 2; index < length.size(); index++)
            {
                trans2plan->length.push_back(length[index]);
            }
            childNodes.push_back(trans2plan);
        }

        // c2r row transform
        {
            TreeNode* c2rPlan = TreeNode::CreateNode(this);
            c2rPlan->length.push_back(length[0]);
            c2rPlan->length.push_back(length[1]);
            c2rPlan->dimension    = 1;
            c2rPlan->inArrayType  = complex_type;
            c2rPlan->outArrayType = outArrayType;
            for(size_t index = 2; index < length.size(); index++)
            {
                c2rPlan->length.push_back(length[index]);
            }
            c2rPlan->build_real_even_1D();
            childNodes.push_back(c2rPlan);
        }
    }
}

void TreeNode::build_real_embed()
{
    // Embed the data into a full-length complex array, perform a
    // complex transform, and then extract the relevant output.

    scheme = CS_REAL_TRANSFORM_USING_CMPLX;

    TreeNode* copyHeadPlan = TreeNode::CreateNode(this);

    // head copy plan
    copyHeadPlan->dimension = dimension;
    copyHeadPlan->length    = length;
    copyHeadPlan->scheme    = (inArrayType == rocfft_array_type_real) ? CS_KERNEL_COPY_R_TO_CMPLX
                                                                   : CS_KERNEL_COPY_HERM_TO_CMPLX;
    childNodes.push_back(copyHeadPlan);

    // complex fft
    TreeNode* fftPlan = TreeNode::CreateNode(this);

    fftPlan->dimension = dimension;
    fftPlan->length    = length;

    fftPlan->RecursiveBuildTree();
    childNodes.push_back(fftPlan);

    // tail copy plan
    TreeNode* copyTailPlan = TreeNode::CreateNode(this);

    copyTailPlan->dimension = dimension;
    copyTailPlan->length    = length;
    copyTailPlan->scheme    = (inArrayType == rocfft_array_type_real) ? CS_KERNEL_COPY_CMPLX_TO_HERM
                                                                   : CS_KERNEL_COPY_CMPLX_TO_R;
    childNodes.push_back(copyTailPlan);
}

void TreeNode::build_real()
{
    if(length[0] % 2 == 0 && inStride[0] == 1 && outStride[0] == 1)
    {
        switch(dimension)
        {
        case 1:
            build_real_even_1D();
            break;
        case 2:
            build_real_even_2D();
            break;
        case 3:
            // TODO: implement
        default:
            build_real_embed();
        }
    }
    else
    {
        build_real_embed();
    }
}

size_t TreeNode::div1DNoPo2(const size_t length0)
{
    const size_t supported[]
        = {4096, 4050, 4000, 3888, 3840, 3750, 3645, 3600, 3456, 3375, 3240, 3200, 3125, 3072,
           3000, 2916, 2880, 2700, 2592, 2560, 2500, 2430, 2400, 2304, 2250, 2187, 2160, 2048,
           2025, 2000, 1944, 1920, 1875, 1800, 1728, 1620, 1600, 1536, 1500, 1458, 1440, 1350,
           1296, 1280, 1250, 1215, 1200, 1152, 1125, 1080, 1024, 1000, 972,  960,  900,  864,
           810,  800,  768,  750,  729,  720,  675,  648,  640,  625,  600,  576,  540,  512,
           500,  486,  480,  450,  432,  405,  400,  384,  375,  360,  324,  320,  300,  288,
           270,  256,  250,  243,  240,  225,  216,  200,  192,  180,  162,  160,  150,  144,
           135,  128,  125,  120,  108,  100,  96,   90,   81,   80,   75,   72,   64,   60,
           54,   50,   48,   45,   40,   36,   32,   30,   27,   25,   24,   20,   18,   16,
           15,   12,   10,   9,    8,    6,    5,    4,    3,    2,    1};

    size_t idx;
    if(length0 > (Large1DThreshold(precision) * Large1DThreshold(precision)))
    {
        idx = 0;
        while(supported[idx] != Large1DThreshold(precision))
        {
            idx++;
        }
        while(length0 % supported[idx] != 0)
        {
            idx++;
        }
    }
    else
    {
        // logic tries to break into as squarish matrix as possible
        size_t sqr = (size_t)sqrt(length0);
        idx        = sizeof(supported) / sizeof(supported[0]) - 1;
        while(supported[idx] < sqr)
        {
            idx--;
        }
        while(length0 % supported[idx] != 0)
        {
            idx++;
        }
    }
    assert(idx < sizeof(supported) / sizeof(supported[0]));
    return length0 / supported[idx];
}

void TreeNode::build_1D()
{
    // Build a node for a 1D FFT

    if(!SupportedLength(length[0]))
    {
        build_1DBluestein();
        return;
    }

    if(length[0] <= Large1DThreshold(precision)) // single kernel algorithm
    {
        scheme = CS_KERNEL_STOCKHAM;
        return;
    }

    size_t divLength1 = 1;

    if(IsPo2(length[0])) // multiple kernels involving transpose
    {
        if(length[0] <= 262144 / PrecisionWidth(precision))
        {
            // Enable block compute under these conditions
            if(1 == PrecisionWidth(precision))
            {
                divLength1 = Pow2Lengths1Single.at(length[0]);
            }
            else
            {
                divLength1 = Pow2Lengths1Double.at(length[0]);
            }
            scheme = (length[0] <= 65536 / PrecisionWidth(precision)) ? CS_L1D_CC : CS_L1D_CRT;
        }
        else
        {
            if(length[0] > (Large1DThreshold(precision) * Large1DThreshold(precision)))
            {
                divLength1 = length[0] / Large1DThreshold(precision);
            }
            else
            {
                size_t in_x = 0;
                size_t len  = length[0];
                while(len != 1)
                {
                    len >>= 1;
                    in_x++;
                }
                in_x /= 2;
                divLength1 = (size_t)1 << in_x;
            }
            scheme = CS_L1D_TRTRT;
        }
    }
    else // if not Pow2
    {
        divLength1 = div1DNoPo2(length[0]);
        scheme     = CS_L1D_TRTRT;
    }

    size_t divLength0 = length[0] / divLength1;

    switch(scheme)
    {
    case CS_L1D_TRTRT:
        build_1DCS_L1D_TRTRT(divLength0, divLength1);
        break;
    case CS_L1D_CC:
        build_1DCS_L1D_CC(divLength0, divLength1);
        break;
    case CS_L1D_CRT:
        build_1DCS_L1D_CRT(divLength0, divLength1);
        break;
    default:
        assert(false);
    }
}

void TreeNode::build_1DBluestein()
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.

    scheme     = CS_BLUESTEIN;
    lengthBlue = FindBlue(length[0]);

    TreeNode* chirpPlan = TreeNode::CreateNode(this);

    chirpPlan->scheme    = CS_KERNEL_CHIRP;
    chirpPlan->dimension = 1;
    chirpPlan->length.push_back(length[0]);
    chirpPlan->lengthBlue = lengthBlue;
    chirpPlan->direction  = direction;
    chirpPlan->batch      = 1;
    chirpPlan->large1D    = 2 * length[0];
    childNodes.push_back(chirpPlan);

    TreeNode* padmulPlan = TreeNode::CreateNode(this);

    padmulPlan->dimension  = 1;
    padmulPlan->length     = length;
    padmulPlan->lengthBlue = lengthBlue;
    padmulPlan->scheme     = CS_KERNEL_PAD_MUL;
    childNodes.push_back(padmulPlan);

    TreeNode* fftiPlan = TreeNode::CreateNode(this);

    fftiPlan->dimension = 1;
    fftiPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftiPlan->length.push_back(length[index]);
    }

    fftiPlan->iOffset = 2 * lengthBlue;
    fftiPlan->oOffset = 2 * lengthBlue;
    fftiPlan->scheme  = CS_KERNEL_STOCKHAM;
    fftiPlan->RecursiveBuildTree();
    childNodes.push_back(fftiPlan);

    TreeNode* fftcPlan = TreeNode::CreateNode(this);

    fftcPlan->dimension = 1;
    fftcPlan->length.push_back(lengthBlue);
    fftcPlan->scheme  = CS_KERNEL_STOCKHAM;
    fftcPlan->batch   = 1;
    fftcPlan->iOffset = lengthBlue;
    fftcPlan->oOffset = lengthBlue;
    fftcPlan->RecursiveBuildTree();
    childNodes.push_back(fftcPlan);

    TreeNode* fftmulPlan = TreeNode::CreateNode(this);

    fftmulPlan->dimension = 1;
    fftmulPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftmulPlan->length.push_back(length[index]);
    }

    fftmulPlan->lengthBlue = lengthBlue;
    fftmulPlan->scheme     = CS_KERNEL_FFT_MUL;
    childNodes.push_back(fftmulPlan);

    TreeNode* fftrPlan = TreeNode::CreateNode(this);

    fftrPlan->dimension = 1;
    fftrPlan->length.push_back(lengthBlue);
    for(size_t index = 1; index < length.size(); index++)
    {
        fftrPlan->length.push_back(length[index]);
    }

    fftrPlan->scheme    = CS_KERNEL_STOCKHAM;
    fftrPlan->direction = -direction;
    fftrPlan->iOffset   = 2 * lengthBlue;
    fftrPlan->oOffset   = 2 * lengthBlue;
    fftrPlan->RecursiveBuildTree();
    childNodes.push_back(fftrPlan);

    TreeNode* resmulPlan = TreeNode::CreateNode(this);

    resmulPlan->dimension  = 1;
    resmulPlan->length     = length;
    resmulPlan->lengthBlue = lengthBlue;
    resmulPlan->scheme     = CS_KERNEL_RES_MUL;
    childNodes.push_back(resmulPlan);
}

void build_1D_Compute_divLengthPow2() {}

void TreeNode::build_1DCS_L1D_TRTRT(const size_t divLength0, const size_t divLength1)
{
    // first transpose
    TreeNode* trans1Plan = TreeNode::CreateNode(this);

    trans1Plan->length.push_back(divLength0);
    trans1Plan->length.push_back(divLength1);

    trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans1Plan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }

    childNodes.push_back(trans1Plan);

    // first row fft
    TreeNode* row1Plan = TreeNode::CreateNode(this);

    // twiddling is done in row2 or transpose2
    row1Plan->large1D = 0;

    row1Plan->length.push_back(divLength1);
    row1Plan->length.push_back(divLength0);

    row1Plan->scheme    = CS_KERNEL_STOCKHAM;
    row1Plan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row1Plan->length.push_back(length[index]);
    }

    row1Plan->RecursiveBuildTree();
    childNodes.push_back(row1Plan);

    // second transpose
    TreeNode* trans2Plan = TreeNode::CreateNode(this);

    trans2Plan->length.push_back(divLength1);
    trans2Plan->length.push_back(divLength0);

    trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans2Plan->dimension = 2;

    trans2Plan->large1D = length[0];

    for(size_t index = 1; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }

    childNodes.push_back(trans2Plan);

    // second row fft
    TreeNode* row2Plan = TreeNode::CreateNode(this);

    row2Plan->length.push_back(divLength0);
    row2Plan->length.push_back(divLength1);

    row2Plan->scheme    = CS_KERNEL_STOCKHAM;
    row2Plan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2Plan->length.push_back(length[index]);
    }

    // algorithm is set up in a way that row2 does not recurse
    assert(divLength0 <= Large1DThreshold(this->precision));

    childNodes.push_back(row2Plan);

    // third transpose
    TreeNode* trans3Plan = TreeNode::CreateNode(this);

    trans3Plan->length.push_back(divLength0);
    trans3Plan->length.push_back(divLength1);

    trans3Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans3Plan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        trans3Plan->length.push_back(length[index]);
    }

    childNodes.push_back(trans3Plan);
}

void TreeNode::build_1DCS_L1D_CC(const size_t divLength0, const size_t divLength1)
{
    // first plan, column-to-column
    TreeNode* col2colPlan = TreeNode::CreateNode(this);

    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];

    col2colPlan->length.push_back(divLength1);
    col2colPlan->length.push_back(divLength0);

    col2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_CC;
    col2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    childNodes.push_back(col2colPlan);

    // second plan, row-to-column
    TreeNode* row2colPlan = TreeNode::CreateNode(this);

    row2colPlan->length.push_back(divLength0);
    row2colPlan->length.push_back(divLength1);

    row2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_RC;
    row2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2colPlan->length.push_back(length[index]);
    }

    childNodes.push_back(row2colPlan);
}

void TreeNode::build_1DCS_L1D_CRT(const size_t divLength0, const size_t divLength1)
{
    // first plan, column-to-column
    TreeNode* col2colPlan = TreeNode::CreateNode(this);

    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D = length[0];

    col2colPlan->length.push_back(divLength1);
    col2colPlan->length.push_back(divLength0);

    col2colPlan->scheme    = CS_KERNEL_STOCKHAM_BLOCK_CC;
    col2colPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        col2colPlan->length.push_back(length[index]);
    }

    childNodes.push_back(col2colPlan);

    // second plan, row-to-row
    TreeNode* row2rowPlan = TreeNode::CreateNode(this);

    row2rowPlan->length.push_back(divLength0);
    row2rowPlan->length.push_back(divLength1);

    row2rowPlan->scheme    = CS_KERNEL_STOCKHAM;
    row2rowPlan->dimension = 1;

    for(size_t index = 1; index < length.size(); index++)
    {
        row2rowPlan->length.push_back(length[index]);
    }

    childNodes.push_back(row2rowPlan);

    // third plan, transpose
    TreeNode* transPlan = TreeNode::CreateNode(this);

    transPlan->length.push_back(divLength0);
    transPlan->length.push_back(divLength1);

    transPlan->scheme    = CS_KERNEL_TRANSPOSE;
    transPlan->dimension = 2;

    for(size_t index = 1; index < length.size(); index++)
    {
        transPlan->length.push_back(length[index]);
    }

    childNodes.push_back(transPlan);
}

void TreeNode::build_CS_2D_RTRT()
{
    // first row fft
    TreeNode* row1Plan = TreeNode::CreateNode(this);

    row1Plan->length.push_back(length[0]);
    row1Plan->dimension = 1;
    row1Plan->length.push_back(length[1]);

    for(size_t index = 2; index < length.size(); index++)
    {
        row1Plan->length.push_back(length[index]);
    }

    row1Plan->RecursiveBuildTree();
    childNodes.push_back(row1Plan);

    // first transpose
    TreeNode* trans1Plan = TreeNode::CreateNode(this);

    trans1Plan->length.push_back(length[0]);
    trans1Plan->length.push_back(length[1]);

    trans1Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans1Plan->dimension = 2;

    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }

    childNodes.push_back(trans1Plan);

    // second row fft
    TreeNode* row2Plan = TreeNode::CreateNode(this);

    row2Plan->length.push_back(length[1]);
    row2Plan->dimension = 1;
    row2Plan->length.push_back(length[0]);

    for(size_t index = 2; index < length.size(); index++)
    {
        row2Plan->length.push_back(length[index]);
    }

    row2Plan->RecursiveBuildTree();
    childNodes.push_back(row2Plan);

    // second transpose
    TreeNode* trans2Plan = TreeNode::CreateNode(this);

    trans2Plan->length.push_back(length[1]);
    trans2Plan->length.push_back(length[0]);

    trans2Plan->scheme    = CS_KERNEL_TRANSPOSE;
    trans2Plan->dimension = 2;

    for(size_t index = 2; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }

    childNodes.push_back(trans2Plan);
}

void TreeNode::RecursiveBuildTree()
{
    // this flag can be enabled when generator can do block column fft in
    // multi-dimension cases and small 2d, 3d within one kernel
    bool MultiDimFuseKernelsAvailable = false;

    if((parent == nullptr)
       && ((inArrayType == rocfft_array_type_real) || (outArrayType == rocfft_array_type_real)))
    {
        build_real();
        return;
    }

    switch(dimension)
    {
    case 1:
        build_1D();
        break;

    case 2:
    {
        if(scheme == CS_KERNEL_TRANSPOSE)
            return;

        if(MultiDimFuseKernelsAvailable)
        {
            // conditions to choose which scheme
            if((length[0] * length[1]) <= 2048)
            {
                scheme = CS_KERNEL_2D_SINGLE;
            }
            else if(length[1] <= 256)
            {
                scheme = CS_2D_RC;
            }
            else
            {
                scheme = CS_2D_RTRT;
            }
        }
        else
        {
            scheme = CS_2D_RTRT;
        }

        switch(scheme)
        {
        case CS_2D_RTRT:
            build_CS_2D_RTRT();
            break;
        case CS_2D_RC:
        {
            // row fft
            TreeNode* rowPlan = TreeNode::CreateNode(this);

            rowPlan->length.push_back(length[0]);
            rowPlan->dimension = 1;
            rowPlan->length.push_back(length[1]);

            for(size_t index = 2; index < length.size(); index++)
            {
                rowPlan->length.push_back(length[index]);
            }

            rowPlan->RecursiveBuildTree();
            childNodes.push_back(rowPlan);

            // column fft
            TreeNode* colPlan = TreeNode::CreateNode(this);

            colPlan->length.push_back(length[1]);
            colPlan->dimension = 1;
            colPlan->length.push_back(length[0]);

            for(size_t index = 2; index < length.size(); index++)
            {
                colPlan->length.push_back(length[index]);
            }

            colPlan->scheme = CS_KERNEL_2D_STOCKHAM_BLOCK_CC;
            childNodes.push_back(colPlan);
        }
        break;
        case CS_KERNEL_2D_SINGLE:
        {
        }
        break;

        default:
            assert(false);
        }
    }
    break;

    case 3:
    {
        if(MultiDimFuseKernelsAvailable)
        {
            // conditions to choose which scheme
            if((length[0] * length[1] * length[2]) <= 2048)
                scheme = CS_KERNEL_3D_SINGLE;
            else if(length[2] <= 256)
                scheme = CS_3D_RC;
            else
                scheme = CS_3D_RTRT;
        }
        else
            scheme = CS_3D_RTRT;

        switch(scheme)
        {
        case CS_3D_RTRT:
        {
            // 2d fft
            TreeNode* xyPlan = TreeNode::CreateNode(this);

            xyPlan->length.push_back(length[0]);
            xyPlan->length.push_back(length[1]);
            xyPlan->dimension = 2;
            xyPlan->length.push_back(length[2]);

            for(size_t index = 3; index < length.size(); index++)
            {
                xyPlan->length.push_back(length[index]);
            }

            xyPlan->RecursiveBuildTree();
            childNodes.push_back(xyPlan);

            // first transpose
            TreeNode* trans1Plan = TreeNode::CreateNode(this);

            trans1Plan->length.push_back(length[0]);
            trans1Plan->length.push_back(length[1]);
            trans1Plan->length.push_back(length[2]);

            trans1Plan->scheme    = CS_KERNEL_TRANSPOSE_XY_Z;
            trans1Plan->dimension = 2;

            for(size_t index = 3; index < length.size(); index++)
            {
                trans1Plan->length.push_back(length[index]);
            }

            childNodes.push_back(trans1Plan);

            // z fft
            TreeNode* zPlan = TreeNode::CreateNode(this);

            zPlan->length.push_back(length[2]);
            zPlan->dimension = 1;
            zPlan->length.push_back(length[0]);
            zPlan->length.push_back(length[1]);

            for(size_t index = 3; index < length.size(); index++)
            {
                zPlan->length.push_back(length[index]);
            }

            zPlan->RecursiveBuildTree();
            childNodes.push_back(zPlan);

            // second transpose
            TreeNode* trans2Plan = TreeNode::CreateNode(this);

            trans2Plan->length.push_back(length[2]);
            trans2Plan->length.push_back(length[0]);
            trans2Plan->length.push_back(length[1]);

            trans2Plan->scheme    = CS_KERNEL_TRANSPOSE_Z_XY;
            trans2Plan->dimension = 2;

            for(size_t index = 3; index < length.size(); index++)
            {
                trans2Plan->length.push_back(length[index]);
            }

            childNodes.push_back(trans2Plan);
        }
        break;
        case CS_3D_RC:
        {
            // 2d fft
            TreeNode* xyPlan = TreeNode::CreateNode(this);

            xyPlan->length.push_back(length[0]);
            xyPlan->length.push_back(length[1]);
            xyPlan->dimension = 2;
            xyPlan->length.push_back(length[2]);

            for(size_t index = 3; index < length.size(); index++)
            {
                xyPlan->length.push_back(length[index]);
            }

            xyPlan->RecursiveBuildTree();
            childNodes.push_back(xyPlan);

            // z col fft
            TreeNode* zPlan = TreeNode::CreateNode(this);

            zPlan->length.push_back(length[2]);
            zPlan->dimension = 1;
            zPlan->length.push_back(length[0]);
            zPlan->length.push_back(length[1]);

            for(size_t index = 3; index < length.size(); index++)
            {
                zPlan->length.push_back(length[index]);
            }

            zPlan->scheme = CS_KERNEL_3D_STOCKHAM_BLOCK_CC;
            childNodes.push_back(zPlan);
        }
        break;
        case CS_KERNEL_3D_SINGLE:
        {
        }
        break;

        default:
            assert(false);
        }
    }
    break;

    default:
        assert(false);
    }
}

void TreeNode::assign_buffers_CS_REAL_TRANSFORM_USING_CMPLX(OperatingBuffer& flipIn,
                                                            OperatingBuffer& flipOut,
                                                            OperatingBuffer& obOutBuf)
{
    assert(parent == nullptr);
    assert(childNodes.size() == 3);

    assert((direction == -1 && childNodes[0]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
           || (direction == 1 && childNodes[0]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX));

    obIn  = OB_USER_IN;
    obOut = placement == rocfft_placement_inplace ? OB_USER_IN : OB_USER_OUT;

    assert((direction == -1 && childNodes[0]->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
           || (direction == 1 && childNodes[0]->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX));

    childNodes[0]->obIn  = obIn;
    childNodes[0]->obOut = OB_TEMP_CMPLX_FOR_REAL;

    childNodes[1]->obIn  = OB_TEMP_CMPLX_FOR_REAL;
    childNodes[1]->obOut = flipIn;
    childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);
    size_t cs = childNodes[1]->childNodes.size();
    if(cs)
    {
        if(childNodes[1]->scheme == CS_BLUESTEIN)
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_BLUESTEIN);
            assert(childNodes[1]->childNodes[1]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        else
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_CMPLX_FOR_REAL);
        }
        assert(childNodes[1]->childNodes[cs - 1]->obOut == OB_TEMP_CMPLX_FOR_REAL);
    }

    assert((direction == -1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
           || (direction == 1 && childNodes[2]->scheme == CS_KERNEL_COPY_CMPLX_TO_R));
    childNodes[2]->obIn  = OB_TEMP_CMPLX_FOR_REAL;
    childNodes[2]->obOut = obOut;
}

void TreeNode::assign_buffers_CS_REAL_TRANSFORM_EVEN(OperatingBuffer& flipIn,
                                                     OperatingBuffer& flipOut,
                                                     OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        obIn  = OB_USER_IN;
        obOut = placement == rocfft_placement_inplace ? OB_USER_IN : OB_USER_OUT;
    }

    if(direction == -1)
    {
        // real-to-complex

        // complex FFT kernel
        childNodes[0]->obIn         = obIn;
        childNodes[0]->obOut        = obIn;
        childNodes[0]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;
        childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        size_t cs = childNodes[0]->childNodes.size();
        if(cs)
        {
            assert(childNodes[0]->childNodes[0]->obIn == obIn);
            assert(childNodes[0]->childNodes[cs - 1]->obOut == obIn);
        }

        // real-to-complex post kernel
        childNodes[1]->obIn  = obIn;
        childNodes[1]->obOut = obOut;
    }
    else
    {
        // complex-to-real

        // complex-to-real pre kernel
        childNodes[0]->obIn  = obIn;
        childNodes[0]->obOut = obOut;

        // complex FFT kernel
        childNodes[1]->obIn         = obOut;
        childNodes[1]->obOut        = obOut;
        flipIn                      = OB_USER_OUT;
        flipOut                     = OB_TEMP;
        childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;
        childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        size_t cs = childNodes[1]->childNodes.size();
        if(cs)
        {
            assert(childNodes[1]->childNodes[0]->obIn == obOut);
            assert(childNodes[1]->childNodes[cs - 1]->obOut == obOut);
        }
        assert(childNodes[1]->obIn == obOut);
        assert(childNodes[1]->obOut == obOut);
    }
}

void TreeNode::assign_buffers_CS_REAL_2D_EVEN(OperatingBuffer& flipIn,
                                              OperatingBuffer& flipOut,
                                              OperatingBuffer& obOutBuf)
{
    assert(scheme == CS_REAL_2D_EVEN);
    assert(parent == nullptr);

    if(direction == -1)
    {
        // RTRT

        childNodes[0]->obIn  = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
        childNodes[0]->obOut = obOutBuf;

        flipIn  = OB_USER_OUT;
        flipOut = OB_TEMP;
        childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes[1]->obIn  = obOutBuf;
        childNodes[1]->obOut = OB_TEMP;

        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes[2]->obIn  = OB_TEMP;
        childNodes[2]->obOut = OB_TEMP;
        flipIn               = OB_TEMP;
        flipOut              = obOutBuf;
        childNodes[2]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[3]->outArrayType = rocfft_array_type_hermitian_interleaved;

        childNodes[3]->obIn  = OB_TEMP;
        childNodes[3]->obOut = obOutBuf;

        obIn  = childNodes[0]->obIn;
        obOut = childNodes[3]->obOut;
    }
    else
    { // TRTR

        // T
        childNodes[0]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[0]->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes[0]->obIn  = OB_USER_IN;
        childNodes[0]->obOut = OB_TEMP;

        // C2C
        childNodes[1]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[1]->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes[1]->obIn  = OB_TEMP;
        childNodes[1]->obOut = OB_TEMP;

        flipIn  = OB_TEMP;
        flipOut = OB_USER_IN;
        childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        // T
        childNodes[2]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[2]->outArrayType = rocfft_array_type_complex_interleaved;

        childNodes[2]->obIn  = OB_TEMP;
        childNodes[2]->obOut = OB_USER_IN;

        // C2R
        childNodes[3]->inArrayType  = rocfft_array_type_complex_interleaved;
        childNodes[3]->outArrayType = rocfft_array_type_real;

        flipIn  = OB_TEMP;
        flipOut = OB_USER_OUT;

        childNodes[3]->obIn  = OB_USER_IN;
        childNodes[3]->obOut = OB_USER_OUT;
        childNodes[3]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        obIn  = OB_USER_IN;
        obOut = OB_USER_OUT;
    }
}

void TreeNode::assign_buffers_CS_BLUESTEIN(OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    assert(childNodes.size() == 7);

    OperatingBuffer savFlipIn  = flipIn;
    OperatingBuffer savFlipOut = flipOut;
    OperatingBuffer savOutBuf  = obOutBuf;

    flipIn   = OB_TEMP_BLUESTEIN;
    flipOut  = OB_TEMP;
    obOutBuf = OB_TEMP_BLUESTEIN;

    assert(childNodes[0]->scheme == CS_KERNEL_CHIRP);
    childNodes[0]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[0]->obOut = OB_TEMP_BLUESTEIN;

    assert(childNodes[1]->scheme == CS_KERNEL_PAD_MUL);
    if(parent == nullptr)
    {
        childNodes[1]->obIn = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
    }
    else
    {
        childNodes[1]->obIn = obIn;
    }

    childNodes[1]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[2]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[2]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[2]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    childNodes[3]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[3]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[3]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    assert(childNodes[4]->scheme == CS_KERNEL_FFT_MUL);
    childNodes[4]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[4]->obOut = OB_TEMP_BLUESTEIN;

    childNodes[5]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[5]->obOut = OB_TEMP_BLUESTEIN;
    childNodes[5]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    assert(childNodes[6]->scheme == CS_KERNEL_RES_MUL);
    childNodes[6]->obIn  = OB_TEMP_BLUESTEIN;
    childNodes[6]->obOut = (parent == nullptr) ? OB_USER_OUT : obOut;

    obIn  = childNodes[1]->obIn;
    obOut = childNodes[6]->obOut;

    flipIn   = savFlipIn;
    flipOut  = savFlipOut;
    obOutBuf = savOutBuf;
}

void TreeNode::assign_buffers_CS_L1D_TRTRT(OperatingBuffer& flipIn,
                                           OperatingBuffer& flipOut,
                                           OperatingBuffer& obOutBuf)
{
    childNodes[0]->obIn = (parent == nullptr)
                              ? ((placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN)
                              : flipIn;
    childNodes[0]->obOut = flipOut;

    std::swap(flipIn, flipOut);

    if(childNodes[1]->childNodes.size())
    {
        childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        size_t cs            = childNodes[1]->childNodes.size();
        childNodes[1]->obIn  = childNodes[1]->childNodes[0]->obIn;
        childNodes[1]->obOut = childNodes[1]->childNodes[cs - 1]->obOut;
    }
    else
    {
        childNodes[1]->obIn  = flipIn;
        childNodes[1]->obOut = obOutBuf;

        if(flipIn != obOutBuf)
        {
            std::swap(flipIn, flipOut);
        }
    }

    if((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
    {
        if(flipIn == OB_TEMP)
        {
            childNodes[2]->obIn  = OB_TEMP;
            childNodes[2]->obOut = obOutBuf;

            childNodes[3]->obIn  = obOutBuf;
            childNodes[3]->obOut = OB_TEMP;

            childNodes[4]->obIn  = OB_TEMP;
            childNodes[4]->obOut = obOutBuf;
        }
        else
        {
            childNodes[2]->obIn  = obOutBuf;
            childNodes[2]->obOut = OB_TEMP;

            childNodes[3]->obIn  = OB_TEMP;
            childNodes[3]->obOut = OB_TEMP;

            childNodes[4]->obIn  = OB_TEMP;
            childNodes[4]->obOut = obOutBuf;
        }

        obIn  = childNodes[0]->obIn;
        obOut = childNodes[4]->obOut;
    }
    else
    {
        assert(obIn == obOut);

        if(obOut == obOutBuf)
        {
            if(childNodes[1]->obOut == OB_TEMP)
            {
                childNodes[2]->obIn  = OB_TEMP;
                childNodes[2]->obOut = obOutBuf;

                childNodes[3]->obIn  = obOutBuf;
                childNodes[3]->obOut = OB_TEMP;

                childNodes[4]->obIn  = OB_TEMP;
                childNodes[4]->obOut = obOutBuf;
            }
            else
            {
                childNodes[2]->obIn  = obOutBuf;
                childNodes[2]->obOut = OB_TEMP;

                childNodes[3]->obIn  = OB_TEMP;
                childNodes[3]->obOut = OB_TEMP;

                childNodes[4]->obIn  = OB_TEMP;
                childNodes[4]->obOut = obOutBuf;
            }
        }
        else
        {
            if(childNodes[1]->obOut == OB_TEMP)
            {
                childNodes[2]->obIn  = OB_TEMP;
                childNodes[2]->obOut = obOutBuf;

                childNodes[3]->obIn  = obOutBuf;
                childNodes[3]->obOut = obOutBuf;

                childNodes[4]->obIn  = obOutBuf;
                childNodes[4]->obOut = OB_TEMP;
            }
            else
            {
                childNodes[2]->obIn  = obOutBuf;
                childNodes[2]->obOut = OB_TEMP;

                childNodes[3]->obIn  = OB_TEMP;
                childNodes[3]->obOut = obOutBuf;

                childNodes[4]->obIn  = obOutBuf;
                childNodes[4]->obOut = OB_TEMP;
            }
        }
    }
}

void TreeNode::assign_buffers_CS_L1D_CC(OperatingBuffer& flipIn,
                                        OperatingBuffer& flipOut,
                                        OperatingBuffer& obOutBuf)
{
    if((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
    {
        if(parent == nullptr)
        {
            childNodes[0]->obIn  = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
            childNodes[0]->obOut = OB_TEMP;

            childNodes[1]->obIn  = OB_TEMP;
            childNodes[1]->obOut = obOutBuf;
        }
        else
        {

            childNodes[0]->obIn  = flipIn;
            childNodes[0]->obOut = flipOut;

            childNodes[1]->obIn  = flipOut;
            childNodes[1]->obOut = flipIn;
        }

        obIn  = childNodes[0]->obIn;
        obOut = childNodes[1]->obOut;
    }
    else
    {
        childNodes[0]->obIn  = obIn;
        childNodes[0]->obOut = flipOut;

        childNodes[1]->obIn  = flipOut;
        childNodes[1]->obOut = obOut;
    }
}

void TreeNode::assign_buffers_CS_L1D_CRT(OperatingBuffer& flipIn,
                                         OperatingBuffer& flipOut,
                                         OperatingBuffer& obOutBuf)
{
    if((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
    {
        if(parent == nullptr)
        {
            childNodes[0]->obIn  = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
            childNodes[0]->obOut = OB_TEMP;

            childNodes[1]->obIn  = OB_TEMP;
            childNodes[1]->obOut = OB_TEMP;

            childNodes[2]->obIn  = OB_TEMP;
            childNodes[2]->obOut = obOutBuf;
        }
        else
        {
            childNodes[0]->obIn  = flipIn;
            childNodes[0]->obOut = flipOut;

            childNodes[1]->obIn  = flipOut;
            childNodes[1]->obOut = flipOut;

            childNodes[2]->obIn  = flipOut;
            childNodes[2]->obOut = flipIn;
        }

        obIn  = childNodes[0]->obIn;
        obOut = childNodes[2]->obOut;
    }
    else
    {
        assert(obIn == flipIn);
        assert(obIn == obOut);

        childNodes[0]->obIn  = flipIn;
        childNodes[0]->obOut = flipOut;

        childNodes[1]->obIn  = flipOut;
        childNodes[1]->obOut = flipOut;

        childNodes[2]->obIn  = flipOut;
        childNodes[2]->obOut = flipIn;
    }
}

void TreeNode::assign_buffers_CS_RTRT(OperatingBuffer& flipIn,
                                      OperatingBuffer& flipOut,
                                      OperatingBuffer& obOutBuf)
{
    childNodes[0]->obIn = (parent == nullptr)
                              ? (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN
                              : obOutBuf;

    childNodes[0]->obOut = obOutBuf;

    flipIn  = obOutBuf;
    flipOut = OB_TEMP;
    childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    childNodes[1]->obIn  = obOutBuf;
    childNodes[1]->obOut = OB_TEMP;

    childNodes[2]->obIn  = OB_TEMP;
    childNodes[2]->obOut = OB_TEMP;

    flipIn  = OB_TEMP;
    flipOut = obOutBuf;
    childNodes[2]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    childNodes[3]->obIn  = OB_TEMP;
    childNodes[3]->obOut = obOutBuf;

    obIn  = childNodes[0]->obIn;
    obOut = childNodes[3]->obOut;
}

void TreeNode::assign_buffers_CS_RC(OperatingBuffer& flipIn,
                                    OperatingBuffer& flipOut,
                                    OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
        childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
    else
        childNodes[0]->obIn = obOutBuf;

    childNodes[0]->obOut = obOutBuf;

    flipIn  = obOutBuf;
    flipOut = OB_TEMP;
    childNodes[0]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    childNodes[1]->obIn  = obOutBuf;
    childNodes[1]->obOut = obOutBuf;

    obIn  = childNodes[0]->obIn;
    obOut = childNodes[1]->obOut;
}

// Assign buffers, taking into account out-of-place transposes and
// padded buffers.
// NB: this recursive function modifies the parameters in the parent call.
void TreeNode::TraverseTreeAssignBuffersLogicA(OperatingBuffer& flipIn,
                                               OperatingBuffer& flipOut,
                                               OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        // Set flipIn, flipOut, and oboutBuf for the root node.
        assert(flipIn == OB_UNINIT);
        assert(flipOut == OB_UNINIT);
        assert(obOutBuf == OB_UNINIT);
        switch(scheme)
        {
        case CS_REAL_TRANSFORM_USING_CMPLX:
            flipIn   = OB_TEMP_CMPLX_FOR_REAL;
            flipOut  = OB_TEMP;
            obOutBuf = OB_TEMP_CMPLX_FOR_REAL;
            break;
        case CS_REAL_TRANSFORM_EVEN:
            // The sub-transform is always in-place.
            flipIn = (direction == -1 || placement == rocfft_placement_inplace) ? OB_USER_IN
                                                                                : OB_USER_OUT;
            flipOut  = OB_TEMP;
            obOutBuf = (direction == -1 || placement == rocfft_placement_inplace) ? OB_USER_IN
                                                                                  : OB_USER_OUT;
            break;
        case CS_BLUESTEIN:
            flipIn   = OB_TEMP_BLUESTEIN;
            flipOut  = OB_TEMP;
            obOutBuf = OB_TEMP_BLUESTEIN;
            break;
        default:
            flipIn   = OB_USER_OUT;
            flipOut  = OB_TEMP;
            obOutBuf = OB_USER_OUT;
        }
    }

#if 0
    auto here = this;
    auto up   = parent;
    while(up != nullptr && here != up)
    {
        here = up;
        up   = parent->parent;
        std::cout << "\t";
    }
    std::cout << "TraverseTreeAssignBuffersLogicA: " << PrintScheme(scheme) << ": "
              << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut) << std::endl;
#endif

    switch(scheme)
    {
    case CS_REAL_TRANSFORM_USING_CMPLX:
        assign_buffers_CS_REAL_TRANSFORM_USING_CMPLX(flipIn, flipOut, obOutBuf);
        break;
    case CS_REAL_TRANSFORM_EVEN:
        assign_buffers_CS_REAL_TRANSFORM_EVEN(flipIn, flipOut, obOutBuf);
        break;
    case CS_REAL_2D_EVEN:
        assign_buffers_CS_REAL_2D_EVEN(flipIn, flipOut, obOutBuf);
        break;
    case CS_BLUESTEIN:
        assign_buffers_CS_BLUESTEIN(flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_TRTRT:
        assign_buffers_CS_L1D_TRTRT(flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_CC:
        assign_buffers_CS_L1D_CC(flipIn, flipOut, obOutBuf);
        break;
    case CS_L1D_CRT:
        assign_buffers_CS_L1D_CRT(flipIn, flipOut, obOutBuf);
        break;
    case CS_2D_RTRT:
    case CS_3D_RTRT:
        assign_buffers_CS_RTRT(flipIn, flipOut, obOutBuf);
        break;
    case CS_2D_RC:
    case CS_3D_RC:
        assign_buffers_CS_RC(flipIn, flipOut, obOutBuf);
        break;
    default:
        if(parent == nullptr)
        {
            obIn  = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
            obOut = obOutBuf;
        }
        else
        {
            assert(obIn != OB_UNINIT);
            assert(obOut != OB_UNINIT);
            if(obIn != obOut)
            {
                std::swap(flipIn, flipOut);
            }
        }
    }

    // Assert that all operating buffers have been assigned
    assert(obIn != OB_UNINIT);
    assert(obOut != OB_UNINIT);
    for(int i = 0; i < childNodes.size(); ++i)
    {
        assert(childNodes[i]->obIn != OB_UNINIT);
        assert(childNodes[i]->obOut != OB_UNINIT);
    }

    // Assert that the kernel chain is connected
    for(int i = 1; i < childNodes.size(); ++i)
    {
        if(childNodes[i - 1]->scheme == CS_KERNEL_CHIRP)
        {
            // The Bluestein algorithm uses a separate buffer which is
            // convoluted with the input; the chain assumption isn't true here.
            // NB: we assume that the CS_KERNEL_CHIRP is first in the chain.
            continue;
        }
        assert(childNodes[i - 1]->obOut == childNodes[i]->obIn);
    }
}

// Set placement variable and in/out array types, if not already set.
void TreeNode::TraverseTreeAssignPlacementsLogicA(const rocfft_array_type rootIn,
                                                  const rocfft_array_type rootOut)
{
    if(parent != nullptr)
    {
        placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;

        if(inArrayType == rocfft_array_type_unset)
        {
            switch(obIn)
            {
            case OB_USER_IN:
                inArrayType = (parent == nullptr) ? rootIn : parent->inArrayType;
                break;
            case OB_USER_OUT:
                inArrayType = (parent == nullptr) ? rootOut : parent->outArrayType;
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
                outArrayType = (parent == nullptr) ? rootIn : parent->inArrayType;
                break;
            case OB_USER_OUT:
                outArrayType = (parent == nullptr) ? rootOut : parent->outArrayType;
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

void TreeNode::assign_params_CS_REAL_TRANSFORM_USING_CMPLX()
{
    assert(childNodes.size() == 3);
    TreeNode* copyHeadPlan = childNodes[0];
    TreeNode* fftPlan      = childNodes[1];
    TreeNode* copyTailPlan = childNodes[2];

    copyHeadPlan->inStride = inStride;
    copyHeadPlan->iDist    = iDist;

    copyHeadPlan->outStride.push_back(1);
    copyHeadPlan->oDist = copyHeadPlan->length[0];
    for(size_t index = 1; index < length.size(); index++)
    {
        copyHeadPlan->outStride.push_back(copyHeadPlan->oDist);
        copyHeadPlan->oDist *= length[index];
    }

    fftPlan->inStride  = copyHeadPlan->outStride;
    fftPlan->iDist     = copyHeadPlan->oDist;
    fftPlan->outStride = fftPlan->inStride;
    fftPlan->oDist     = fftPlan->iDist;

    fftPlan->TraverseTreeAssignParamsLogicA();

    copyTailPlan->inStride = fftPlan->outStride;
    copyTailPlan->iDist    = fftPlan->oDist;

    copyTailPlan->outStride = outStride;
    copyTailPlan->oDist     = oDist;
}

void TreeNode::assign_params_CS_REAL_TRANSFORM_EVEN()
{
    assert(childNodes.size() == 2);

    if(direction == -1)
    {
        // forward transform, r2c

        // iDist is in reals, subplan->iDist is in complexes

        TreeNode* fftPlan = childNodes[0];
        fftPlan->inStride = inStride;
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            fftPlan->inStride[i] /= 2;
        }
        fftPlan->iDist     = iDist / 2;
        fftPlan->outStride = inStride;
        for(int i = 1; i < fftPlan->outStride.size(); ++i)
        {
            fftPlan->outStride[i] /= 2;
        }
        fftPlan->oDist = iDist / 2;
        fftPlan->TraverseTreeAssignParamsLogicA();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        TreeNode* postPlan = childNodes[1];
        assert(postPlan->scheme == CS_KERNEL_R_TO_CMPLX);
        postPlan->inStride = inStride;
        for(int i = 1; i < postPlan->inStride.size(); ++i)
        {
            postPlan->inStride[i] /= 2;
        }
        postPlan->iDist     = iDist / 2;
        postPlan->outStride = outStride;
        postPlan->oDist     = oDist;

        assert(postPlan->length.size() == postPlan->inStride.size());
        assert(postPlan->length.size() == postPlan->outStride.size());
    }
    else
    {
        // backward transform, c2r

        // oDist is in reals, subplan->oDist is in complexes

        TreeNode* prePlan = childNodes[0];
        assert(prePlan->scheme == CS_KERNEL_CMPLX_TO_R);

        prePlan->iDist = iDist;
        prePlan->oDist = oDist / 2;

        // Strides are actually distances for multimensional transforms.
        // Only the first value is used, but we require dimension values.
        prePlan->inStride  = inStride;
        prePlan->outStride = outStride;
        // Strides are in complex types
        for(int i = 1; i < prePlan->outStride.size(); ++i)
        {
            //prePlan->inStride[i] /= 2;
            prePlan->outStride[i] /= 2;
        }

        TreeNode* fftPlan = childNodes[1];
        // Transform the strides from real to complex.

        fftPlan->inStride  = outStride;
        fftPlan->iDist     = oDist / 2;
        fftPlan->outStride = outStride;
        fftPlan->oDist     = fftPlan->iDist;
        // The strides must be translated from real to complex.
        for(int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            fftPlan->inStride[i] /= 2;
            fftPlan->outStride[i] /= 2;
        }

        fftPlan->TraverseTreeAssignParamsLogicA();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        assert(prePlan->length.size() == prePlan->inStride.size());
        assert(prePlan->length.size() == prePlan->outStride.size());
    }
}

void TreeNode::assign_params_CS_L1D_CC()
{
    TreeNode* col2colPlan = childNodes[0];
    TreeNode* row2colPlan = childNodes[1];

    if((obOut == OB_USER_OUT) || (obOut == OB_TEMP_CMPLX_FOR_REAL))
    {
        // B -> T
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        col2colPlan->outStride.push_back(col2colPlan->length[1]);
        col2colPlan->outStride.push_back(1);
        col2colPlan->oDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            col2colPlan->inStride.push_back(inStride[index]);
            col2colPlan->outStride.push_back(col2colPlan->oDist);
            col2colPlan->oDist *= length[index];
        }

        // T -> B
        row2colPlan->inStride.push_back(1);
        row2colPlan->inStride.push_back(row2colPlan->length[0]);
        row2colPlan->iDist = length[0];

        row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
        row2colPlan->outStride.push_back(outStride[0]);
        row2colPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
        {
            row2colPlan->inStride.push_back(row2colPlan->iDist);
            row2colPlan->iDist *= length[index];
            row2colPlan->outStride.push_back(outStride[index]);
        }
    }
    else
    {
        // here we don't have B info right away, we get it through its parent

        // TODO: what is this assert for?
        assert(parent->obOut == OB_USER_IN || parent->obOut == OB_USER_OUT
               || parent->obOut == OB_TEMP_CMPLX_FOR_REAL
               || parent->scheme == CS_REAL_TRANSFORM_EVEN);

        // T-> B
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        for(size_t index = 1; index < length.size(); index++)
            col2colPlan->inStride.push_back(inStride[index]);

        if(parent->scheme == CS_L1D_TRTRT)
        {
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
            col2colPlan->outStride.push_back(parent->outStride[0]);
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]
                                             * col2colPlan->length[0]);
            col2colPlan->oDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                col2colPlan->outStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert(parent->outStride[0] == 1);

            col2colPlan->outStride.push_back(col2colPlan->length[1]);
            col2colPlan->outStride.push_back(1);
            col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

            for(size_t index = 1; index < length.size(); index++)
            {
                col2colPlan->outStride.push_back(col2colPlan->oDist);
                col2colPlan->oDist *= length[index];
            }
        }

        // B -> T
        if(parent->scheme == CS_L1D_TRTRT)
        {
            row2colPlan->inStride.push_back(parent->outStride[0]);
            row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0]);
            row2colPlan->inStride.push_back(parent->outStride[0] * row2colPlan->length[0]
                                            * row2colPlan->length[1]);
            row2colPlan->iDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                row2colPlan->inStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2colPlan->inStride.push_back(1);
            row2colPlan->inStride.push_back(row2colPlan->length[0]);
            row2colPlan->iDist = row2colPlan->length[0] * row2colPlan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2colPlan->inStride.push_back(row2colPlan->iDist);
                row2colPlan->iDist *= length[index];
            }
        }

        row2colPlan->outStride.push_back(outStride[0] * row2colPlan->length[1]);
        row2colPlan->outStride.push_back(outStride[0]);
        row2colPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            row2colPlan->outStride.push_back(outStride[index]);
    }
}

void TreeNode::assign_params_CS_L1D_CRT()
{
    TreeNode* col2colPlan = childNodes[0];
    TreeNode* row2rowPlan = childNodes[1];
    TreeNode* transPlan   = childNodes[2];

    if(parent != NULL)
        assert(obIn == obOut);

    if((obOut == OB_USER_OUT) || (obOut == OB_TEMP_CMPLX_FOR_REAL))
    {
        // B -> T
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        col2colPlan->outStride.push_back(col2colPlan->length[1]);
        col2colPlan->outStride.push_back(1);
        col2colPlan->oDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            col2colPlan->inStride.push_back(inStride[index]);
            col2colPlan->outStride.push_back(col2colPlan->oDist);
            col2colPlan->oDist *= length[index];
        }

        // T -> T
        row2rowPlan->inStride.push_back(1);
        row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
        row2rowPlan->iDist = length[0];

        for(size_t index = 1; index < length.size(); index++)
        {
            row2rowPlan->inStride.push_back(row2rowPlan->iDist);
            row2rowPlan->iDist *= length[index];
        }

        row2rowPlan->outStride = row2rowPlan->inStride;
        row2rowPlan->oDist     = row2rowPlan->iDist;

        // T -> B
        transPlan->inStride = row2rowPlan->outStride;
        transPlan->iDist    = row2rowPlan->oDist;

        transPlan->outStride.push_back(outStride[0]);
        transPlan->outStride.push_back(outStride[0] * (transPlan->length[1]));
        transPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            transPlan->outStride.push_back(outStride[index]);
    }
    else
    {
        // here we don't have B info right away, we get it through its parent
        assert((parent->obOut == OB_USER_OUT) || (parent->obOut == OB_TEMP_CMPLX_FOR_REAL));

        // T -> B
        col2colPlan->inStride.push_back(inStride[0] * col2colPlan->length[1]);
        col2colPlan->inStride.push_back(inStride[0]);
        col2colPlan->iDist = iDist;

        for(size_t index = 1; index < length.size(); index++)
            col2colPlan->inStride.push_back(inStride[index]);

        if(parent->scheme == CS_L1D_TRTRT)
        {
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]);
            col2colPlan->outStride.push_back(parent->outStride[0]);
            col2colPlan->outStride.push_back(parent->outStride[0] * col2colPlan->length[1]
                                             * col2colPlan->length[0]);
            col2colPlan->oDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                col2colPlan->outStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert(parent->outStride[0] == 1);
            for(size_t index = 1; index < parent->length.size(); index++)
                assert(parent->outStride[index]
                       == (parent->outStride[index - 1] * parent->length[index - 1]));

            col2colPlan->outStride.push_back(col2colPlan->length[1]);
            col2colPlan->outStride.push_back(1);
            col2colPlan->oDist = col2colPlan->length[1] * col2colPlan->length[0];

            for(size_t index = 1; index < length.size(); index++)
            {
                col2colPlan->outStride.push_back(col2colPlan->oDist);
                col2colPlan->oDist *= length[index];
            }
        }

        // B -> B
        if(parent->scheme == CS_L1D_TRTRT)
        {
            row2rowPlan->inStride.push_back(parent->outStride[0]);
            row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0]);
            row2rowPlan->inStride.push_back(parent->outStride[0] * row2rowPlan->length[0]
                                            * row2rowPlan->length[1]);
            row2rowPlan->iDist = parent->oDist;

            for(size_t index = 1; index < parent->length.size(); index++)
                row2rowPlan->inStride.push_back(parent->outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2rowPlan->inStride.push_back(1);
            row2rowPlan->inStride.push_back(row2rowPlan->length[0]);
            row2rowPlan->iDist = row2rowPlan->length[0] * row2rowPlan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2rowPlan->inStride.push_back(row2rowPlan->iDist);
                row2rowPlan->iDist *= length[index];
            }
        }

        row2rowPlan->outStride = row2rowPlan->inStride;
        row2rowPlan->oDist     = row2rowPlan->iDist;

        // B -> T
        transPlan->inStride = row2rowPlan->outStride;
        transPlan->iDist    = row2rowPlan->oDist;

        transPlan->outStride.push_back(outStride[0]);
        transPlan->outStride.push_back(outStride[0] * transPlan->length[1]);
        transPlan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            transPlan->outStride.push_back(outStride[index]);
    }
}

void TreeNode::assign_params_CS_BLUESTEIN()
{
    TreeNode* chirpPlan  = childNodes[0];
    TreeNode* padmulPlan = childNodes[1];
    TreeNode* fftiPlan   = childNodes[2];
    TreeNode* fftcPlan   = childNodes[3];
    TreeNode* fftmulPlan = childNodes[4];
    TreeNode* fftrPlan   = childNodes[5];
    TreeNode* resmulPlan = childNodes[6];

    chirpPlan->inStride.push_back(1);
    chirpPlan->iDist = chirpPlan->lengthBlue;
    chirpPlan->outStride.push_back(1);
    chirpPlan->oDist = chirpPlan->lengthBlue;

    padmulPlan->inStride = inStride;
    padmulPlan->iDist    = iDist;

    padmulPlan->outStride.push_back(1);
    padmulPlan->oDist = padmulPlan->lengthBlue;
    for(size_t index = 1; index < length.size(); index++)
    {
        padmulPlan->outStride.push_back(padmulPlan->oDist);
        padmulPlan->oDist *= length[index];
    }

    fftiPlan->inStride  = padmulPlan->outStride;
    fftiPlan->iDist     = padmulPlan->oDist;
    fftiPlan->outStride = fftiPlan->inStride;
    fftiPlan->oDist     = fftiPlan->iDist;

    fftiPlan->TraverseTreeAssignParamsLogicA();

    fftcPlan->inStride  = chirpPlan->outStride;
    fftcPlan->iDist     = chirpPlan->oDist;
    fftcPlan->outStride = fftcPlan->inStride;
    fftcPlan->oDist     = fftcPlan->iDist;

    fftcPlan->TraverseTreeAssignParamsLogicA();

    fftmulPlan->inStride  = fftiPlan->outStride;
    fftmulPlan->iDist     = fftiPlan->oDist;
    fftmulPlan->outStride = fftmulPlan->inStride;
    fftmulPlan->oDist     = fftmulPlan->iDist;

    fftrPlan->inStride  = fftmulPlan->outStride;
    fftrPlan->iDist     = fftmulPlan->oDist;
    fftrPlan->outStride = fftrPlan->inStride;
    fftrPlan->oDist     = fftrPlan->iDist;

    fftrPlan->TraverseTreeAssignParamsLogicA();

    resmulPlan->inStride  = fftrPlan->outStride;
    resmulPlan->iDist     = fftrPlan->oDist;
    resmulPlan->outStride = outStride;
    resmulPlan->oDist     = oDist;
}

void TreeNode::assign_params_CS_L1D_TRTRT()
{
    size_t biggerDim = childNodes[0]->length[0] > childNodes[0]->length[1]
                           ? childNodes[0]->length[0]
                           : childNodes[0]->length[1];
    size_t smallerDim = biggerDim == childNodes[0]->length[0] ? childNodes[0]->length[1]
                                                              : childNodes[0]->length[0];
    size_t padding = 0;
    if(((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512))
        padding = 64;

    TreeNode* trans1Plan = childNodes[0];
    TreeNode* row1Plan   = childNodes[1];
    TreeNode* trans2Plan = childNodes[2];
    TreeNode* row2Plan   = childNodes[3];
    TreeNode* trans3Plan = childNodes[4];

    trans1Plan->inStride.push_back(inStride[0]);
    trans1Plan->inStride.push_back(trans1Plan->length[0]);
    trans1Plan->iDist = iDist;
    for(size_t index = 1; index < length.size(); index++)
        trans1Plan->inStride.push_back(inStride[index]);

    if(trans1Plan->obOut == OB_TEMP)
    {
        trans1Plan->outStride.push_back(1);
        trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
        trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            trans1Plan->outStride.push_back(trans1Plan->oDist);
            trans1Plan->oDist *= length[index];
        }
    }
    else
    {
        trans1Plan->transTileDir = TTD_IP_VER;

        if(parent->scheme == CS_L1D_TRTRT)
        {
            trans1Plan->outStride.push_back(outStride[0]);
            trans1Plan->outStride.push_back(outStride[0] * (trans1Plan->length[1]));
            trans1Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                trans1Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            assert((parent->obOut == OB_USER_OUT) || (parent->obOut == OB_TEMP_CMPLX_FOR_REAL));

            assert(parent->outStride[0] == 1);
            for(size_t index = 1; index < parent->length.size(); index++)
                assert(parent->outStride[index]
                       == (parent->outStride[index - 1] * parent->length[index - 1]));

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1]);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                trans1Plan->outStride.push_back(trans1Plan->oDist);
                trans1Plan->oDist *= length[index];
            }
        }
    }

    row1Plan->inStride = trans1Plan->outStride;
    row1Plan->iDist    = trans1Plan->oDist;

    if(row1Plan->placement == rocfft_placement_inplace)
    {
        row1Plan->outStride = row1Plan->inStride;
        row1Plan->oDist     = row1Plan->iDist;
    }
    else
    {
        // TODO: add documentation for assert.
        assert((row1Plan->obOut == OB_USER_IN) || (row1Plan->obOut == OB_USER_OUT)
               || (row1Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
               || (row1Plan->obOut == OB_TEMP_BLUESTEIN));

        row1Plan->outStride.push_back(outStride[0]);
        row1Plan->outStride.push_back(outStride[0] * row1Plan->length[0]);
        row1Plan->oDist = oDist;

        for(size_t index = 1; index < length.size(); index++)
            row1Plan->outStride.push_back(outStride[index]);
    }

    row1Plan->TraverseTreeAssignParamsLogicA();

    trans2Plan->inStride = row1Plan->outStride;
    trans2Plan->iDist    = row1Plan->oDist;

    if(trans2Plan->obOut == OB_TEMP)
    {
        trans2Plan->outStride.push_back(1);
        trans2Plan->outStride.push_back(trans2Plan->length[1] + padding);
        trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            trans2Plan->outStride.push_back(trans2Plan->oDist);
            trans2Plan->oDist *= length[index];
        }
    }
    else
    {
        trans2Plan->transTileDir = TTD_IP_VER;

        if((parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)))
        {
            trans2Plan->outStride.push_back(outStride[0]);
            trans2Plan->outStride.push_back(outStride[0] * (trans2Plan->length[1]));
            trans2Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                trans2Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            trans2Plan->outStride.push_back(1);
            trans2Plan->outStride.push_back(trans2Plan->length[1]);
            trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                trans2Plan->outStride.push_back(trans2Plan->oDist);
                trans2Plan->oDist *= length[index];
            }
        }
    }

    row2Plan->inStride = trans2Plan->outStride;
    row2Plan->iDist    = trans2Plan->oDist;

    if(row2Plan->obIn == row2Plan->obOut)
    {
        row2Plan->outStride = row2Plan->inStride;
        row2Plan->oDist     = row2Plan->iDist;
    }
    else if(row2Plan->obOut == OB_TEMP)
    {
        row2Plan->outStride.push_back(1);
        row2Plan->outStride.push_back(row2Plan->length[0] + padding);
        row2Plan->oDist = row2Plan->length[1] * row2Plan->outStride[1];

        for(size_t index = 1; index < length.size(); index++)
        {
            row2Plan->outStride.push_back(row2Plan->oDist);
            row2Plan->oDist *= length[index];
        }
    }
    else
    {
        if((parent == NULL) || (parent && (parent->scheme == CS_L1D_TRTRT)))
        {
            row2Plan->outStride.push_back(outStride[0]);
            row2Plan->outStride.push_back(outStride[0] * (row2Plan->length[0]));
            row2Plan->oDist = oDist;

            for(size_t index = 1; index < length.size(); index++)
                row2Plan->outStride.push_back(outStride[index]);
        }
        else
        {
            // we dont have B info here, need to assume packed data and descended
            // from 2D/3D
            row2Plan->outStride.push_back(1);
            row2Plan->outStride.push_back(row2Plan->length[0]);
            row2Plan->oDist = row2Plan->length[0] * row2Plan->length[1];

            for(size_t index = 1; index < length.size(); index++)
            {
                row2Plan->outStride.push_back(row2Plan->oDist);
                row2Plan->oDist *= length[index];
            }
        }
    }

    if(trans3Plan->obOut != OB_TEMP)
        trans3Plan->transTileDir = TTD_IP_VER;

    trans3Plan->inStride = row2Plan->outStride;
    trans3Plan->iDist    = row2Plan->oDist;

    trans3Plan->outStride.push_back(outStride[0]);
    trans3Plan->outStride.push_back(outStride[0] * (trans3Plan->length[1]));
    trans3Plan->oDist = oDist;

    for(size_t index = 1; index < length.size(); index++)
        trans3Plan->outStride.push_back(outStride[index]);
}

void TreeNode::assign_params_CS_2D_RTRT()
{
    TreeNode* row1Plan   = childNodes[0];
    TreeNode* trans1Plan = childNodes[1];
    TreeNode* row2Plan   = childNodes[2];
    TreeNode* trans2Plan = childNodes[3];

    size_t biggerDim  = length[0] > length[1] ? length[0] : length[1];
    size_t smallerDim = biggerDim == length[0] ? length[1] : length[0];
    size_t padding    = 0;
    if(((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512))
        padding = 64;

    row1Plan->inStride  = inStride;
    row1Plan->iDist     = iDist;
    row1Plan->outStride = outStride;
    row1Plan->oDist     = oDist;
    row1Plan->TraverseTreeAssignParamsLogicA();

    // B -> T
    assert(trans1Plan->obOut == OB_TEMP);
    trans1Plan->inStride = row1Plan->outStride;
    trans1Plan->iDist    = row1Plan->oDist;
    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
    trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    assert(row2Plan->obOut == OB_TEMP);
    row2Plan->inStride  = trans1Plan->outStride;
    row2Plan->iDist     = trans1Plan->oDist;
    row2Plan->outStride = row2Plan->inStride;
    row2Plan->oDist     = row2Plan->iDist;
    row2Plan->TraverseTreeAssignParamsLogicA();

    // T -> B
    trans2Plan->inStride  = row2Plan->outStride;
    trans2Plan->iDist     = row2Plan->oDist;
    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

void TreeNode::assign_params_CS_REAL_2D_EVEN()
{
    const size_t biggerDim  = std::max(length[0], length[1]);
    const size_t smallerDim = std::min(length[0], length[1]);
    const size_t padding
        = (((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512)) ? 64 : 0;

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto row1Plan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            row1Plan->inStride = inStride;
            row1Plan->iDist    = iDist;

            row1Plan->outStride = outStride;
            row1Plan->oDist     = oDist;

            row1Plan->TraverseTreeAssignParamsLogicA();
        }

        auto trans1Plan = childNodes[1];
        {
            // B -> T
            trans1Plan->inStride = row1Plan->outStride;
            trans1Plan->iDist    = row1Plan->oDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }

        auto row2Plan = childNodes[2];
        {
            // T -> T
            row2Plan->inStride = trans1Plan->outStride;
            row2Plan->iDist    = trans1Plan->oDist;

            row2Plan->outStride = row2Plan->inStride;
            row2Plan->oDist     = row2Plan->iDist;

            row2Plan->TraverseTreeAssignParamsLogicA();
        }

        auto trans2Plan = childNodes[3];
        {
            // T -> B
            trans2Plan->inStride = row2Plan->outStride;
            trans2Plan->iDist    = row2Plan->oDist;

            trans2Plan->outStride = outStride;
            trans2Plan->oDist     = oDist;
        }
    }
    else
    {
        auto trans1Plan = childNodes[0];
        {
            trans1Plan->inStride = inStride;
            trans1Plan->iDist    = iDist;

            trans1Plan->outStride.push_back(1);
            trans1Plan->outStride.push_back(trans1Plan->length[1] + padding);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[1];
        }
        auto c2cPlan = childNodes[1];
        {
            c2cPlan->inStride = trans1Plan->outStride;
            c2cPlan->iDist    = trans1Plan->oDist;

            c2cPlan->outStride = c2cPlan->inStride;
            c2cPlan->oDist     = c2cPlan->iDist;

            c2cPlan->TraverseTreeAssignParamsLogicA();
        }
        auto trans2Plan = childNodes[2];
        {
            trans2Plan->inStride = trans1Plan->outStride;
            trans2Plan->iDist    = trans1Plan->oDist;

            trans2Plan->outStride = trans1Plan->inStride;
            trans2Plan->oDist     = trans2Plan->length[0] * trans2Plan->outStride[1];
        }
        auto c2rPlan = childNodes[3];
        {
            c2rPlan->inStride = trans2Plan->outStride;
            c2rPlan->iDist    = trans2Plan->oDist;

            c2rPlan->outStride = outStride;
            c2rPlan->oDist     = oDist;

            c2rPlan->TraverseTreeAssignParamsLogicA();
        }
    }
}

void TreeNode::assign_params_CS_2D_RC_STRAIGHT()
{
    TreeNode* rowPlan = childNodes[0];
    TreeNode* colPlan = childNodes[1];

    // B -> B
    // assert((rowPlan->obOut == OB_USER_OUT) || (rowPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
    //        || (rowPlan->obOut == OB_TEMP_BLUESTEIN));
    rowPlan->inStride = inStride;
    rowPlan->iDist    = iDist;

    rowPlan->outStride = outStride;
    rowPlan->oDist     = oDist;

    rowPlan->TraverseTreeAssignParamsLogicA();

    // B -> B
    assert((colPlan->obOut == OB_USER_OUT) || (colPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (colPlan->obOut == OB_TEMP_BLUESTEIN));
    colPlan->inStride.push_back(inStride[1]);
    colPlan->inStride.push_back(inStride[0]);
    for(size_t index = 2; index < length.size(); index++)
        colPlan->inStride.push_back(inStride[index]);

    colPlan->iDist = rowPlan->oDist;

    colPlan->outStride = colPlan->inStride;
    colPlan->oDist     = colPlan->iDist;
}

void TreeNode::assign_params_CS_3D_RTRT()
{
    TreeNode* xyPlan     = childNodes[0];
    TreeNode* trans1Plan = childNodes[1];
    TreeNode* zPlan      = childNodes[2];
    TreeNode* trans2Plan = childNodes[3];

    size_t biggerDim  = (length[0] * length[1]) > length[2] ? (length[0] * length[1]) : length[2];
    size_t smallerDim = biggerDim == (length[0] * length[1]) ? length[2] : (length[0] * length[1]);
    size_t padding    = 0;
    if(((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512))
        padding = 64;

    // B -> B
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->TraverseTreeAssignParamsLogicA();

    // B -> T
    assert(trans1Plan->obOut == OB_TEMP);
    trans1Plan->inStride = xyPlan->outStride;
    trans1Plan->iDist    = xyPlan->oDist;

    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[2] + padding);
    trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[1]);
    trans1Plan->oDist = trans1Plan->length[1] * trans1Plan->outStride[2];

    for(size_t index = 3; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    assert(zPlan->obOut == OB_TEMP);
    zPlan->inStride = trans1Plan->outStride;
    zPlan->iDist    = trans1Plan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->TraverseTreeAssignParamsLogicA();

    // T -> B
    assert((trans2Plan->obOut == OB_USER_OUT) || (trans2Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (trans2Plan->obOut == OB_TEMP_BLUESTEIN));
    trans2Plan->inStride = zPlan->outStride;
    trans2Plan->iDist    = zPlan->oDist;

    trans2Plan->outStride = outStride;
    trans2Plan->oDist     = oDist;
}

void TreeNode::assign_params_CS_3D_RC_STRAIGHT()
{
    TreeNode* xyPlan = childNodes[0];
    TreeNode* zPlan  = childNodes[1];

    // B -> B
    assert((xyPlan->obOut == OB_USER_OUT) || (xyPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (xyPlan->obOut == OB_TEMP_BLUESTEIN));
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->TraverseTreeAssignParamsLogicA();

    // B -> B
    assert((zPlan->obOut == OB_USER_OUT) || (zPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
           || (zPlan->obOut == OB_TEMP_BLUESTEIN));
    zPlan->inStride.push_back(inStride[2]);
    zPlan->inStride.push_back(inStride[0]);
    zPlan->inStride.push_back(inStride[1]);
    for(size_t index = 3; index < length.size(); index++)
        zPlan->inStride.push_back(inStride[index]);

    zPlan->iDist = xyPlan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;
}

// Set strides and distances
void TreeNode::TraverseTreeAssignParamsLogicA()
{
#if 0
    // Debug output information
    auto        here = this;
    auto        up   = parent;
    std::string tabs;
    while(up != nullptr && here != up)
    {
        here = up;
        up   = parent->parent;
        tabs += "\t";
    }
    std::cout << tabs << "TraverseTreeAssignParamsLogicA: " << PrintScheme(scheme) << std::endl;
    std::cout << tabs << "\tlength:";
    for(auto i : length)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << tabs << "\tistride:";
    for(auto i : inStride)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << tabs << "\tostride:";
    for(auto i : outStride)
        std::cout << i << " ";
    std::cout << std::endl;
#endif

    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());

    switch(scheme)
    {
    case CS_REAL_TRANSFORM_USING_CMPLX:
        assign_params_CS_REAL_TRANSFORM_USING_CMPLX();
        break;
    case CS_REAL_TRANSFORM_EVEN:
        assign_params_CS_REAL_TRANSFORM_EVEN();
        break;
    case CS_REAL_2D_EVEN:
        assign_params_CS_REAL_2D_EVEN();
        break;
    case CS_BLUESTEIN:
        assign_params_CS_BLUESTEIN();
        break;
    case CS_L1D_TRTRT:
        assign_params_CS_L1D_TRTRT();
        break;
    case CS_L1D_CC:
        assign_params_CS_L1D_CC();
        break;
    case CS_L1D_CRT:
        assign_params_CS_L1D_CRT();
        break;
    case CS_2D_RTRT:
        assign_params_CS_2D_RTRT();
        break;
    case CS_2D_RC:
    case CS_2D_STRAIGHT:
        assign_params_CS_2D_RC_STRAIGHT();
        break;
    case CS_3D_RTRT:
        assign_params_CS_3D_RTRT();
        break;
    case CS_3D_RC:
    case CS_3D_STRAIGHT:
        assign_params_CS_3D_RC_STRAIGHT();
        break;
    default:
        return;
    }
}

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

void TreeNode::Print(std::ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << std::endl << indentStr.c_str() << "scheme: " << PrintScheme(scheme).c_str();
    os << std::endl << indentStr.c_str();
    os << "dimension: " << dimension;
    os << std::endl << indentStr.c_str();
    os << "batch: " << batch;
    os << std::endl << indentStr.c_str();
    os << "length: ";
    for(size_t i = 0; i < length.size(); i++)
    {
        os << length[i] << " ";
    }

    os << std::endl << indentStr.c_str() << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << " ";

    os << std::endl << indentStr.c_str() << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << " ";

    os << std::endl << indentStr.c_str();
    os << "iOffset: " << iOffset;
    os << std::endl << indentStr.c_str();
    os << "oOffset: " << oOffset;

    os << std::endl << indentStr.c_str();
    os << "iDist: " << iDist;
    os << std::endl << indentStr.c_str();
    os << "oDist: " << oDist;

    os << std::endl << indentStr.c_str();
    os << "direction: " << direction;

    os << std::endl << indentStr.c_str();
    os << ((placement == rocfft_placement_inplace) ? "inplace" : "not inplace");

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
    os << std::endl << indentStr.c_str() << "TTD: " << transTileDir;
    os << std::endl << indentStr.c_str() << "large1D: " << large1D;
    os << std::endl << indentStr.c_str() << "lengthBlue: " << lengthBlue << std::endl;

    os << indentStr << PrintOperatingBuffer(obIn) << " -> " << PrintOperatingBuffer(obOut)
       << std::endl;
    os << indentStr << PrintOperatingBufferCode(obIn) << " -> " << PrintOperatingBufferCode(obOut)
       << std::endl;

    if(childNodes.size())
    {
        std::vector<TreeNode*>::const_iterator children_p;
        for(children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
        {
            (*children_p)->Print(os, indent + 1);
        }
    }
}

void ProcessNode(ExecPlan& execPlan)
{
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->dimension);

    execPlan.rootPlan->RecursiveBuildTree();

    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->inStride.size());
    assert(execPlan.rootPlan->length.size() == execPlan.rootPlan->outStride.size());

    OperatingBuffer flipIn = OB_UNINIT, flipOut = OB_UNINIT, obOutBuf = OB_UNINIT;
    execPlan.rootPlan->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

    execPlan.rootPlan->TraverseTreeAssignPlacementsLogicA(execPlan.rootPlan->inArrayType,
                                                          execPlan.rootPlan->outArrayType);
    execPlan.rootPlan->TraverseTreeAssignParamsLogicA();

    size_t tmpBufSize       = 0;
    size_t cmplxForRealSize = 0;
    size_t blueSize         = 0;
    size_t chirpSize        = 0;
    execPlan.rootPlan->TraverseTreeCollectLeafsLogicA(
        execPlan.execSeq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
    execPlan.workBufSize      = tmpBufSize + cmplxForRealSize + blueSize + chirpSize;
    execPlan.tmpWorkBufSize   = tmpBufSize;
    execPlan.copyWorkBufSize  = cmplxForRealSize;
    execPlan.blueWorkBufSize  = blueSize;
    execPlan.chirpWorkBufSize = chirpSize;
}

void PrintNode(std::ostream& os, const ExecPlan& execPlan)
{
    os << "**********************************************************************"
          "*********"
       << std::endl;

    size_t N = execPlan.rootPlan->batch;
    for(size_t i = 0; i < execPlan.rootPlan->length.size(); i++)
        N *= execPlan.rootPlan->length[i];
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

    os << "======================================================================"
          "========="
       << std::endl
       << std::endl;
}
