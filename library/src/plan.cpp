/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "plan.h"
#include "logging.h"
#include "private.h"
#include "radix_table.h"
#include "repo.h"
#include "rocfft.h"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
// clang-format off
#define VERSION_STRING (TO_STR(rocfft_version_major) "." \
                        TO_STR(rocfft_version_minor) "." \
                        TO_STR(rocfft_version_patch) "." \
                        TO_STR(rocfft_version_tweak) "-" \
			TO_STR(rocfft_version_commit_id))
// clang-format on
rocfft_status rocfft_plan_description_set_scale_float(rocfft_plan_description description,
                                                      float                   scale)
{
    description->scale = scale;

    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_scale_double(rocfft_plan_description description,
                                                       double                  scale)
{
    description->scale = scale;

    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      rocfft_array_type       in_array_type,
                                                      rocfft_array_type       out_array_type,
                                                      const size_t*           in_offsets,
                                                      const size_t*           out_offsets,
                                                      size_t                  in_strides_size,
                                                      const size_t*           in_strides,
                                                      size_t                  in_distance,
                                                      size_t                  out_strides_size,
                                                      const size_t*           out_strides,
                                                      size_t                  out_distance)
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
        for(size_t i = 0; i < MIN(3, in_strides_size); i++)
            description->inStrides[i] = in_strides[i];
    }

    if(in_distance != 0)
        description->inDist = in_distance;

    if(out_strides != nullptr)
    {
        for(size_t i = 0; i < MIN(3, out_strides_size); i++)
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
                                          rocfft_result_placement       placement,
                                          rocfft_transform_type         transform_type,
                                          rocfft_precision              precision,
                                          size_t                        dimensions,
                                          const size_t*                 lengths,
                                          size_t                        number_of_transforms,
                                          const rocfft_plan_description description,
                                          bool                          dry_run)
{
    // Initialize plan's parameters, no computation
    if(description != nullptr)
    {
        switch(transform_type)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
        {
            if(placement == rocfft_placement_inplace)
            {
                if(description->inArrayType == rocfft_array_type_complex_interleaved)
                {
                    if(description->outArrayType != rocfft_array_type_complex_interleaved)
                        return rocfft_status_invalid_array_type;
                }
                else if(description->inArrayType == rocfft_array_type_complex_planar)
                {
                    if(description->outArrayType != rocfft_array_type_complex_planar)
                        return rocfft_status_invalid_array_type;
                }
                else
                    return rocfft_status_invalid_array_type;
            }
            else
            {
                if(((description->inArrayType == rocfft_array_type_complex_interleaved)
                    || (description->inArrayType == rocfft_array_type_complex_planar)))
                {
                    if(!((description->outArrayType == rocfft_array_type_complex_interleaved)
                         || (description->outArrayType == rocfft_array_type_complex_planar)))
                        return rocfft_status_invalid_array_type;
                }
                else
                    return rocfft_status_invalid_array_type;
            }
        }
        break;
        case rocfft_transform_type_real_forward:
        {
            if(description->inArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;

            if(placement == rocfft_placement_inplace)
            {
                if(description->outArrayType != rocfft_array_type_hermitian_interleaved)
                    return rocfft_status_invalid_array_type;
            }
            else
            {
                if(!((description->outArrayType == rocfft_array_type_hermitian_interleaved)
                     || (description->outArrayType == rocfft_array_type_hermitian_planar)))
                    return rocfft_status_invalid_array_type;
            }
        }
        break;
        case rocfft_transform_type_real_inverse:
        {
            if(description->outArrayType != rocfft_array_type_real)
                return rocfft_status_invalid_array_type;

            if(placement == rocfft_placement_inplace)
            {
                if(description->inArrayType != rocfft_array_type_hermitian_interleaved)
                    return rocfft_status_invalid_array_type;
            }
            else
            {
                if(!((description->inArrayType == rocfft_array_type_hermitian_interleaved)
                     || (description->inArrayType == rocfft_array_type_hermitian_planar)))
                    return rocfft_status_invalid_array_type;
            }
        }
        break;
        }

        if((placement == rocfft_placement_inplace)
           && ((transform_type == rocfft_transform_type_complex_forward)
               || (transform_type == rocfft_transform_type_complex_inverse)))
        {
            for(size_t i = 0; i < 3; i++)
                if(description->inStrides[i] != description->outStrides[i])
                    return rocfft_status_invalid_strides;

            if(description->inDist != description->outDist)
                return rocfft_status_invalid_distance;

            for(size_t i = 0; i < 2; i++)
                if(description->inOffset[i] != description->outOffset[i])
                    return rocfft_status_invalid_offset;
        }
    }

    if(dimensions > 3)
        return rocfft_status_invalid_dimensions;

    rocfft_plan p = plan;
    // problem dimensions specified by user
    p->rank = dimensions;

    size_t prodLength = 1;
    for(size_t i = 0; i < (p->rank); i++)
    {
        prodLength *= lengths[i];
        p->lengths[i] = lengths[i];
    }

    p->batch     = number_of_transforms;
    p->placement = placement;
    p->precision = precision;
    if(precision == rocfft_precision_double)
    {
        p->base_type_size = sizeof(double);
    }
    else
    {
        p->base_type_size = sizeof(float);
    }
    p->transformType = transform_type;

    if(description != nullptr)
        p->desc = *description;
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
        {
            p->desc.inArrayType  = rocfft_array_type_real;
            p->desc.outArrayType = rocfft_array_type_hermitian_interleaved;
        }
        break;
        case rocfft_transform_type_real_inverse:
        {
            p->desc.inArrayType  = rocfft_array_type_hermitian_interleaved;
            p->desc.outArrayType = rocfft_array_type_real;
        }
        break;
        }
    }

    if(p->desc.inStrides[0] == 0)
    {
        p->desc.inStrides[0] = 1;

        if(p->transformType == rocfft_transform_type_real_inverse)
        {
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.inStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.inDist == 0)
                p->desc.inDist = dist;
        }
        else if((p->transformType == rocfft_transform_type_real_forward)
                && (p->placement == rocfft_placement_inplace))
        {
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

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
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.inStrides[i] = p->lengths[i - 1] * p->desc.inStrides[i - 1];
        }
    }

    if(p->desc.outStrides[0] == 0)
    {
        p->desc.outStrides[0] = 1;

        if(p->transformType == rocfft_transform_type_real_forward)
        {
            size_t dist = 1 + (p->lengths[0]) / 2;

            for(size_t i = 1; i < (p->rank); i++)
            {
                p->desc.outStrides[i] = dist;
                dist *= p->lengths[i];
            }

            if(p->desc.outDist == 0)
                p->desc.outDist = dist;
        }
        else if((p->transformType == rocfft_transform_type_real_inverse)
                && (p->placement == rocfft_placement_inplace))
        {
            size_t dist = 2 * (1 + (p->lengths[0]) / 2);

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
            for(size_t i = 1; i < (p->rank); i++)
                p->desc.outStrides[i] = p->lengths[i - 1] * p->desc.outStrides[i - 1];
        }
    }

    if(p->desc.inDist == 0)
    {
        p->desc.inDist = p->lengths[p->rank - 1] * p->desc.inStrides[p->rank - 1];
    }

    if(p->desc.outDist == 0)
    {
        p->desc.outDist = p->lengths[p->rank - 1] * p->desc.outStrides[p->rank - 1];
    }

    /*if(!SupportedLength(prodLength))
  {
      printf("This size %zu is not supported in rocFFT, will return;\n",
  prodLength);
      return rocfft_status_invalid_dimensions;
  }*/
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
                                 rocfft_result_placement       placement,
                                 rocfft_transform_type         transform_type,
                                 rocfft_precision              precision,
                                 size_t                        dimensions,
                                 const size_t*                 lengths,
                                 size_t                        number_of_transforms,
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

ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, size_t len)
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

std::string PrintScheme(ComputeScheme cs)
{
    std::string str;

    switch(cs)
    {
    case CS_KERNEL_STOCKHAM:
        str += "CS_KERNEL_STOCKHAM";
        break;
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
        str += "CS_KERNEL_STOCKHAM_BLOCK_CC";
        break;
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
        str += "CS_KERNEL_STOCKHAM_BLOCK_RC";
        break;
    case CS_KERNEL_TRANSPOSE:
        str += "CS_KERNEL_TRANSPOSE";
        break;
    case CS_KERNEL_TRANSPOSE_XY_Z:
        str += "CS_KERNEL_TRANSPOSE_XY_Z";
        break;
    case CS_KERNEL_TRANSPOSE_Z_XY:
        str += "CS_KERNEL_TRANSPOSE_Z_XY";
        break;
    case CS_REAL_TRANSFORM_USING_CMPLX:
        str += "CS_REAL_TRANSFORM_USING_CMPLX";
        break;
    case CS_KERNEL_COPY_R_TO_CMPLX:
        str += "CS_KERNEL_COPY_R_TO_CMPLX";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
        str += "CS_KERNEL_COPY_CMPLX_TO_HERM";
        break;
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
        str += "CS_KERNEL_COPY_HERM_TO_CMPLX";
        break;
    case CS_KERNEL_COPY_CMPLX_TO_R:
        str += "CS_KERNEL_COPY_CMPLX_TO_R";
        break;
    case CS_BLUESTEIN:
        str += "CS_BLUESTEIN";
        break;
    case CS_KERNEL_CHIRP:
        str += "CS_KERNEL_CHIRP";
        break;
    case CS_KERNEL_PAD_MUL:
        str += "CS_KERNEL_PAD_MUL";
        break;
    case CS_KERNEL_FFT_MUL:
        str += "CS_KERNEL_FFT_MUL";
        break;
    case CS_KERNEL_RES_MUL:
        str += "CS_KERNEL_RES_MUL";
        break;
    case CS_L1D_TRTRT:
        str += "CS_L1D_TRTRT";
        break;
    case CS_L1D_CC:
        str += "CS_L1D_CC";
        break;
    case CS_L1D_CRT:
        str += "CS_L1D_CRT";
        break;
    case CS_2D_STRAIGHT:
        str += "CS_2D_STRAIGHT";
        break;
    case CS_2D_RTRT:
        str += "CS_2D_RTRT";
        break;
    case CS_2D_RC:
        str += "CS_2D_RC";
        break;
    case CS_KERNEL_2D_STOCKHAM_BLOCK_CC:
        str += "CS_KERNEL_2D_STOCKHAM_BLOCK_CC";
        break;
    case CS_KERNEL_2D_SINGLE:
        str += "CS_KERNEL_2D_SINGLE";
        break;
    case CS_3D_STRAIGHT:
        str += "CS_3D_STRAIGHT";
        break;
    case CS_3D_RTRT:
        str += "CS_3D_RTRT";
        break;
    case CS_3D_RC:
        str += "CS_3D_RC";
        break;
    case CS_KERNEL_3D_STOCKHAM_BLOCK_CC:
        str += "CS_KERNEL_3D_STOCKHAM_BLOCK_CC";
        break;
    case CS_KERNEL_3D_SINGLE:
        str += "CS_KERNEL_3D_SINGLE";
        break;

    default:
        str += "CS_NONE";
        break;
    }

    return str;
}

void TreeNode::RecursiveBuildTree()
{
    // this flag can be enabled when generator can do block column fft in
    // multi-dimension cases and small 2d, 3d within one kernel
    bool MultiDimFuseKernelsAvailable = false;

    if((parent == nullptr)
       && ((inArrayType == rocfft_array_type_real) || (outArrayType == rocfft_array_type_real)))
    {
        scheme = CS_REAL_TRANSFORM_USING_CMPLX;

        TreeNode* copyHeadPlan = TreeNode::CreateNode(this);

        // head copy plan
        copyHeadPlan->dimension = dimension;
        copyHeadPlan->length    = length;

        if(inArrayType == rocfft_array_type_real)
            copyHeadPlan->scheme = CS_KERNEL_COPY_R_TO_CMPLX;
        else if(outArrayType == rocfft_array_type_real)
            copyHeadPlan->scheme = CS_KERNEL_COPY_HERM_TO_CMPLX;

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

        if(inArrayType == rocfft_array_type_real)
            copyTailPlan->scheme = CS_KERNEL_COPY_CMPLX_TO_HERM;
        else if(outArrayType == rocfft_array_type_real)
            copyTailPlan->scheme = CS_KERNEL_COPY_CMPLX_TO_R;

        childNodes.push_back(copyTailPlan);

        return;
    }

    switch(dimension)
    {
    case 1:
    {
        if(!SupportedLength(length[0]))
        {
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
            // Enable block compute under these conditions
            if(length[0] <= 262144 / PrecisionWidth(precision))
            {
                if(1 == PrecisionWidth(precision))
                {
                    switch(length[0])
                    {
                    case 8192:
                        divLength1 = 64;
                        break;
                    case 16384:
                        divLength1 = 64;
                        break;
                    case 32768:
                        divLength1 = 128;
                        break;
                    case 65536:
                        divLength1 = 256;
                        break;
                    case 131072:
                        divLength1 = 64;
                        break;
                    case 262144:
                        divLength1 = 64;
                        break;
                    default:
                        assert(false);
                    }
                }
                else
                {
                    switch(length[0])
                    {
                    case 4096:
                        divLength1 = 64;
                        break;
                    case 8192:
                        divLength1 = 64;
                        break;
                    case 16384:
                        divLength1 = 64;
                        break;
                    case 32768:
                        divLength1 = 128;
                        break;
                    case 65536:
                        divLength1 = 64;
                        break;
                    case 131072:
                        divLength1 = 64;
                        break;
                    default:
                        assert(false);
                    }
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
            size_t supported[] = {
                4096, 4050, 4000, 3888, 3840, 3750, 3645, 3600, 3456, 3375, 3240, 3200, 3125, 3072,
                3000, 2916, 2880, 2700, 2592, 2560, 2500, 2430, 2400, 2304, 2250, 2187, 2160, 2048,
                2025, 2000, 1944, 1920, 1875, 1800, 1728, 1620, 1600, 1536, 1500, 1458, 1440, 1350,
                1296, 1280, 1250, 1215, 1200, 1152, 1125, 1080, 1024, 1000, 972,  960,  900,  864,
                810,  800,  768,  750,  729,  720,  675,  648,  640,  625,  600,  576,  540,  512,
                500,  486,  480,  450,  432,  405,  400,  384,  375,  360,  324,  320,  300,  288,
                270,  256,  250,  243,  240,  225,  216,  200,  192,  180,  162,  160,  150,  144,
                135,  128,  125,  120,  108,  100,  96,   90,   81,   80,   75,   72,   64,   60,
                54,   50,   48,   45,   40,   36,   32,   30,   27,   25,   24,   20,   18,   16,
                15,   12,   10,   9,    8,    6,    5,    4,    3,    2,    1};

            size_t threshold_id = 0;
            while(supported[threshold_id] != Large1DThreshold(precision))
                threshold_id++;

            if(length[0] > (Large1DThreshold(precision) * Large1DThreshold(precision)))
            {
                size_t idx = threshold_id;
                while(length[0] % supported[idx] != 0)
                    idx++;

                divLength1 = length[0] / supported[idx];
            }
            else
            {
                // logic tries to break into as squarish matrix as possible
                size_t sqr = (size_t)sqrt(length[0]);
                size_t i   = sizeof(supported) / sizeof(supported[0]) - 1;
                while(supported[i] < sqr)
                    i--;
                while(length[0] % supported[i] != 0)
                    i++;

                divLength1 = length[0] / supported[i];
            }

            scheme = CS_L1D_TRTRT;
        }

        size_t divLength0 = length[0] / divLength1;

        switch(scheme)
        {
        case CS_L1D_TRTRT:
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
        break;
        case CS_L1D_CC:
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
        break;
        case CS_L1D_CRT:
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
        break;
        default:
            assert(false);
        }
    }
    break;

    case 2:
    {
        if(scheme == CS_KERNEL_TRANSPOSE)
            return;

        if(MultiDimFuseKernelsAvailable)
        {
            // conditions to choose which scheme
            if((length[0] * length[1]) <= 2048)
                scheme = CS_KERNEL_2D_SINGLE;
            else if(length[1] <= 256)
                scheme = CS_2D_RC;
            else
                scheme = CS_2D_RTRT;
        }
        else
            scheme = CS_2D_RTRT;

        switch(scheme)
        {
        case CS_2D_RTRT:
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

// logic A - using out-of-place transposes & complex-to-complex & with padding
void TreeNode::TraverseTreeAssignBuffersLogicA(OperatingBuffer& flipIn,
                                               OperatingBuffer& flipOut,
                                               OperatingBuffer& obOutBuf)
{
    if(parent == nullptr)
    {
        if(scheme == CS_REAL_TRANSFORM_USING_CMPLX)
        {
            flipIn  = OB_TEMP_CMPLX_FOR_REAL;
            flipOut = OB_TEMP;

            obOutBuf = OB_TEMP_CMPLX_FOR_REAL;
        }
        else if(scheme == CS_BLUESTEIN)
        {
            flipIn  = OB_TEMP_BLUESTEIN;
            flipOut = OB_TEMP;

            obOutBuf = OB_TEMP_BLUESTEIN;
        }
        else
        {
            flipIn  = OB_USER_OUT;
            flipOut = OB_TEMP;

            obOutBuf = OB_USER_OUT;
        }
    }

    if(scheme == CS_REAL_TRANSFORM_USING_CMPLX)
    {
        assert(parent == nullptr);
        childNodes[0]->obIn  = OB_USER_IN;
        childNodes[0]->obOut = OB_TEMP_CMPLX_FOR_REAL;

        childNodes[1]->obIn  = OB_TEMP_CMPLX_FOR_REAL;
        childNodes[1]->obOut = OB_TEMP_CMPLX_FOR_REAL;
        childNodes[1]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);
        size_t cs = childNodes[1]->childNodes.size();
        if(cs)
        {
            assert(childNodes[1]->childNodes[0]->obIn == OB_TEMP_CMPLX_FOR_REAL);
            assert(childNodes[1]->childNodes[cs - 1]->obOut == OB_TEMP_CMPLX_FOR_REAL);
        }

        childNodes[2]->obIn  = OB_TEMP_CMPLX_FOR_REAL;
        childNodes[2]->obOut = OB_USER_OUT;

        obIn  = childNodes[0]->obIn;
        obOut = childNodes[2]->obOut;
    }
    else if(scheme == CS_BLUESTEIN)
    {
        OperatingBuffer savFlipIn  = flipIn;
        OperatingBuffer savFlipOut = flipOut;
        OperatingBuffer savOutBuf  = obOutBuf;

        flipIn   = OB_TEMP_BLUESTEIN;
        flipOut  = OB_TEMP;
        obOutBuf = OB_TEMP_BLUESTEIN;

        childNodes[0]->obIn  = OB_TEMP_BLUESTEIN;
        childNodes[0]->obOut = OB_TEMP_BLUESTEIN;

        if(parent == nullptr)
        {
            childNodes[1]->obIn
                = (placement == rocfft_placement_inplace) ? OB_USER_OUT : OB_USER_IN;
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

        childNodes[4]->obIn  = OB_TEMP_BLUESTEIN;
        childNodes[4]->obOut = OB_TEMP_BLUESTEIN;

        childNodes[5]->obIn  = OB_TEMP_BLUESTEIN;
        childNodes[5]->obOut = OB_TEMP_BLUESTEIN;
        childNodes[5]->TraverseTreeAssignBuffersLogicA(flipIn, flipOut, obOutBuf);

        childNodes[6]->obIn = OB_TEMP_BLUESTEIN;

        if(parent == nullptr)
        {
            childNodes[6]->obOut = OB_USER_OUT;
        }
        else
        {
            childNodes[6]->obOut = obOut;
        }

        obIn  = childNodes[1]->obIn;
        obOut = childNodes[6]->obOut;

        flipIn   = savFlipIn;
        flipOut  = savFlipOut;
        obOutBuf = savOutBuf;
    }
    else if(scheme == CS_L1D_TRTRT)
    {
        if(parent == nullptr)
        {
            childNodes[0]->obIn = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
        }
        else
            childNodes[0]->obIn = flipIn;

        childNodes[0]->obOut = flipOut;

        OperatingBuffer t;
        t       = flipIn;
        flipIn  = flipOut;
        flipOut = t;

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
                OperatingBuffer t;
                t       = flipIn;
                flipIn  = flipOut;
                flipOut = t;
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
    else if(scheme == CS_L1D_CC)
    {
        if((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
        {
            if(parent == nullptr)
            {
                childNodes[0]->obIn
                    = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
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
            assert(obIn == flipIn);
            assert(obIn == obOut);

            childNodes[0]->obIn  = flipIn;
            childNodes[0]->obOut = flipOut;

            childNodes[1]->obIn  = flipOut;
            childNodes[1]->obOut = flipIn;
        }
    }
    else if(scheme == CS_L1D_CRT)
    {
        if((obIn == OB_UNINIT) && (obOut == OB_UNINIT))
        {
            if(parent == nullptr)
            {
                childNodes[0]->obIn
                    = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
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
    else if((scheme == CS_2D_RTRT) || (scheme == CS_3D_RTRT))
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
    else if((scheme == CS_2D_RC) || (scheme == CS_3D_RC))
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
    else
    {
        if(parent == nullptr)
        {
            obIn  = (placement == rocfft_placement_inplace) ? obOutBuf : OB_USER_IN;
            obOut = obOutBuf;
        }
        else
        {
            if((obIn == OB_UNINIT) || (obOut == OB_UNINIT))
                assert(false);

            if(obIn != obOut)
            {
                OperatingBuffer t;
                t       = flipIn;
                flipIn  = flipOut;
                flipOut = t;
            }
        }
    }
}

void TreeNode::TraverseTreeAssignPlacementsLogicA(rocfft_array_type rootIn,
                                                  rocfft_array_type rootOut)
{

    if(parent != nullptr)
    {
        placement = (obIn == obOut) ? rocfft_placement_inplace : rocfft_placement_notinplace;

        switch(obIn)
        {
        case OB_USER_IN:
            inArrayType = rootIn;
            break;
        case OB_USER_OUT:
            inArrayType = rootOut;
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

        switch(obOut)
        {
        case OB_USER_IN:
            assert(false);
            break;
        case OB_USER_OUT:
            outArrayType = rootOut;
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

    std::vector<TreeNode*>::iterator children_p;
    for(children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
    {
        (*children_p)->TraverseTreeAssignPlacementsLogicA(rootIn, rootOut);
    }
}

void TreeNode::TraverseTreeAssignParamsLogicA()
{
    switch(scheme)
    {
    case CS_REAL_TRANSFORM_USING_CMPLX:
    {
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
    break;
    case CS_BLUESTEIN:
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
    break;
    case CS_L1D_TRTRT:
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
            assert((row1Plan->obOut == OB_USER_OUT) || (row1Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
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
    break;
    case CS_L1D_CC:
    {
        TreeNode* col2colPlan = childNodes[0];
        TreeNode* row2colPlan = childNodes[1];

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
            assert((parent->obOut == OB_USER_OUT) || (parent->obOut == OB_TEMP_CMPLX_FOR_REAL));

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
    break;
    case CS_L1D_CRT:
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
    break;
    case CS_2D_RTRT:
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

        // B -> B
        assert((row1Plan->obOut == OB_USER_OUT) || (row1Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
               || (row1Plan->obOut == OB_TEMP_BLUESTEIN));
        row1Plan->inStride = inStride;
        row1Plan->iDist    = iDist;

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
        row2Plan->inStride = trans1Plan->outStride;
        row2Plan->iDist    = trans1Plan->oDist;

        row2Plan->outStride = row2Plan->inStride;
        row2Plan->oDist     = row2Plan->iDist;

        row2Plan->TraverseTreeAssignParamsLogicA();

        // T -> B
        assert((trans2Plan->obOut == OB_USER_OUT) || (trans2Plan->obOut == OB_TEMP_CMPLX_FOR_REAL)
               || (trans2Plan->obOut == OB_TEMP_BLUESTEIN));
        trans2Plan->inStride = row2Plan->outStride;
        trans2Plan->iDist    = row2Plan->oDist;

        trans2Plan->outStride = outStride;
        trans2Plan->oDist     = oDist;
    }
    break;
    case CS_2D_RC:
    case CS_2D_STRAIGHT:
    {
        TreeNode* rowPlan = childNodes[0];
        TreeNode* colPlan = childNodes[1];

        // B -> B
        assert((rowPlan->obOut == OB_USER_OUT) || (rowPlan->obOut == OB_TEMP_CMPLX_FOR_REAL)
               || (rowPlan->obOut == OB_TEMP_BLUESTEIN));
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
    };
    break;
    case CS_3D_RTRT:
    {
        TreeNode* xyPlan     = childNodes[0];
        TreeNode* trans1Plan = childNodes[1];
        TreeNode* zPlan      = childNodes[2];
        TreeNode* trans2Plan = childNodes[3];

        size_t biggerDim
            = (length[0] * length[1]) > length[2] ? (length[0] * length[1]) : length[2];
        size_t smallerDim
            = biggerDim == (length[0] * length[1]) ? length[2] : (length[0] * length[1]);
        size_t padding = 0;
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
    break;
    case CS_3D_RC:
    case CS_3D_STRAIGHT:
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
    };
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
        assert(length.size() == inStride.size());
        assert(length.size() == outStride.size());

        if(scheme == CS_KERNEL_CHIRP)
            chirpSize = (2 * lengthBlue) > chirpSize ? (2 * lengthBlue) : chirpSize;

        if(obOut == OB_TEMP_BLUESTEIN)
            blueSize = (oDist * batch) > blueSize ? (oDist * batch) : blueSize;
        if(obOut == OB_TEMP_CMPLX_FOR_REAL)
            cmplxForRealSize
                = (oDist * batch) > cmplxForRealSize ? (oDist * batch) : cmplxForRealSize;
        if(obOut == OB_TEMP)
            tmpBufSize = (oDist * batch) > tmpBufSize ? (oDist * batch) : tmpBufSize;
        seq.push_back(this);
    }
    else
    {
        std::vector<TreeNode*>::iterator children_p;
        for(children_p = childNodes.begin(); children_p != childNodes.end(); children_p++)
        {
            (*children_p)
                ->TraverseTreeCollectLeafsLogicA(
                    seq, tmpBufSize, cmplxForRealSize, blueSize, chirpSize);
        }
    }
}

void TreeNode::Print(std::ostream& os, int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << std::endl << indentStr.c_str();
    os << "dimension: " << dimension;
    os << std::endl << indentStr.c_str();
    os << "batch: " << batch;
    os << std::endl << indentStr.c_str();
    os << "length: " << length[0];
    for(size_t i = 1; i < length.size(); i++)
        os << ", " << length[i];

    os << std::endl << indentStr.c_str() << "iStrides: ";
    for(size_t i = 0; i < inStride.size(); i++)
        os << inStride[i] << ", ";
    os << iDist;

    os << std::endl << indentStr.c_str() << "oStrides: ";
    for(size_t i = 0; i < outStride.size(); i++)
        os << outStride[i] << ", ";
    os << oDist;

    os << std::endl << indentStr.c_str();
    os << "iOffset: " << iOffset;
    os << std::endl << indentStr.c_str();
    os << "oOffset: " << oOffset;

    os << std::endl << indentStr.c_str();
    os << "direction: " << direction;

    os << std::endl << indentStr.c_str();
    os << ((placement == rocfft_placement_inplace) ? "inplace" : "not inplace") << "  ";
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
    }
    os << std::endl << indentStr.c_str() << "scheme: " << PrintScheme(scheme).c_str();
    os << std::endl << indentStr.c_str() << "TTD: " << transTileDir;
    os << std::endl << indentStr.c_str() << "large1D: " << large1D;
    os << std::endl
       << indentStr.c_str() << "lengthBlue: " << lengthBlue << std::endl
       << indentStr.c_str();

    if(obIn == OB_USER_IN)
        os << "A -> ";
    else if(obIn == OB_USER_OUT)
        os << "B -> ";
    else if(obIn == OB_TEMP)
        os << "T -> ";
    else if(obIn == OB_TEMP_CMPLX_FOR_REAL)
        os << "C -> ";
    else if(obIn == OB_TEMP_BLUESTEIN)
        os << "S -> ";
    else
        os << "ERR -> ";

    if(obOut == OB_USER_IN)
        os << "A";
    else if(obOut == OB_USER_OUT)
        os << "B";
    else if(obOut == OB_TEMP)
        os << "T";
    else if(obOut == OB_TEMP_CMPLX_FOR_REAL)
        os << "C";
    else if(obOut == OB_TEMP_BLUESTEIN)
        os << "S";
    else
        os << "ERR";

    os << std::endl;

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
    OperatingBuffer flipIn, flipOut, obOutBuf;
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
                for(size_t i = 0; i < ((*curr_p)->inStride.size()); i++)
                {
                    if(((*curr_p)->inStride[i]) != ((*curr_p)->outStride[i]))
                        os << "error in stride assignments" << std::endl;
                    if(((*curr_p)->iDist) != ((*curr_p)->oDist))
                        os << "error in dist assignments" << std::endl;
                }
            }

            if(((*prev_p)->scheme != CS_KERNEL_CHIRP) && ((*curr_p)->scheme != CS_KERNEL_CHIRP))
                if((*prev_p)->obOut != (*curr_p)->obIn)
                    os << "error in buffer assignments" << std::endl;

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
