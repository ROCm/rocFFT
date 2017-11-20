/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <iostream>
#include "rocfft.h"
#include "private.h"
#include "hipfft.h"
#include "plan.h"

struct hipfftHandle_t
{
    rocfft_plan ip_forward;
    rocfft_plan op_forward;
    rocfft_plan ip_inverse;
    rocfft_plan op_inverse;

    hipfftHandle_t() :
        ip_forward(nullptr),
        op_forward(nullptr),
        ip_inverse(nullptr),
        op_inverse(nullptr)
    {}
};

hipfftResult hipfftPlan1d(hipfftHandle *plan,
                                 int nx,
                                 hipfftType type,
                                 int batch )
{
    hipfftHandle handle = nullptr;
    hipfftCreate(&handle);
    *plan = handle;

    return hipfftMakePlan1d(*plan, nx, type, batch, nullptr);
}

hipfftResult hipfftPlan2d(hipfftHandle *plan,
                                 int nx, int ny,
                                 hipfftType type)
{
    hipfftHandle handle = nullptr;
    hipfftCreate(&handle);
    *plan = handle;

    return hipfftMakePlan2d(*plan, nx, ny, type, nullptr);
}

hipfftResult hipfftPlan3d(hipfftHandle *plan,
                                 int nx, int ny, int nz,
                                 hipfftType type)
{
    hipfftHandle handle = nullptr;
    hipfftCreate(&handle);
    *plan = handle;

    return hipfftMakePlan3d(*plan, nx, ny, nz, type, nullptr);
}

hipfftResult hipfftPlanMany(hipfftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   hipfftType type,
                                   int batch)
{
    hipfftHandle handle = nullptr;
    hipfftCreate(&handle);
    *plan = handle;

    return hipfftMakePlanMany(*plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
}

hipfftResult hipfftMakePlan1d(hipfftHandle plan,
                                     int nx,
                                     hipfftType type,
                                     int batch,
                                     size_t *workSize)
{
    size_t lengths[1];
    lengths[0] = nx;
    size_t number_of_transforms = batch;

    switch(type)
    {
        case HIPFFT_R2C:
            break;
        case HIPFFT_C2R:
            break;
        case HIPFFT_C2C:
            rocfft_plan_create_internal(plan->ip_forward,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    1, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_forward,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    1, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->ip_inverse,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    1, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_inverse,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    1, lengths, number_of_transforms, nullptr);
            break;

        case HIPFFT_D2Z:
            break;
        case HIPFFT_Z2D:
            break;
        case HIPFFT_Z2Z:
            break;
        default:
            assert(false);
    }

    if(workSize != nullptr)
        rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan,
                                     int nx, int ny,
                                     hipfftType type,
                                     size_t *workSize)
{
    size_t lengths[2];
    lengths[0] = ny;
    lengths[1] = nx;
    size_t number_of_transforms = 1;

    switch(type)
    {
        case HIPFFT_R2C:
            break;
        case HIPFFT_C2R:
            break;
        case HIPFFT_C2C:
            rocfft_plan_create_internal(plan->ip_forward,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    2, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_forward,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    2, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->ip_inverse,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    2, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_inverse,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    2, lengths, number_of_transforms, nullptr);
            break;

        case HIPFFT_D2Z:
            break;
        case HIPFFT_Z2D:
            break;
        case HIPFFT_Z2Z:
            break;
        default:
            assert(false);
    }

    if(workSize != nullptr)
        rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftMakePlan3d(hipfftHandle plan,
                                     int nx, int ny, int nz,
                                     hipfftType type,
                                     size_t *workSize)
{
    size_t lengths[3];
    lengths[0] = nz;
    lengths[1] = ny;
    lengths[2] = nx;
    size_t number_of_transforms = 1;

    switch(type)
    {
        case HIPFFT_R2C:
            break;
        case HIPFFT_C2R:
            break;
        case HIPFFT_C2C:
            rocfft_plan_create_internal(plan->ip_forward,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    3, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_forward,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    3, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->ip_inverse,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    3, lengths, number_of_transforms, nullptr);
            rocfft_plan_create_internal(plan->op_inverse,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    3, lengths, number_of_transforms, nullptr);
            break;

        case HIPFFT_D2Z:
            break;
        case HIPFFT_Z2D:
            break;
        case HIPFFT_Z2Z:
            break;
        default:
            assert(false);
    }

    if(workSize != nullptr)
        rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                       int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       hipfftType type,
                                       int batch,
                                       size_t *workSize)
{
    size_t lengths[3];
    for(size_t i=0; i<rank; i++)
        lengths[i] = n[rank-1-i];

    size_t number_of_transforms = batch;

    rocfft_plan_description desc = nullptr;
    if( (inembed != nullptr) || (onembed != nullptr))
    {
        rocfft_plan_description_create(&desc);

        size_t i_strides[3] = {1,1,1};
        size_t o_strides[3] = {1,1,1};

        // todo: following logic only for complex-to-complex, todo real
        if(inembed == nullptr)
        {
            for(size_t i=1; i<rank; i++)
                i_strides[i] = lengths[i-1]*i_strides[i-1];

        }
        else
        {
            i_strides[0] = istride;

            size_t inembed_lengths[3];
            for(size_t i=0; i<rank; i++)
                inembed_lengths[i] = inembed[rank-1-i];

            for(size_t i=1; i<rank; i++)
                i_strides[i] = inembed_lengths[i-1]*i_strides[i-1];
        }

        if(onembed == nullptr)
        {
            for(size_t i=1; i<rank; i++)
                o_strides[i] = lengths[i-1]*o_strides[i-1];

        }
        else
        {
            o_strides[0] = ostride;

            size_t onembed_lengths[3];
            for(size_t i=0; i<rank; i++)
                onembed_lengths[i] = onembed[rank-1-i];

            for(size_t i=1; i<rank; i++)
                o_strides[i] = onembed_lengths[i-1]*o_strides[i-1];
        }

        rocfft_plan_description_set_data_layout( desc,  rocfft_array_type_complex_interleaved,
                                rocfft_array_type_complex_interleaved,
                                0, 0,
                                rank, i_strides, idist,
                                rank, o_strides, odist );
    }

    switch(type)
    {
        case HIPFFT_R2C:
            break;
        case HIPFFT_C2R:
            break;
        case HIPFFT_C2C:
            rocfft_plan_create_internal(plan->ip_forward,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    rank, lengths, number_of_transforms, desc);
            rocfft_plan_create_internal(plan->op_forward,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_forward,
                    rocfft_precision_single,
                    rank, lengths, number_of_transforms, desc);
            rocfft_plan_create_internal(plan->ip_inverse,
                    rocfft_placement_inplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    rank, lengths, number_of_transforms, desc);
            rocfft_plan_create_internal(plan->op_inverse,
                    rocfft_placement_notinplace,
                    rocfft_transform_type_complex_inverse,
                    rocfft_precision_single,
                    rank, lengths, number_of_transforms, desc);
            break;

        case HIPFFT_D2Z:
            break;
        case HIPFFT_Z2D:
            break;
        case HIPFFT_Z2Z:
            break;
        default:
            assert(false);
    }

    rocfft_plan_description_destroy(desc);

    if(workSize != nullptr)
        rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftMakePlanMany64(hipfftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         long long int *onembed,
                                         long long int ostride, long long int odist,
                                         hipfftType type,
                                         long long int batch,
                                         size_t * workSize)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSizeMany64(hipfftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride, long long int idist,
                                        long long int *onembed,
                                        long long int ostride, long long int odist,
                                        hipfftType type,
                                        long long int batch,
                                        size_t *workSize)
{
    return HIPFFT_SUCCESS;
}




hipfftResult hipfftEstimate1d(int nx,
                                     hipfftType type,
                                     int batch,
                                     size_t *workSize)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftEstimate2d(int nx, int ny,
                                     hipfftType type,
                                     size_t *workSize)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz,
                                     hipfftType type,
                                     size_t *workSize)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftEstimateMany(int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       hipfftType type,
                                       int batch,
                                       size_t *workSize)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftCreate(hipfftHandle * plan)
{
    hipfftHandle h = new hipfftHandle_t;

    rocfft_plan_allocate(&h->ip_forward);
    rocfft_plan_allocate(&h->op_forward);
    rocfft_plan_allocate(&h->ip_inverse);
    rocfft_plan_allocate(&h->op_inverse);

    *plan = h;

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize1d(hipfftHandle plan,
                                    int nx,
                                    hipfftType type,
                                    int batch,
                                    size_t *workSize )
{
    hipfftHandle p;
    hipfftPlan1d(&p, nx, type, batch);
    rocfft_plan_get_work_buffer_size(p->ip_forward, workSize);
    hipfftDestroy(p);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize2d(hipfftHandle plan,
                                    int nx, int ny,
                                    hipfftType type,
                                    size_t *workSize)
{
    hipfftHandle p;
    hipfftPlan2d(&p, nx, ny, type);
    rocfft_plan_get_work_buffer_size(p->ip_forward, workSize);
    hipfftDestroy(p);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize3d(hipfftHandle plan,
                                    int nx, int ny, int nz,
                                    hipfftType type,
                                    size_t *workSize)
{
    hipfftHandle p;
    hipfftPlan3d(&p, nx, ny, nz, type);
    rocfft_plan_get_work_buffer_size(p->ip_forward, workSize);
    hipfftDestroy(p);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan,
                                      int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      hipfftType type, int batch, size_t *workSize)
{
    hipfftHandle p;
    hipfftPlanMany(&p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
    rocfft_plan_get_work_buffer_size(p->ip_forward, workSize);
    hipfftDestroy(p);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize)
{
    rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2C(hipfftHandle plan,
                                  hipfftComplex *idata,
                                  hipfftComplex *odata,
                                  int direction)
{
    void *in[1];
    in[0] = (void *)idata;

    void *out[1];
    out[0] = (void *)odata;

    if(direction == -1)
    {
        rocfft_execute( plan->ip_forward, in, out, nullptr );
    }
    else
    {
        rocfft_execute( plan->ip_forward, in, out, nullptr );
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecR2C(hipfftHandle plan,
                                  hipfftReal *idata,
                                  hipfftComplex *odata)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2R(hipfftHandle plan,
                                  hipfftComplex *idata,
                                  hipfftReal *odata)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2Z(hipfftHandle plan,
                                  hipfftDoubleComplex *idata,
                                  hipfftDoubleComplex *odata,
                                  int direction)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecD2Z(hipfftHandle plan,
                                  hipfftDoubleReal *idata,
                                  hipfftDoubleComplex *odata)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2D(hipfftHandle plan,
                                  hipfftDoubleComplex *idata,
                                  hipfftDoubleReal *odata)
{
    return HIPFFT_SUCCESS;
}


// utility functions
hipfftResult hipfftSetStream(hipfftHandle plan,
                                    hipStream_t stream)
{
    return HIPFFT_SUCCESS;
}

/*
hipfftResult hipfftSetCompatibilityMode(hipfftHandle plan,
                                               hipfftCompatibility mode)
{
    return HIPFFT_SUCCESS;
}
*/

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    if(plan != nullptr)
    {
        rocfft_plan_destroy(plan->ip_forward);
        rocfft_plan_destroy(plan->op_forward);
        rocfft_plan_destroy(plan->ip_inverse);
        rocfft_plan_destroy(plan->op_inverse);
        
        delete plan;
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetVersion(int *version)
{
    return HIPFFT_SUCCESS;
}

