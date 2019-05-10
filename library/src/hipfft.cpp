/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "hipfft.h"
#include "plan.h"
#include "private.h"
#include "rocfft.h"
#include <sstream>

#define ROC_FFT_CHECK_ALLOC_FAILED(ret)  \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_ALLOC_FAILED;  \
        }                                \
    }

#define ROC_FFT_CHECK_INVALID_VALUE(ret) \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_INVALID_VALUE; \
        }                                \
    }

#define ROC_FFT_CHECK_EXEC_FAILED(ret)   \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_EXEC_FAILED;   \
        }                                \
    }

#define HIP_FFT_CHECK_AND_RETURN(ret) \
    {                                 \
        if(ret != HIPFFT_SUCCESS)     \
        {                             \
            return ret;               \
        }                             \
    }

struct hipfftHandle_t
{
    rocfft_plan           ip_forward;
    rocfft_plan           op_forward;
    rocfft_plan           ip_inverse;
    rocfft_plan           op_inverse;
    rocfft_execution_info info;
    void*                 workBuffer;

    hipfftHandle_t()
        : ip_forward(nullptr)
        , op_forward(nullptr)
        , ip_inverse(nullptr)
        , op_inverse(nullptr)
        , info(nullptr)
        , workBuffer(nullptr)
    {
    }
};

/*! \brief Creates a 1D FFT plan configuration for the size and data type. The
 * batch parameter tells how many 1D transforms to perform
 */
hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan1d(*plan, nx, type, batch, nullptr);
}

/*! \brief Creates a 2D FFT plan configuration according to the sizes and data
 * type.
 */
hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan2d(*plan, nx, ny, type, nullptr);
}

/*! \brief Creates a 3D FFT plan configuration according to the sizes and data
 * type.
 */
hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan3d(*plan, nx, ny, nz, type, nullptr);
}

hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlanMany(
        *plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
}

hipfftResult hipfftMakePlan_internal(hipfftHandle            plan,
                                     size_t                  dim,
                                     size_t*                 lengths,
                                     hipfftType              type,
                                     size_t                  number_of_transforms,
                                     rocfft_plan_description desc,
                                     size_t*                 workSize,
                                     bool                    dry_run)
{
    size_t workBufferSize = 0;

    switch(type)
    {
    case HIPFFT_R2C:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->ip_forward,
                                                                rocfft_placement_inplace,
                                                                rocfft_transform_type_real_forward,
                                                                rocfft_precision_single,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->op_forward,
                                                                rocfft_placement_notinplace,
                                                                rocfft_transform_type_real_forward,
                                                                rocfft_precision_single,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        break;
    case HIPFFT_C2R:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->ip_inverse,
                                                                rocfft_placement_inplace,
                                                                rocfft_transform_type_real_inverse,
                                                                rocfft_precision_single,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->op_inverse,
                                                                rocfft_placement_notinplace,
                                                                rocfft_transform_type_real_inverse,
                                                                rocfft_precision_single,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        break;
    case HIPFFT_C2C:
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->ip_forward,
                                        rocfft_placement_inplace,
                                        rocfft_transform_type_complex_forward,
                                        rocfft_precision_single,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->op_forward,
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_complex_forward,
                                        rocfft_precision_single,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->ip_inverse,
                                        rocfft_placement_inplace,
                                        rocfft_transform_type_complex_inverse,
                                        rocfft_precision_single,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->op_inverse,
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_complex_inverse,
                                        rocfft_precision_single,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        break;

    case HIPFFT_D2Z:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->ip_forward,
                                                                rocfft_placement_inplace,
                                                                rocfft_transform_type_real_forward,
                                                                rocfft_precision_double,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->op_forward,
                                                                rocfft_placement_notinplace,
                                                                rocfft_transform_type_real_forward,
                                                                rocfft_precision_double,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        break;
    case HIPFFT_Z2D:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->ip_inverse,
                                                                rocfft_placement_inplace,
                                                                rocfft_transform_type_real_inverse,
                                                                rocfft_precision_double,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create_internal(plan->op_inverse,
                                                                rocfft_placement_notinplace,
                                                                rocfft_transform_type_real_inverse,
                                                                rocfft_precision_double,
                                                                dim,
                                                                lengths,
                                                                number_of_transforms,
                                                                desc,
                                                                dry_run));
        break;
    case HIPFFT_Z2Z:
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->ip_forward,
                                        rocfft_placement_inplace,
                                        rocfft_transform_type_complex_forward,
                                        rocfft_precision_double,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->op_forward,
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_complex_forward,
                                        rocfft_precision_double,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->ip_inverse,
                                        rocfft_placement_inplace,
                                        rocfft_transform_type_complex_inverse,
                                        rocfft_precision_double,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_create_internal(plan->op_inverse,
                                        rocfft_placement_notinplace,
                                        rocfft_transform_type_complex_inverse,
                                        rocfft_precision_double,
                                        dim,
                                        lengths,
                                        number_of_transforms,
                                        desc,
                                        dry_run));
        break;
    default:
        assert(false);
    }

    size_t tmpBufferSize = 0;
    if(plan->ip_forward)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->op_forward)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->ip_inverse)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->op_inverse)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }

    if(workBufferSize > 0)
    {
        if(plan->workBuffer)
            if(hipFree(plan->workBuffer) != HIP_SUCCESS)
                return HIPFFT_ALLOC_FAILED;
        if(hipMalloc(&plan->workBuffer, workBufferSize) != HIP_SUCCESS)
            return HIPFFT_ALLOC_FAILED;
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_execution_info_set_work_buffer(plan->info, plan->workBuffer, workBufferSize));
    }

    if(workSize != nullptr)
        *workSize = workBufferSize;

    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

/*! \brief Assume hipfftCreate has been called. Creates a 1D FFT plan
 * configuration for the size and data type. The batch parameter tells how many
 * 1D transforms to perform
 */
hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{

    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[1];
    lengths[0]                                   = nx;
    size_t                  number_of_transforms = batch;
    rocfft_plan_description desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 1, lengths, type, number_of_transforms, desc, workSize, false);
}

/*! \brief Assume hipfftCreate has been called. Creates a 2D FFT plan
 * configuration according to the sizes and data type.
 */
hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{

    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[2];
    lengths[0]                                   = ny;
    lengths[1]                                   = nx;
    size_t                  number_of_transforms = 1;
    rocfft_plan_description desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 2, lengths, type, number_of_transforms, desc, workSize, false);
}

/*! \brief Assume hipfftCreate has been called. Creates a 3D FFT plan
 * configuration according to the sizes and data type.
 */
hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{

    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[3];
    lengths[0]                                   = nz;
    lengths[1]                                   = ny;
    lengths[2]                                   = nx;
    size_t                  number_of_transforms = 1;
    rocfft_plan_description desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 3, lengths, type, number_of_transforms, desc, workSize, false);
}

/*! \brief

    Creates a FFT plan according to the dimension rank, sizes specified in the
   array n.
    The batch parameter tells hipfft how many transforms to perform. Used in
   complicated usage case like flexbile input & output layout

    \details
    plan 	Pointer to the hipfftHandle object

    rank 	Dimensionality of n.

    n 	    Array of size rank, describing the size of each dimension, n[0]
   being the size of the outermost and n[rank-1] innermost (contiguous)
   dimension of a transform.

    inembed 	Define the number of elements in each dimension the input array.
                Pointer of size rank that indicates the storage dimensions of
   the input data in memory.
                If set to NULL all other advanced data layout parameters are
   ignored.

    istride 	The distance between two successive input elements in the least
   significant (i.e., innermost) dimension

    idist 	    The distance between the first element of two consecutive
   matrices/vetors in a batch of the input data

    onembed 	Define the number of elements in each dimension the output
   array.
                Pointer of size rank that indicates the storage dimensions of
   the output data in memory.
                If set to NULL all other advanced data layout parameters are
   ignored.

    ostride 	The distance between two successive output elements in the
   output array in the least significant (i.e., innermost) dimension

    odist 	    The distance between the first element of two consecutive
   matrices/vectors in a batch of the output data

    batch 	    number of transforms
 */
hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                int          rank,
                                int*         n,
                                int*         inembed,
                                int          istride,
                                int          idist,
                                int*         onembed,
                                int          ostride,
                                int          odist,
                                hipfftType   type,
                                int          batch,
                                size_t*      workSize)
{

    size_t lengths[3];
    for(size_t i = 0; i < rank; i++)
        lengths[i] = n[rank - 1 - i];

    size_t number_of_transforms = batch;

    size_t workBufferSize = 0;

    rocfft_plan_description desc = nullptr;
    if((inembed != nullptr) || (onembed != nullptr))
    {
        rocfft_plan_description_create(&desc);

        size_t i_strides[3] = {1, 1, 1};
        size_t o_strides[3] = {1, 1, 1};

        // pre-fetch the default params in case one of inembed and onembed
        // is NULL
        hipfftMakePlan_internal(
            plan, rank, lengths, type, number_of_transforms, nullptr, workSize, true);

        if(inembed == nullptr) // restore the default strides
        {
            for(size_t i = 1; i < rank; i++)
                i_strides[i] = plan->ip_forward->desc.inStrides[i];
        }
        else
        {
            i_strides[0] = istride;

            size_t inembed_lengths[3];
            for(size_t i = 0; i < rank; i++)
                inembed_lengths[i] = inembed[rank - 1 - i];

            for(size_t i = 1; i < rank; i++)
                i_strides[i] = inembed_lengths[i - 1] * i_strides[i - 1];
        }

        if(onembed == nullptr) // restore the default strides
        {
            for(size_t i = 1; i < rank; i++)
                o_strides[i] = plan->ip_forward->desc.outStrides[i];
        }
        else
        {
            o_strides[0] = ostride;

            size_t onembed_lengths[3];
            for(size_t i = 0; i < rank; i++)
                onembed_lengths[i] = onembed[rank - 1 - i];

            for(size_t i = 1; i < rank; i++)
                o_strides[i] = onembed_lengths[i - 1] * o_strides[i - 1];
        }

        // Decide the inArrayType and outArrayType based on the transform type
        rocfft_array_type in_array_type, out_array_type;
        switch(type)
        {
        case HIPFFT_R2C:
        case HIPFFT_D2Z:
            in_array_type  = rocfft_array_type_real;
            out_array_type = rocfft_array_type_hermitian_interleaved;
            break;
        case HIPFFT_C2R:
        case HIPFFT_Z2D:
            in_array_type  = rocfft_array_type_hermitian_interleaved;
            out_array_type = rocfft_array_type_real;
            break;
        case HIPFFT_C2C:
        case HIPFFT_Z2Z:
            in_array_type  = rocfft_array_type_complex_interleaved;
            out_array_type = rocfft_array_type_complex_interleaved;
            break;
        defaut:
            in_array_type  = rocfft_array_type_complex_interleaved;
            out_array_type = rocfft_array_type_complex_interleaved;
            break;
        }

        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_data_layout(desc,
                                                                            in_array_type,
                                                                            out_array_type,
                                                                            0,
                                                                            0,
                                                                            rank,
                                                                            i_strides,
                                                                            idist,
                                                                            rank,
                                                                            o_strides,
                                                                            odist));
    }

    hipfftResult ret = hipfftMakePlan_internal(
        plan, rank, lengths, type, number_of_transforms, desc, workSize, false);

    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_destroy(desc));

    return ret;
}

hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
                                  int            rank,
                                  long long int* n,
                                  long long int* inembed,
                                  long long int  istride,
                                  long long int  idist,
                                  long long int* onembed,
                                  long long int  ostride,
                                  long long int  odist,
                                  hipfftType     type,
                                  long long int  batch,
                                  size_t*        workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

/*============================================================================================*/

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftEstimateMany(int        rank,
                                int*       n,
                                int*       inembed,
                                int        istride,
                                int        idist,
                                int*       onembed,
                                int        ostride,
                                int        odist,
                                hipfftType type,
                                int        batch,
                                size_t*    workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftCreate(hipfftHandle* plan)
{
    hipfftHandle h = new hipfftHandle_t;

    ROC_FFT_CHECK_ALLOC_FAILED(rocfft_plan_allocate(&h->ip_forward));
    ROC_FFT_CHECK_ALLOC_FAILED(rocfft_plan_allocate(&h->op_forward));
    ROC_FFT_CHECK_ALLOC_FAILED(rocfft_plan_allocate(&h->ip_inverse));
    ROC_FFT_CHECK_ALLOC_FAILED(rocfft_plan_allocate(&h->op_inverse));

    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_create(&h->info));

    *plan = h;

    return HIPFFT_SUCCESS;
}

/*! \brief gives an accurate estimate of the work area size required for a plan

    Once plan generation has been done, either with the original API or the
   extensible API,
    this call returns the actual size of the work area required to support the
   plan.
    Callers who choose to manage work area allocation within their application
   must use this call after plan generation,
    and after any hipfftSet*() calls subsequent to plan generation, if those
   calls might alter the required work space size.

 */

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult hipfftGetSize_internal(hipfftHandle plan, hipfftType type, size_t* workSize)
{

    if(type == HIPFFT_C2C || type == HIPFFT_Z2Z) // TODO
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }
    else if(type == HIPFFT_C2R || type == HIPFFT_Z2D)
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }
    else // R2C or D2Z
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }

    return HIPFFT_SUCCESS;
}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{

    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftPlan1d(&p, nx, type, batch));
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(p->ip_forward, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftPlan2d(&p, nx, ny, type));
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(p->ip_forward, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftPlan3d(&p, nx, ny, nz, type));
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(p->ip_forward, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult hipfftGetSizeMany(hipfftHandle plan,
                               int          rank,
                               int*         n,
                               int*         inembed,
                               int          istride,
                               int          idist,
                               int*         onembed,
                               int          ostride,
                               int          odist,
                               hipfftType   type,
                               int          batch,
                               size_t*      workSize)
{

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(
        hipfftPlanMany(&p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(p->ip_forward, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize)
{

    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize));
    // return hipfftGetSize_internal(plan, type, workArea);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
                                 int            rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int  istride,
                                 long long int  idist,
                                 long long int* onembed,
                                 long long int  ostride,
                                 long long int  odist,
                                 hipfftType     type,
                                 long long int  batch,
                                 size_t*        workSize)
{
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize));
    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

/*============================================================================================*/

/*! \brief
    executes a single-precision complex-to-complex transform plan in the
   transform direction as specified by direction parameter.
    If idata and odata are the same, this method does an in-place transform,
   otherwise an outofplace transform.
 */
hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
{
    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    if(direction == HIPFFT_FORWARD)
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_forward : plan->op_forward, in, out, plan->info));
    }
    else
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));
    }

    return HIPFFT_SUCCESS;
}

/*! \brief
    executes a single-precision real-to-complex, forward, cuFFT transform plan.
 */
hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_forward : plan->op_forward, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

/*! \brief
    executes a single-precision real-to-complex, inverse, cuFFT transform plan.
 */
hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

/*! \brief
    executes a double-precision complex-to-complex transform plan in the
   transform direction as specified by direction parameter.
    If idata and odata are the same, this method does an in-place transform,
   otherwise an outofplace transform.
 */
hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    if(direction == HIPFFT_FORWARD)
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_forward : plan->op_forward, in, out, plan->info));
    }
    else
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));
    }

    return HIPFFT_SUCCESS;
}

/*! \brief
    executes a double-precision real-to-complex, forward, cuFFT transform plan.
 */
hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_forward : plan->op_forward, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

// Helper functions

/*! \brief
    Associates a HIP stream with a cuFFT plan. All kernel launched with this
   plan execution are associated with this stream
    until the plan is destroyed or the reset to another stream. Returns an error
   in the multiple GPU case as multiple GPU plans perform operations in their
   own streams.
*/
hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_stream(plan->info, stream));
    return HIPFFT_SUCCESS;
}

/*! \brief
Function hipfftSetCompatibilityMode is deprecated.

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
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_forward));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_forward));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_inverse));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_inverse));

        hipFree(plan->workBuffer);
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_destroy(plan->info));

        delete plan;
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetVersion(int* version)
{
    char v[256];
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_get_version_string(v, 256));

    // assume maximum 2 digts for each, so xx.xx.xx.xx -> xxxxxxxx
    std::ostringstream       result;
    std::vector<std::string> sections;

    std::istringstream iss(v);
    std::string        tmp_str;
    while(std::getline(iss, tmp_str, '.'))
    {
        sections.push_back(tmp_str);
    }

    for(size_t i = 0; i < sections.size(); i++)
    {
        std::vector<std::string> sl;
        // remove potential git tag string
        std::istringstream iss(sections[i]);
        while(std::getline(iss, tmp_str, '-'))
        {
            sl.push_back(tmp_str);
        }
        if(sl[0].size() == 0)
            result << "00";
        else if(sl[0].size() == 1)
            result << "0" << sl[0][0];
        else
            result << sl[0].at(sl[0].size() - 2) << sl[0].at(sl[0].size() - 1);
    }

    *version = std::stoi(result.str());
    return HIPFFT_SUCCESS;
}
