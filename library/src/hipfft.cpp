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


hipfftResult hipfftMakePlan(hipfftHandle plan, int dim,
                                     size_t *lengths,
                                     hipfftType type,
                                     int batch,
                                     const rocfft_plan_description description)
{

    hipfftResult status = HIPFFT_SUCCESS;

    rocfft_precision precision;
    rocfft_transform_type transform_type;
    rocfft_result_placement placement;

    switch(type) {
        case HIPFFT_C2C  :
        {
            precision = rocfft_precision_single;
        }
        break; 
        case HIPFFT_R2C  :
        {
            precision = rocfft_precision_single;
            transform_type  = rocfft_transform_type_real_forward;
        }
        break;
        case HIPFFT_C2R  :
        {
            precision = rocfft_precision_single;
            transform_type  = rocfft_transform_type_real_inverse;
        }
        case HIPFFT_Z2Z  :
        {
            precision = rocfft_precision_double;
        }
        break;
        case HIPFFT_D2Z  :
        {
            precision = rocfft_precision_double;
            transform_type  = rocfft_transform_type_real_forward;
        }
        break;
        case HIPFFT_Z2D  :
        {
            precision = rocfft_precision_double;
            transform_type  = rocfft_transform_type_real_inverse;
        }
        break;
        default:  
            status = HIPFFT_INVALID_VALUE; break; 
    }
        
    if(status != HIPFFT_SUCCESS)
    {
        return status;
    }

    if(type == HIPFFT_C2C || type == HIPFFT_Z2Z)//if complex2complex
    {
            //has to create 4 plans   
	        rocfft_plan_create_internal(	&plan->forward_inplace,
					        rocfft_placement_inplace,
					        rocfft_transform_type_complex_forward,
					        precision,
					        dim, lengths, batch, description);

	        rocfft_plan_create_internal(	&plan->inverse_inplace,
					        rocfft_placement_inplace,
					        rocfft_transform_type_complex_inverse,
					        precision,
					        dim, lengths, batch, description);

	        rocfft_plan_create_internal(	&plan->forward_notinplace,
					        rocfft_placement_notinplace,
					        rocfft_transform_type_complex_forward,
					        precision,
					        dim, lengths, batch, description);

	        rocfft_plan_create_internal(	&plan->inverse_notinplace,
					        rocfft_placement_notinplace,
					        rocfft_transform_type_complex_inverse,
					        precision,
					        dim, lengths, batch, description);

    }
    else// real2hermitina or hermitian2real
    {
        if(transform_type  == rocfft_transform_type_real_forward)
        {        
	        rocfft_plan_create_internal(	&plan->forward_inplace,
					        rocfft_placement_inplace,
					        transform_type,
					        precision,
					        dim, lengths, batch, description);

	        rocfft_plan_create_internal(	&plan->forward_notinplace,
					        rocfft_placement_notinplace,
					        transform_type,
					        precision,
					        dim, lengths, batch, description);

        }
        else// inverse
        {
	        rocfft_plan_create_internal(	&plan->inverse_inplace,
					        rocfft_placement_inplace,
					        transform_type,
					        precision,
					        dim, lengths, batch, description);

	        rocfft_plan_create_internal(	&plan->inverse_notinplace,
					        rocfft_placement_notinplace,
					        transform_type,
					        precision,
					        dim, lengths, batch, description);

        }
    }
	return status;
}

/*============================================================================================*/

hipfftResult hipfftCreate(hipfftHandle * handle)
{
	hipfftHandle h = new hipfftHandle_t;

	*handle = h;

	return HIPFFT_SUCCESS;
}


/*! \brief Creates a 1D FFT plan configuration for the size and data type. The batch parameter tells how many 1D transforms to perform
 */

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
   
    if (nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

}

/*! \brief Creates a 2D FFT plan configuration according to the sizes and data type.
 */

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

    if (nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

}

/*! \brief Creates a 3D FFT plan configuration according to the sizes and data type.
 */

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

    if (nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

}

/*! \brief 

    Creates a FFT plan according to the dimension rank, sizes specified in the array n. 
    The batch parameter tells hipfft how many transforms to perform. Used in complicated usage case like flexbile input & output layout

    \detaisl 
    plan 	Pointer to the hipfftHandle object

    rank 	Dimensionality of n.

    n 	    Array of size rank, describing the size of each dimension, n[0] being the size of the outermost and n[rank-1] innermost (contiguous) dimension of a transform.
>>>>>>> add copy kernels; add more implementation in hipfft

    inembed 	Define the number of elements in each dimension the output array.
                Pointer of size rank that indicates the storage dimensions of the input data in memory. 
                If set to NULL all other advanced data layout parameters are ignored.

    istride 	The distance between two successive input elements in the least significant (i.e., innermost) dimension

    idist 	    The distance between the first element of two consecutive matrices/vetors in a batch of the input data

    onembed 	Define the number of elements in each dimension the output array.
                Pointer of size rank that indicates the storage dimensions of the output data in memory. 
                If set to NULL all other advanced data layout parameters are ignored.

    ostride 	The distance between two successive output elements in the output array in the least significant (i.e., innermost) dimension

    odist 	    The distance between the first element of two consecutive matrices/vectors in a batch of the output data

    batch 	    number of transforms
 */
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
=======
	size_t lengths[3];
	for(size_t i=0; i<rank; i++)
		lengths[i] = n[rank-1-i];

    hipfftResult status;

	if( (inembed == nullptr) && (onembed == nullptr))
	{
        status = hipfftMakePlan(plan, rank, lengths, type, batch, nullptr);
	}
	else
	{
		rocfft_plan_description desc = nullptr;
		rocfft_plan_description_create(&desc);

		size_t i_strides[3] = {1,1,1};
		size_t o_strides[3] = {1,1,1};

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

        rocfft_array_type input_array_type ;
        rocfft_array_type output_array_type ;

        //TODO: only support interleaved data layout
        if(type == HIPFFT_C2C || type == HIPFFT_Z2Z)  
        {
            input_array_type = rocfft_array_type_complex_interleaved;
            output_array_type = rocfft_array_type_complex_interleaved;
        }
        else if(type == HIPFFT_C2R || type == HIPFFT_Z2D)  
        {
            input_array_type = rocfft_array_type_hermitian_interleaved;
            output_array_type = rocfft_array_type_real;
        }
        else //R2C or R2Z
        {
            input_array_type = rocfft_array_type_real;
            output_array_type = rocfft_array_type_hermitian_interleaved;
        }

		rocfft_plan_description_set_data_layout( desc,  input_array_type,
								output_array_type,
								0, 0,
								rank, i_strides, idist,
								rank, o_strides, odist );

        status = hipfftMakePlan(plan, rank, lengths, type, batch, desc);

		rocfft_plan_description_destroy(desc);
	}

	return status;
>>>>>>> add copy kernels; add more implementation in hipfft
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



/*============================================================================================*/


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


/*! \brief gives an accurate estimate of the work area size required for a plan

    Once plan generation has been done, either with the original API or the extensible API, 
    this call returns the actual size of the work area required to support the plan. 
    Callers who choose to manage work area allocation within their application must use this call after plan generation, 
    and after any hipfftSet*() calls subsequent to plan generation, if those calls might alter the required work space size.

 */

hipfftResult hipfftGetSize(hipfftHandle handle, size_t *workSize)//TODO, we cannot implement this function
{
	return HIPFFT_NOT_SUPPORTED;
}


/*! \brief gives an accurate estimate of the work area size required for a plan
 */

hipfftResult hipfftGetSize_internal(hipfftHandle plan,
                                    hipfftType type,
                                    size_t *workSize )
{

    if(type == HIPFFT_C2C || type == HIPFFT_Z2Z)  //TODO
    {
        rocfft_plan_get_work_buffer_size( (const rocfft_plan)(&(plan->forward_notinplace)), workSize );
    }
    else if(type == HIPFFT_C2R || type == HIPFFT_Z2D)  
    {
        rocfft_plan_get_work_buffer_size( (const rocfft_plan)(&(plan->forward_notinplace)), workSize );
    }
    else //R2C or D2Z
    {
        rocfft_plan_get_work_buffer_size( (const rocfft_plan)(&(plan->forward_notinplace)), workSize );
    }
>>>>>>> add copy kernels; add more implementation in hipfft

    *plan = h;

    return HIPFFT_SUCCESS;
}


/*! \brief gives an accurate estimate of the work area size required for a plan
 */

>>>>>>> add copy kernels; add more implementation in hipfft
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

    if (nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }


}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

    if (nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

    if (nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }


}

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

    if (rank < 0 || istride < 0 || ostride < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    return hipfftGetSize_internal(plan, type, workArea);
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
    rocfft_plan_get_work_buffer_size(plan->ip_forward, workSize);
    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *workArea)
{
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

/*! \brief 
    executes a single-precision complex-to-complex transform plan in the transform direction as specified by direction parameter. 
    If idata and odata are the same, this method does an in-place transform, otherwise an outofplace transform.
 */
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

/*! \brief 
    executes a single-precision real-to-complex, forward, cuFFT transform plan.
 */
hipfftResult hipfftExecR2C(hipfftHandle plan,
                                  hipfftReal *idata,
                                  hipfftComplex *odata)
{

	void *in[1];
	in[0] = (void *)idata;

	void *out[1];
	out[0] = (void *)odata;

    rocfft_execute( &plan->forward_notinplace, in, out, nullptr );

	return HIPFFT_SUCCESS;

}

/*! \brief 
    executes a single-precision real-to-complex, inverse, cuFFT transform plan.
 */
hipfftResult hipfftExecC2R(hipfftHandle plan,
                                  hipfftComplex *idata,
                                  hipfftReal *odata)
{

	void *in[1];
	in[0] = (void *)idata;

	void *out[1];
	out[0] = (void *)odata;

    rocfft_execute( &plan->inverse_notinplace, in, out, nullptr );

	return HIPFFT_SUCCESS;

}

/*! \brief 
    executes a double-precision complex-to-complex transform plan in the transform direction as specified by direction parameter. 
    If idata and odata are the same, this method does an in-place transform, otherwise an outofplace transform.
 */
hipfftResult hipfftExecZ2Z(hipfftHandle plan,
                                  hipfftDoubleComplex *idata,
                                  hipfftDoubleComplex *odata,
                                  int direction)
{

	void *in[1];
	in[0] = (void *)idata;

	void *out[1];
	out[0] = (void *)odata;

	if(direction == -1)
	{
        if( idata == odata)
        {
		    rocfft_execute( &plan->forward_inplace, in, out, nullptr );
        }
        else
        {
		    rocfft_execute( &plan->forward_notinplace, in, out, nullptr );
        }
	}
	else
	{
        if( idata == odata)
        {
		    rocfft_execute( &plan->inverse_inplace, in, out, nullptr );
        }
        else
        {
		    rocfft_execute( &plan->inverse_notinplace, in, out, nullptr );
        }
	}

	return HIPFFT_SUCCESS;

}

/*! \brief 
    executes a double-precision real-to-complex, forward, cuFFT transform plan.
 */
hipfftResult hipfftExecD2Z(hipfftHandle plan,
                                  hipfftDoubleReal *idata,
                                  hipfftDoubleComplex *odata)
{

	void *in[1];
	in[0] = (void *)idata;

	void *out[1];
	out[0] = (void *)odata;

    rocfft_execute( &plan->forward_notinplace, in, out, nullptr );

	return HIPFFT_SUCCESS;

}

hipfftResult hipfftExecZ2D(hipfftHandle plan,
                                  hipfftDoubleComplex *idata,
                                  hipfftDoubleReal *odata)
{

	void *in[1];
	in[0] = (void *)idata;

	void *out[1];
	out[0] = (void *)odata;

    rocfft_execute( &plan->inverse_notinplace, in, out, nullptr );

	return HIPFFT_SUCCESS;

}

/*============================================================================================*/

// Helper functions

/*! \brief 
    Associates a HIP stream with a cuFFT plan. All kernel launched with this plan execution are associated with this stream 
    until the plan is destroyed or the reset to another stream. Returns an error in the multiple GPU case as multiple GPU plans perform operations in their own streams.
*/
hipfftResult hipfftSetStream(hipfftHandle plan,
                                    hipStream_t stream)
{
	return HIPFFT_SUCCESS;//TODO
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

