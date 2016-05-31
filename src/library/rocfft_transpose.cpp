#define __HIPCC__

#if defined(__NVCC__)
#include "helper_math.h"
#endif
#include <vector>
#include <iostream>
#include <hip_runtime.h>
#include "rocfft_transpose.h"
#include "rocfft_transpose_kernel.h"

struct rocfft_transpose_plan_t
{
    size_t rank;
    std::vector<size_t> *lengths;
    std::vector<size_t> *in_stride;
    std::vector<size_t> *out_stride;
    size_t in_dist;
    size_t out_dist;
    size_t batch;

    rocfft_transpose_precision precision;
    rocfft_transpose_array_type array_type;
    rocfft_transpose_placement placement;
};

static void isTransposeDataPacked(const rocfft_transpose_plan &plan, bool &packed_data_in, bool &packed_data_out)
{
   //input data
   packed_data_in = true;
   packed_data_out = true;
   
   size_t packed_data_size = 1;
   for(int i = 0; i < plan->rank; i++)
   {
       packed_data_size *= (plan->lengths)->at(i);
   }
   if(plan->in_dist > packed_data_size)
       packed_data_in = false;
   if(plan->out_dist > packed_data_size)
       packed_data_out = false;
}

rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                      rocfft_transpose_precision precision, 
                                                      rocfft_transpose_array_type array_type,
                                                      rocfft_transpose_placement placement,
                                                      size_t dimensions, 
                                                      const size_t *lengths, 
                                                      const size_t *in_stride,
                                                      const size_t *out_stride,
                                                      const size_t in_dist,
                                                      const size_t out_dist,
                                                      size_t number_of_transforms,
                                                      const rocfft_transpose_description *description )
{
    if(dimensions != 2) // only support 2 dimension for now
        return rocfft_transpose_status_not_implemented;
    if(lengths == NULL)
        return rocfft_transpose_status_failure;
    if(number_of_transforms < 1)
        return rocfft_transpose_status_failure;

    rocfft_transpose_plan p = new rocfft_transpose_plan_t;
    p->rank = dimensions;
    p->lengths = new std::vector<size_t>(lengths, lengths + dimensions);
    p->in_stride = new std::vector<size_t>(in_stride, in_stride + dimensions);
    p->out_stride = new std::vector<size_t>(out_stride, out_stride + dimensions);

    p->in_dist = in_dist;
    p->out_dist = out_dist;
    p->batch = number_of_transforms;
    p->precision = precision;
    p->array_type = array_type;
    p->placement = placement;
    *plan = p;
    
    return rocfft_transpose_status_success;
}

rocfft_transpose_status rocfft_transpose_outplace_real(const rocfft_transpose_plan plan, void *in_buffer, void *out_buffer)
{
    const int block_size_16 = 16;//dim3(input_row_size/16/4, input_col_size/16/4, 1)
    const int block_size_8 = 8;
    const int block_size_1 = 1;
    const int micro_tile_size_1 = 1;
    const int micro_tile_size_2 = 2;
    const int micro_tile_size_4 = 4;
    int input_row_size = plan->lengths->at(1);
    int input_col_size = plan->lengths->at(0);
    int ld_in = plan->in_stride->at(1);
    int ld_out = plan->out_stride->at(1);
    int batch_size = plan->batch;
    
    bool packed_data_in = false;// whether the input data is packed
    bool packed_data_out = false;// whether the output data is packed
    isTransposeDataPacked(plan, packed_data_in, packed_data_out);
    
    if(packed_data_in && packed_data_out)
    {
        if(plan->precision == rocfft_transpose_precision_single)
        {
            if(input_row_size % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
            {
                // the kernel should be able to work on any size with no guarantee of performance 
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else
            {
                return rocfft_transpose_status_not_implemented;
            }
        }
        else if(plan->precision == rocfft_transpose_precision_double)
        {
            if(input_row_size % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
            {
                // the kernel should be able to work on any size with no guarantee of performance
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else
            {
                return rocfft_transpose_status_not_implemented;
            }
        }    
        else//not single or double precision
            return rocfft_transpose_status_not_implemented;
        }
        else//not packed data
           return rocfft_transpose_status_not_implemented;
    hipDeviceSynchronize();
    return rocfft_transpose_status_success;
}

rocfft_transpose_status rocfft_transpose_outplace_complex_planar_to_complex_planar(const rocfft_transpose_plan plan, void *in_buffer_real, void *in_buffer_imag, void *out_buffer_real, void *out_buffer_imag)
{
    const int block_size_16 = 16;//dim3(input_row_size/16/4, input_col_size/16/4, 1)
    const int block_size_8 = 8;
    const int block_size_1 = 1;
    const int micro_tile_size_1 = 1;
    const int micro_tile_size_2 = 2;
    const int micro_tile_size_4 = 4;
    int input_row_size = plan->lengths->at(1);
    int input_col_size = plan->lengths->at(0);
    int ld_in = plan->in_stride->at(1);
    int ld_out = plan->out_stride->at(1);
    int batch_size = plan->batch;
    
    bool packed_data_in = false;
    bool packed_data_out = false;
    isTransposeDataPacked(plan, packed_data_in, packed_data_out);


   if(packed_data_in && packed_data_out)
   {
        if(plan->precision == rocfft_transpose_precision_single)
        {
            if(input_row_size  % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace_complex_planar_to_complex_planar<float,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer_real, (float*)in_buffer_imag, (float*)out_buffer_real, (float*)out_buffer_imag, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace_complex_planar_to_complex_planar<float,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer_real, (float*)in_buffer_imag, (float*)out_buffer_real, (float*)out_buffer_imag, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
            {
                // the kernel should be able to work on any size with no guarantee of performance 
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace_complex_planar_to_complex_planar<float,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (float*)in_buffer_real, (float*)in_buffer_imag, (float*)out_buffer_real, (float*)out_buffer_imag, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else
                 return rocfft_transpose_status_not_implemented;
        }
	else//not single precision
            return rocfft_transpose_status_not_implemented;
   }
   else// not packed data
       return rocfft_transpose_status_not_implemented;

   return rocfft_transpose_status_not_implemented;
}

rocfft_transpose_status rocfft_transpose_outplace_complex_interleaved_to_complex_interleaved(const rocfft_transpose_plan plan, void *in_buffer, void *out_buffer)
{
    const int block_size_16 = 16;//dim3(input_row_size/16/4, input_col_size/16/4, 1)
    const int block_size_8 = 8;
    const int block_size_1 = 1;
    const int micro_tile_size_1 = 1;
    const int micro_tile_size_2 = 2;
    const int micro_tile_size_4 = 4;
    int input_row_size = plan->lengths->at(1);
    int input_col_size = plan->lengths->at(0);
    int ld_in = plan->in_stride->at(1);
    int ld_out = plan->out_stride->at(1);
    int batch_size = plan->batch;
    
    bool packed_data_in = false;
    bool packed_data_out = false;
    isTransposeDataPacked(plan, packed_data_in, packed_data_out);   
    
    if(packed_data_in && packed_data_out)
    {
        if(plan->precision == rocfft_transpose_precision_single)
        {
            if(input_row_size  % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
            {
                // the kernel should be able to work on any size with no guarantee of performance 
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else
                 return rocfft_transpose_status_not_implemented;
        }
        else if(plan->precision == rocfft_transpose_precision_double)
        {
            /*if(input_row_size  % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double2,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double2*)in_buffer, (double2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }*/
            if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
            {
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double2,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double2*)in_buffer, (double2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
            {
                // the kernel should be able to work on any size with no guarantee of performance
                hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double2,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (double2*)in_buffer, (double2*)out_buffer, input_row_size, input_col_size, ld_in, ld_out, batch_size );
            }
            else
                 return rocfft_transpose_status_not_implemented;

        }
	else//not single precision
            return rocfft_transpose_status_not_implemented;
	}
    else// not packed data
        return rocfft_transpose_status_not_implemented;

    hipDeviceSynchronize();
    return rocfft_transpose_status_success;
}

rocfft_transpose_status rocfft_transpose_execute( const rocfft_transpose_plan plan,
                                                             void **in_buffer,
                                                             void **out_buffer,
                                                             rocfft_transpose_execution_info info )
{
    if(plan->placement == rocfft_transpose_placement_inplace)        
        return rocfft_transpose_status_not_implemented;
 
    if(plan->array_type != rocfft_transpose_array_type_real_to_real
       && plan->array_type != rocfft_transpose_array_type_complex_planar_to_complex_planar
       && plan->array_type != rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved)
        return rocfft_transpose_status_not_implemented;

    if(plan->array_type == rocfft_transpose_array_type_real_to_real)
        return rocfft_transpose_outplace_real(plan, *in_buffer, *out_buffer);

    if(plan->array_type == rocfft_transpose_array_type_complex_planar_to_complex_planar)
        return rocfft_transpose_outplace_complex_planar_to_complex_planar(plan, in_buffer[0], in_buffer[1], out_buffer[0], out_buffer[1]);

    if(plan->array_type == rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved)
        return rocfft_transpose_outplace_complex_interleaved_to_complex_interleaved(plan, *in_buffer, *out_buffer);

    return rocfft_transpose_status_not_implemented;
}


rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan )
{ 
        delete plan->out_stride;
        delete plan->in_stride;
        delete plan->lengths;
        delete plan;
        return rocfft_transpose_status_success;
}
