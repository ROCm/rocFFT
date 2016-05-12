#define __HIPCC__

#if defined(__NVCC__)
#include "helper_math.h"
#endif
#include <iostream>
#include <hip_runtime.h>
#include "rocfft_transpose.h"
#include "rocfft_transpose_kernel.h"

struct rocfft_transpose_plan_t
{
    size_t rank;
    size_t lengths[2];
    size_t batch;

    rocfft_transpose_precision precision;
    rocfft_transpose_array_type array_type;
    rocfft_transpose_placement placement;
};


rocfft_transpose_status rocfft_transpose_plan_create( rocfft_transpose_plan *plan,
                                                                 rocfft_transpose_precision precision, rocfft_transpose_array_type array_type,
                                                                 rocfft_transpose_placement placement,
                                                                 size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                                                 const rocfft_transpose_description *description )
{
    if(dimensions != 2)
        return rocfft_transpose_status_not_implemented;
    if(lengths == NULL)
        return rocfft_transpose_status_failure;
    if(number_of_transforms < 1)
        return rocfft_transpose_status_failure;

    rocfft_transpose_plan p = new rocfft_transpose_plan_t;
    p->rank = dimensions;
    p->lengths[0] = lengths[0];
    p->lengths[1] = lengths[1];
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
    int input_row_size = plan->lengths[1];
    int input_col_size = plan->lengths[0];
    int batch_size = plan->batch;
    

    if(plan->precision == rocfft_transpose_precision_single)
    {
        if(input_row_size % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
        {
            // the kernel should be able to work on any size with no guarantee of performance 
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (float*)in_buffer, (float*)out_buffer, input_row_size, input_col_size, batch_size );
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
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
        {
            // the kernel should be able to work on any size with no guarantee of performance
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<double,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (double*)in_buffer, (double*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else
        {
            return rocfft_transpose_status_not_implemented;
        }
    }    
    else
        return rocfft_transpose_status_not_implemented;
    
    hipDeviceSynchronize();
    return rocfft_transpose_status_success;
}

rocfft_transpose_status rocfft_transpose_outplace_complex_planar_to_complex_planar(const rocfft_transpose_plan plan, void *in_buffer_real, void *in_buffer_imag, void *out_buffer_real, void *out_buffer_imag)
{
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
    int input_row_size = plan->lengths[1];
    int input_col_size = plan->lengths[0];
    int batch_size = plan->batch;
    

    if(plan->precision == rocfft_transpose_precision_single)
    {
        if(input_row_size % (block_size_16*micro_tile_size_4) == 0 && input_col_size % (block_size_16*micro_tile_size_4) == 0)
        {

            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_4,micro_tile_size_4,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_4/block_size_16 * batch_size, input_row_size/micro_tile_size_4/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, batch_size );

        }
        else if(input_row_size % (block_size_16*micro_tile_size_2) == 0 && input_col_size % (block_size_16*micro_tile_size_2) == 0)
        {
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_2,micro_tile_size_2,block_size_16,block_size_16>), dim3(input_col_size/micro_tile_size_2/block_size_16 * batch_size, input_row_size/micro_tile_size_2/block_size_16), dim3(block_size_16, block_size_16, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else if(input_row_size % (block_size_1*micro_tile_size_1) == 0 && input_col_size % (block_size_1*micro_tile_size_1) == 0)
        {
            // the kernel should be able to work on any size with no guarantee of performance 
            hipLaunchKernel(HIP_KERNEL_NAME(transpose_kernel_outplace<float2,micro_tile_size_1,micro_tile_size_1,block_size_1,block_size_1>), dim3(input_col_size/micro_tile_size_1/block_size_1 * batch_size, input_row_size/micro_tile_size_1/block_size_1), dim3(block_size_1, block_size_1, 1), 0, 0, (float2*)in_buffer, (float2*)out_buffer, input_row_size, input_col_size, batch_size );
        }
        else
             return rocfft_transpose_status_not_implemented;
    }
    else
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
}


rocfft_transpose_status rocfft_transpose_plan_destroy( rocfft_transpose_plan plan )
{
        delete plan;
        return rocfft_transpose_status_success;
}
