/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <iostream>
#include "rocfft.h"
#include "rocfft_hip.h"
#include "../kernels/common.h"

template<typename T>
__global__
void real2complex_kernel(hipLaunchParm lp, size_t input_size, real_type_t<T> *input, T *output)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if( tid < input_size)
    {
        output[tid].y = 0.0;
        output[tid].x = input[tid];
    }
}




/*! \brief auxiliary function

    change a read vector into a complex one by padding the imaginary part with 0.

    @param[in]
    input_buffer
          data type dictated by precision parameter (float or double)

    @param[in,output]
    output_buffer
          data type: complex version of input_buffer (float2 or double2)

    @param[in]
    input_size 
           length of input buffer and output buffer
threads,
    @param[in]
    precision 
          data type of input buffer. rocfft_precision_single or rocfft_precsion_double

    ********************************************************************/

/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/
void real2complex(size_t input_size, 
                  void* input_buffer, void* output_buffer, 
                  rocfft_precision precision)
{

    if(input_size == 0) return;
    
    int blocks = (input_size-1)/512 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    if(precision == rocfft_precision_single) 
        hipLaunchKernel(HIP_KERNEL_NAME( real2complex_kernel<float2> ), grid, threads, 0, 0, input_size, (float *)input_buffer, (float2 *)output_buffer);  
    else 
        hipLaunchKernel(HIP_KERNEL_NAME( real2complex_kernel<double2> ), grid, threads, 0, 0, input_size, (double *)input_buffer, (double2 *)output_buffer);  

    return;    
}


/*============================================================================================*/



template<typename T>
__global__
void complex2hermitian_kernel(hipLaunchParm lp, size_t length, T *input, size_t input_distance, T *output, size_t output_distance)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    int input_offset = hipBlockIdx_y * input_distance;

    int output_offset = hipBlockIdx_y * output_distance;

    int bound = length/2 + 1;

    if( tid < bound)//only read and write the first [length/2+1] elements due to conjugate symmetry
    {
        output[output_offset + tid] = input[input_offset + tid];
    }
}




/*! \brief auxiliary function

    read from input_buffer and store the first  [1 + length/2] elements to the output_buffer

    @param[in]
    length 
           size of input buffer

    @param[in]
    input_buffer
          data type dictated by precision parameter but complex type (float2 or double2)

    @param[in]
    input_distance 
           distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type dictated by precision parameter but complex type (float2 or double2)
          but only store first [1 + length/2] elements according to conjugate symmetry

    @param[in]
    output_distance 
           distance between consecutive batch members for output buffer

    @param[in]
    batch 
           number of transforms

    @param[in]
    precision 
           data type of input and output buffer. rocfft_precision_single or rocfft_precsion_double

    ********************************************************************/

void complex2hermitian(size_t length, 
                       void* input_buffer, size_t input_distance, 
                       void* output_buffer, size_t output_distance, 
                       size_t batch, rocfft_precision precision)
{

    if(length == 0) return;
    
    int blocks = (length-1)/512 + 1;

    dim3 grid(blocks, batch, 1);//the second dimension is used for batching 
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    if(precision == rocfft_precision_single) 
        hipLaunchKernel( complex2hermitian_kernel<float2>, grid, threads, 0, 0, length, (float2 *)input_buffer, input_distance, (float2 *)output_buffer, output_distance);  
    else 
        hipLaunchKernel( complex2hermitian_kernel<double2>, grid, threads, 0, 0, length, (double2 *)input_buffer, input_distance, (double2 *)output_buffer, output_distance);  

    return;    
}


/*============================================================================================*/

