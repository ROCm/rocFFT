/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <iostream>
#include "rocfft.h"
#include "rocfft_hip.h"
#include "kernel_launch.h"
#include "./kernels/common.h"


template<typename T>
__global__
void complex2real_kernel(hipLaunchParm lp, size_t input_size, T *input, size_t input_distance, real_type_t<T> *output, size_t output_distance)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    int input_offset = hipBlockIdx_y * input_distance;

    int output_offset = hipBlockIdx_y * output_distance;

    input += input_offset;
    output += output_offset;

    if( tid < input_size)
    {
        output[tid] = input[tid].x;
    }
}


/*! \brief auxiliary function

    convert a complex vector into a real one by only taking the real part of the complex vector

    @param[in]
    input_size 
           size of input buffer

    @param[in]
    input_buffer
          data type : float2 or double2 

    @param[in]
    input_distance 
          distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type : float or double

    @param[in]
    output_distance 
           distance between consecutive batch members for output buffer

    @param[in]
    batch 
           number of transforms

    @param[in]
    precision 
          data type of input buffer. rocfft_precision_single or rocfft_precsion_double


    ********************************************************************/

//currently only works for stride=1 cases
void complex2real(const void *data_p, void *back_p) 
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

    size_t input_size = 1;

    for(int i=0; i<data->node->length.size();i++){
        input_size *= data->node->length[i];
    }

    if(input_size == 1) return;
    
    size_t input_distance = data->node->iDist;
    size_t output_distance = data->node->oDist;

    void* input_buffer = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch = data->node->batch;
    rocfft_precision precision = data->node->precision;
    
    int blocks = (input_size-1)/512 + 1;

    dim3 grid(blocks, batch, 1);//the second dimension is used for batching 
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    if(precision == rocfft_precision_single) 
        hipLaunchKernel( complex2real_kernel<float2>, grid, threads, 0, 0, input_size, (float2 *)input_buffer, input_distance, (float *)output_buffer, output_distance);  
    else 
        hipLaunchKernel( complex2real_kernel<double2>, grid, threads, 0, 0, input_size, (double2 *)input_buffer, input_distance, (double *)output_buffer,
output_distance);

    return;    
}


/*============================================================================================*/


template<typename T>
__global__
void hermitian2complex_kernel(hipLaunchParm lp, size_t N, T *input, size_t input_distance, T *output, size_t output_distance)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    int input_offset = hipBlockIdx_y * input_distance;

    int output_offset = hipBlockIdx_y * output_distance;

    int bound = N/2 + 1; 

    input += input_offset;
    output += output_offset;

    if( tid < bound) //only read the first [N/2+1] elements due to conjugate symmetry
    {
        // tid && (N - tid) are mirror;
        output[N-tid] = output[tid] = input[tid]; 
    }
}


/*! \brief auxiliary function

    read from input_buffer of hermitian structure into an output_buffer of regular complex structure by padding 0

    @param[in]
    input_size 
           size of input buffer

    @param[in]
    input_buffer
          data type : complex type (float2 or double2)
          but only store first [1 + output_size/2] elements according to conjugate symmetry

    @param[in]
    input_distance 
           distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type : complex type (float2 or double2) of size output_size

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

void hermitian2complex(const void *data_p, void *back_p) 
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

    size_t input_size = 1;

    for(int i=0; i<data->node->length.size();i++){
        input_size *= data->node->length[i];
    }

    if(input_size == 1) return;
    
    size_t input_distance = data->node->iDist;
    size_t output_distance = data->node->oDist;

    void* input_buffer = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch = data->node->batch;
    rocfft_precision precision = data->node->precision;

    size_t output_size; 
    if (input_size/2){
        output_size = (input_size-1)*2;
    } 
    else{
        output_size = (input_size-1)*2 + 1;
    }

    int blocks = (input_size-1)/512 + 1;

    dim3 grid(blocks, batch, 1);//the second dimension is used for batching 
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    if(precision == rocfft_precision_single) 
        hipLaunchKernel( hermitian2complex_kernel<float2>, grid, threads, 0, 0, output_size, (float2 *)input_buffer, input_distance, (float2 *)output_buffer, output_distance);  
    else 
        hipLaunchKernel( hermitian2complex_kernel<double2>, grid, threads, 0, 0, output_size, (double2 *)input_buffer, input_distance, (double2 *)output_buffer, output_distance);  

    return;    
}




