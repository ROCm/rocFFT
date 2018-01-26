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
void complex2real_kernel(size_t input_size, T *input, size_t input_distance, real_type_t<T> *output, size_t output_distance)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    size_t input_offset = hipBlockIdx_y * input_distance;

    size_t output_offset = hipBlockIdx_y * output_distance;

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

    for(size_t i=0; i<data->node->length.size();i++){
        input_size *= data->node->length[i];//flat the dimension to 1D
    }

    if(input_size == 1) return;
    
    size_t input_distance = data->node->iDist;
    size_t output_distance = data->node->oDist;

    void* input_buffer = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch = data->node->batch;
    rocfft_precision precision = data->node->precision;
    
    size_t blocks = (input_size-1)/512 + 1;

    dim3 grid(blocks, batch, 1);//the second dimension is used for batching 
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream; 
    /*
    float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_size*batch);
    hipMemcpy(tmp, input_buffer, sizeof(float2)*input_size*batch, hipMemcpyDeviceToHost);
    
    for(size_t j=0; j< (data->node->length.size() == 2 ? (data->node->length[1]) : 1); j++)
    {
        for(size_t i=0; i<data->node->length[0]; i++)
        { 
            printf("kernel output[%zu][%zu]=(%f, %f) \n", i, j, tmp[j*data->node->length[0]+i].x, tmp[j*data->node->length[0]+i].y);
        }
    }

    free(tmp);*/
    if(precision == rocfft_precision_single) 
        hipLaunchKernelGGL( complex2real_kernel<float2>, grid, threads, 0, rocfft_stream, input_size, (float2 *)input_buffer, input_distance, (float *)output_buffer, output_distance);  
    else 
        hipLaunchKernelGGL( complex2real_kernel<double2>, grid, threads, 0, rocfft_stream, input_size, (double2 *)input_buffer, input_distance, (double *)output_buffer,
output_distance);

    return;    
}


/*============================================================================================*/


template<typename T>
__global__
void hermitian2complex_kernel(const size_t problem_size, const size_t hermitian_size, T *input, size_t input_distance, T *output, size_t output_distance)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    size_t input_offset = hipBlockIdx_z * input_distance;

    size_t output_offset = hipBlockIdx_z * output_distance;

    input_offset += hipBlockIdx_y * hermitian_size;//notice for 1D, hipBlockIdx_y == 0 and thus has no effect for input_offset
    output_offset += hipBlockIdx_y * problem_size;//notice for 1D, hipBlockIdx_y == 0 and thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    if (tid == 0){ // simply write the first elment to output
        output[0] = input[0];
        return;
    }

    if( tid < hermitian_size) //only read the first [N/2+1] elements due to conjugate symmetry, where N = problem_size
    {
        // tid && (N-tid) are conjugate mirror with the sign of imag part flipped;
        // for example if N = 7,hermitian_size = 7/2+1 = 4
        // input elemnts of size (N/2+1=4) = [28.0, -3.5+7.2i, -3.5+2.7i, -3.5+0.79i]
        // output would be  [28.0, -3.5+7.26i, -3.5+2.7i, -3.5+0.79i, -3.5-0.79i, -3.5-2.7i, -3.5-7.26i];
        // where tid=1 && tid=6, tid=2 && tid==5, tid==3 && tid==4  are mirrors 
        // if (tid == N-tid), then it is a real value,flipping imag sign has no effect 
	    T res = input[tid];
        output[tid] = res; 
        output[problem_size-tid].x = res.x;
        output[problem_size-tid].y = -res.y; 

	//printf("problem_size=%d, hermitian_size=%d, tid=%d, ouput.x=%f, ouput.y=%f\n", (int)problem_size, (int)hermitian_size, (int)tid, output[tid].x, output[tid].y);
    }
}


/*! \brief auxiliary function

    read from input_buffer of hermitian structure into an output_buffer of regular complex structure by padding 0

    @param[in]
    problem_size 
           size of problem, not the size of input buffer

    @param[in]
    input_buffer
          data type : complex type (float2 or double2)
          but only store first [1 + problem_size/2] elements according to conjugate symmetry

    @param[in]
    input_distance 
           distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type : complex type (float2 or double2) of size problem_size

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

    size_t problem_size = data->node->length[0];//problem_size is the innermost dimension
    size_t hermitian_size = problem_size/2 + 1;

    if(problem_size == 1) return;
    
    size_t input_distance = data->node->iDist;
    size_t output_distance = data->node->oDist;

    void* input_buffer = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i=1; i<data->node->length.size(); i++)
        { 
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;
    //
    size_t blocks = ( hermitian_size - 1 )/512 + 1;	

    if(high_dimension > 65535 || batch > 65535 ) printf("2D and 3D or batch is too big; not implemented\n");
    //the z dimension is used for batching, 
    //if 2D or 3D, the number of blocks along y will multiple high dimensions 
    //notice the maximum # of thread blocks in y & z is 65535 according to HIP && CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1);//use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream; 

    if(precision == rocfft_precision_single) 
        hipLaunchKernelGGL( hermitian2complex_kernel<float2>, grid, threads, 0, rocfft_stream, problem_size, hermitian_size, (float2 *)input_buffer, input_distance, (float2 *)output_buffer, output_distance);  
    else 
        hipLaunchKernelGGL( hermitian2complex_kernel<double2>, grid, threads, 0, rocfft_stream, problem_size, hermitian_size, (double2 *)input_buffer, input_distance, (double2 *)output_buffer, output_distance);  

    return;    
}




