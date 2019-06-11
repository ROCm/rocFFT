/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "./kernels/common.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include <iostream>

template <typename T>
__global__ void complex2real_kernel(size_t          input_size,
                                    size_t          input_stride,
                                    size_t          output_stride,
                                    T*              input,
                                    size_t          input_distance,
                                    real_type_t<T>* output,
                                    size_t          output_distance)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    size_t input_offset  = hipBlockIdx_z * input_distance;
    size_t output_offset = hipBlockIdx_z * output_distance;

    input_offset += hipBlockIdx_y * input_stride; // notice for 1D, hipBlockIdx_y
    // == 0 and thus has no effect
    // for input_offset
    output_offset += hipBlockIdx_y * output_stride; // notice for 1D, hipBlockIdx_y == 0 and
    // thus has no effect for output_offset

    input += input_offset;
    output += output_offset;

    if(tid < input_size)
    {
        output[tid] = input[tid].x;
    }
}

/*! \brief auxiliary function

    convert a complex vector into a real one by only taking the real part of the
   complex vector

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
          data type of input buffer. rocfft_precision_single or
   rocfft_precsion_double


    ********************************************************************/

// currently only works for stride=1 cases
void complex2real(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t input_size = data->node->length[0];

    if(input_size == 1)
        return;

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;
    //
    size_t blocks = (input_size - 1) / 512 + 1;

    if(high_dimension > 65535 || batch > 65535)
        printf("2D and 3D or batch is too big; not implemented\n");
    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;
    /*
  float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_size*batch);
  hipMemcpy(tmp, input_buffer, sizeof(float2)*input_size*batch,
  hipMemcpyDeviceToHost);

  for(size_t j=0; j< (data->node->length.size() == 2 ? (data->node->length[1]) :
  1); j++)
  {
      for(size_t i=0; i<data->node->length[0]; i++)
      {
          printf("kernel output[%zu][%zu]=(%f, %f) \n", i, j,
  tmp[j*data->node->length[0]+i].x, tmp[j*data->node->length[0]+i].y);
      }
  }

  free(tmp);*/
    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(complex2real_kernel<float2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (float2*)input_buffer,
                           input_distance,
                           (float*)output_buffer,
                           output_distance);
    else
        hipLaunchKernelGGL(complex2real_kernel<double2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           input_size,
                           input_stride,
                           output_stride,
                           (double2*)input_buffer,
                           input_distance,
                           (double*)output_buffer,
                           output_distance);

    return;
}

/*============================================================================================*/

template <typename T>
__global__ void hermitian2complex_kernel(size_t hermitian_size,
                                         size_t dim_0,
                                         size_t dim_1,
                                         size_t dim_2,
                                         size_t input_stride,
                                         size_t output_stride,
                                         T*     input,
                                         size_t input_distance,
                                         T*     output,
                                         size_t output_distance)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    size_t input_offset   = hipBlockIdx_z * input_distance;
    size_t outputs_offset = hipBlockIdx_z * output_distance; // straight copy
    size_t outputc_offset = hipBlockIdx_z * output_distance; // conjugate copy

    // straight copy indices
    size_t is0 = tid;
    size_t is1 = hipBlockIdx_y % dim_1;
    size_t is2 = hipBlockIdx_y / dim_1;

    // conjugate copy indices
    size_t ic0 = (is0 == 0) ? 0 : dim_0 - is0;
    size_t ic1 = (is1 == 0) ? 0 : dim_1 - is1;
    size_t ic2 = (is2 == 0) ? 0 : dim_2 - is2;

    input_offset += hipBlockIdx_y * input_stride + is0; // notice for 1D,
    // hipBlockIdx_y == 0 and
    // thus has no effect for
    // input_offset
    outputs_offset += (is2 * dim_1 + is1) * output_stride + is0;
    outputc_offset += (ic2 * dim_1 + ic1) * output_stride + ic0;

    input += input_offset;
    T* outputs = output + outputs_offset;
    T* outputc = output + outputc_offset;

    if((is0 == 0) || (is0 * 2 == dim_0)) // simply write the element to output
    {
        outputs[0] = input[0];
        return;
    }

    if(is0 < hermitian_size)
    {
        T res      = input[0];
        outputs[0] = res;
        outputc[0] = T(res.x, -res.y);
    }
}

/*! \brief auxiliary function

    read from input_buffer of hermitian structure into an output_buffer of
   regular complex structure by padding 0

    @param[in]
    dim_0
           size of problem, not the size of input buffer

    @param[in]
    input_buffer
          data type : complex type (float2 or double2)
          but only store first [1 + dim_0/2] elements according to conjugate
   symmetry

    @param[in]
    input_distance
           distance between consecutive batch members for input buffer

    @param[in,output]
    output_buffer
          data type : complex type (float2 or double2) of size dim_0

    @param[in]
    output_distance
           distance between consecutive batch members for output buffer

    @param[in]
    batch
           number of transforms

    @param[in]
    precision
           data type of input and output buffer. rocfft_precision_single or
   rocfft_precsion_double

    ********************************************************************/

void hermitian2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    size_t dim_0          = data->node->length[0]; // dim_0 is the innermost dimension
    size_t hermitian_size = dim_0 / 2 + 1;

    if(dim_0 == 1)
        return;

    size_t input_distance  = data->node->iDist;
    size_t output_distance = data->node->oDist;

    size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    size_t batch          = data->node->batch;
    size_t high_dimension = 1;
    if(data->node->length.size() > 1)
    {
        for(int i = 1; i < data->node->length.size(); i++)
        {
            high_dimension *= data->node->length[i];
        }
    }
    rocfft_precision precision = data->node->precision;
    //
    size_t blocks = (hermitian_size - 1) / 512 + 1;

    if(data->node->length.size() > 3)
        std::cout << "Error: dimension larger than 3, which is not handled" << std::endl;

    size_t dim_1 = 1, dim_2 = 1;
    if(data->node->length.size() >= 2)
        dim_1 = data->node->length[1];
    if(data->node->length.size() == 3)
        dim_2 = data->node->length[2];

    if(high_dimension > 65535 || batch > 65535)
        printf("2D and 3D or batch is too big; not implemented\n");
    // the z dimension is used for batching,
    // if 2D or 3D, the number of blocks along y will multiple high dimensions
    // notice the maximum # of thread blocks in y & z is 65535 according to HIP &&
    // CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    /*float2* tmp; tmp = (float2*)malloc(sizeof(float2)*input_distance*batch);
  hipMemcpy(tmp, input_buffer, sizeof(float2)*input_distance*batch,
  hipMemcpyDeviceToHost);

  printf("herm size %d, dim0 %d, dim1 %d, dim2 %d\n", hermitian_size, dim_0,
  dim_1, dim_2);
  printf("input_stride %d output_stride %d input_distance %d output_distance
  %d\n", input_stride, output_stride, input_distance, output_distance);

  //for(size_t j=0;j<data->node->length[1]; j++)
  size_t j = 0;
  {
      for(size_t i=0; i<input_stride; i++)
      {
          printf("kernel input[%zu][%zu]=(%f, %f) \n", j, i, tmp[j*input_stride
  + i].x, tmp[j*input_stride + i].y);
      }
  }*/

    if(precision == rocfft_precision_single)
        hipLaunchKernelGGL(hermitian2complex_kernel<float2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           hermitian_size,
                           dim_0,
                           dim_1,
                           dim_2,
                           input_stride,
                           output_stride,
                           (float2*)input_buffer,
                           input_distance,
                           (float2*)output_buffer,
                           output_distance);
    else
        hipLaunchKernelGGL(hermitian2complex_kernel<double2>,
                           grid,
                           threads,
                           0,
                           rocfft_stream,
                           hermitian_size,
                           dim_0,
                           dim_1,
                           dim_2,
                           input_stride,
                           output_stride,
                           (double2*)input_buffer,
                           input_distance,
                           (double2*)output_buffer,
                           output_distance);

    /*float2* tmpo; tmpo = (float2*)malloc(sizeof(float2)*output_distance*batch);
  hipMemcpy(tmpo, output_buffer, sizeof(float2)*output_distance*batch,
  hipMemcpyDeviceToHost);

  {
      for(size_t i=0; i<output_stride; i++)
      {
          printf("kernel output[%zu][%zu]=(%f, %f) \n", j, i,
  tmpo[j*output_stride + i].x, tmpo[j*output_stride + i].y);
      }
  }*/
    return;
}
