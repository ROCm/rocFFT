/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <complex>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include "rocfft_transpose.h"


int main()
{
    std::cout << "rocfft outplace transpose test" << std::endl;
    size_t input_row_size = 1024;
    size_t input_col_size = 1024;
    size_t batch_size = 3;

    std::cout << "input_row_size = " << input_row_size << ", input_col_size = " << input_col_size << std::endl;
    std::cout << "batch_size = " << batch_size << std::endl;

    size_t output_row_size = input_col_size;
    size_t output_col_size = input_row_size;

    size_t input_leading_dim_size = input_col_size;
    size_t output_leading_dim_size = output_col_size;
    //allocate host memory
    std::vector<std::complex<float> > input_matrix(input_row_size * input_col_size * batch_size);
    std::vector<std::complex<float> > output_matrix(output_row_size * output_col_size * batch_size, 0);

    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_col_size + i * input_col_size + j] =
                std::complex<float>(b * input_row_size * input_col_size + i * input_col_size +j, b * input_row_size * input_col_size + i * input_col_size +j);
            }
        }
    }


    //print some input matrix
    std::cout << "print some input values: " << std::endl;
    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            std::cout << input_matrix[i * input_col_size + j + 0*input_col_size*input_row_size] << " ";
        }
        std::cout << std::endl;
    }

    hipError_t err;
    //create device memory
    float2 *input_matrix_device, *output_matrix_device;

    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_col_size * sizeof(float2));
    if(err == hipSuccess)
       std::cout << "input_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "input_matrix_device allocation was unsuccessful" << std::endl;

    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float2));
    if(err == hipSuccess)
       std::cout << "output_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "output_matrix_device allocation was unsuccessful" << std::endl;

    //copy data to device
    err = hipMemcpy(input_matrix_device, input_matrix.data(), batch_size * input_row_size * input_col_size * sizeof(float2), hipMemcpyHostToDevice);
    if(err == hipSuccess)
       std::cout << "input_matrix_device copy host to device was successful" << std::endl;
    else
       std::cout << "input_matrix_device copy host to device was unsuccessful" << std::endl;

    //create transpose only plan
    rocfft_transpose_status status;
    rocfft_transpose_plan plan = NULL;
    std::vector<size_t> lengths = {(size_t)input_col_size, (size_t)input_row_size};
    std::vector<size_t> in_stride = {1, input_col_size};
    std::vector<size_t> out_stride = {1, output_col_size};
    size_t in_dist = input_col_size * input_row_size;
    size_t out_dist = output_col_size * output_row_size;

    status = rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_single, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved, rocfft_transpose_placement_notinplace,
                                 lengths.size(), lengths.data(), in_stride.data(), out_stride.data(), in_dist, out_dist, batch_size, NULL);
    if(status == rocfft_transpose_status_success)
       std::cout << "rocfft_transpose_plan_create was successful" << std::endl;
    else
       std::cout << "rocfft_transpose_plan_create was unsuccessful" << std::endl;

    //execute plan
    status = rocfft_transpose_execute(plan, (void**)&input_matrix_device, (void**)&output_matrix_device, NULL);
    if(status == rocfft_transpose_status_success)
       std::cout << "rocfft_transpose_execute was successful" << std::endl;
    else
       std::cout << "rocfft_transpose_execute was unsuccessful" << std::endl;


    //destroy plan
    status = rocfft_transpose_plan_destroy(plan);
    if(status == rocfft_transpose_status_success)
       std::cout << "rocfft_transpose_plan_destroy was successful" << std::endl;
    else
       std::cout << "rocfft_transpose_plan_destroy was unsuccessful" << std::endl;

    //copy data from device to host
    err = hipMemcpy(output_matrix.data(), output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float2), hipMemcpyDeviceToHost);
    if(err == hipSuccess)
       std::cout << "output_matrix_device copy device to host was successful" << std::endl;
    else
       std::cout << "output_matrix_device copy device to host was unsuccessful" << std::endl;

    //print output matrix
    std::cout << "print some output values: " << std::endl;
    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 16; j++)
        {
            std::cout << output_matrix[i * output_col_size + j + 0*input_col_size*input_row_size] << " ";
        }
        std::cout << std::endl;
    }

    //check result
    bool passed = true;
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                if(input_matrix[b * input_col_size*input_row_size + i * input_col_size + j] != output_matrix[b * input_col_size*input_row_size + j * input_row_size + i])
                {
                    passed = false;
                    break;
                }
            }
        }
    }
    if(passed)
       std::cout << "correctness PASSED" << std::endl;
    else
       std::cout << "correctness FAILED" << std::endl;

    hipFree(input_matrix_device);
    hipFree(output_matrix_device);
}
