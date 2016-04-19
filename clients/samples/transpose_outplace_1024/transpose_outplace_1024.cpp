#include <iostream>
#include <vector>
#include <hip_runtime.h>
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
    
    //allocate host memory
    std::vector<float> input_matrix(input_row_size * input_col_size * batch_size);
    std::vector<float> output_matrix(output_row_size * output_col_size * batch_size, 0);
    
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_col_size + i * input_col_size + j] = 
                (float)(b * input_row_size * input_col_size + i * input_col_size +j);
            } 
        }
    }

    hipError_t err;
    //create device memory
    float *input_matrix_device, *output_matrix_device;

    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_col_size * sizeof(float));
    if(err == hipSuccess)
       std::cout << "input_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "input_matrix_device allocation was unsuccessful" << std::endl;

    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float));
    if(err == hipSuccess)
       std::cout << "output_matrix_device allocation was successful" << std::endl;
    else
       std::cout << "output_matrix_device allocation was unsuccessful" << std::endl;
    
    //copy data to device
    err = hipMemcpy(input_matrix_device, input_matrix.data(), batch_size * input_row_size * input_col_size * sizeof(float), hipMemcpyHostToDevice);
    if(err == hipSuccess)
       std::cout << "input_matrix_device copy host to device was successful" << std::endl;
    else
       std::cout << "input_matrix_device copy host to device was unsuccessful" << std::endl;

    //create rocfft transpose buffer
    rocfft_transpose_buffer input_buffer = NULL;
    rocfft_transpose_buffer_create_with_ptr(&input_buffer, input_matrix_device);
   
    rocfft_transpose_buffer output_buffer = NULL;
    rocfft_transpose_buffer_create_with_ptr(&output_buffer, output_matrix_device);

    //create transpose only plan
    rocfft_transpose_plan plan = NULL;
    std::vector<size_t> lengths = {(size_t)input_col_size, (size_t)input_row_size};
    rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_single, rocfft_transpose_array_type_real, rocfft_transpose_placement_notinplace,
                                 lengths.size(), lengths.data(), batch_size, NULL);

    //execute plan
    rocfft_transpose_status status = rocfft_transpose_execute(plan, &input_buffer, &output_buffer, NULL);
    if(status == rocfft_transpose_status_success)
       std::cout << "rocfft_transpose_execute was successful" << std::endl;
    else
       std::cout << "rocfft_transpose_execute was unsuccessful" << std::endl;


    //destroy plan
    rocfft_transpose_plan_destroy(plan);
    
    //copy data from device to host
    err = hipMemcpy(output_matrix.data(), output_matrix_device, batch_size * output_row_size * output_col_size * sizeof(float), hipMemcpyDeviceToHost);
    if(err == hipSuccess)
       std::cout << "output_matrix_device copy device to host was successful" << std::endl;
    else
       std::cout << "output_matrix_device copy device to host was unsuccessful" << std::endl;

    hipFree(input_matrix_device);
    hipFree(output_matrix_device);
    rocfft_transpose_buffer_destroy(input_buffer);
    rocfft_transpose_buffer_destroy(output_buffer);
}
