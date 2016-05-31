#ifndef ROCFFT_ACCURACY_TEST_COMMON_H
#define ROCFFT_ACCURACY_TEST_COMMON_H

#include <gtest/gtest.h>
#include <math.h>
#include <complex>
#include <rocfft_transpose.h>
#include <hip_runtime.h>

template<typename T>
void transpose_reference(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size, T *input_matrix, T *output_matrix)
{
    //transpose per batch
    for(size_t b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                output_matrix[b*input_col_size*output_leading_dim_size + j*output_leading_dim_size + i] = input_matrix[b*input_row_size*input_leading_dim_size + i*input_leading_dim_size + j];
            }
        }
    }
}

template<typename T>
rocfft_transpose_status create_transpose_plan_test(rocfft_transpose_plan &plan, rocfft_transpose_array_type array_type, rocfft_transpose_placement placement, std::vector<size_t> lengths, std::vector<size_t> in_stride, std::vector<size_t> out_stride, size_t in_dist, size_t out_dist, size_t batch_size);

template<>
rocfft_transpose_status create_transpose_plan_test<float>(rocfft_transpose_plan &plan, rocfft_transpose_array_type array_type, rocfft_transpose_placement placement, std::vector<size_t> lengths, std::vector<size_t> in_stride, std::vector<size_t> out_stride, size_t in_dist, size_t out_dist, size_t batch_size)
{
    return rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_single, array_type, placement,
                                 lengths.size(), lengths.data(), in_stride.data(), out_stride.data(), in_dist, out_dist, batch_size, NULL);
}

template<>
rocfft_transpose_status create_transpose_plan_test<double>(rocfft_transpose_plan &plan, rocfft_transpose_array_type array_type, rocfft_transpose_placement placement, std::vector<size_t> lengths, std::vector<size_t> in_stride, std::vector<size_t> out_stride, size_t in_dist, size_t out_dist, size_t batch_size)
{
    return rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_double, array_type, placement,
                                 lengths.size(), lengths.data(), in_stride.data(), out_stride.data(), in_dist, out_dist, batch_size, NULL);
}

template<>
rocfft_transpose_status create_transpose_plan_test<float2>(rocfft_transpose_plan &plan, rocfft_transpose_array_type array_type, rocfft_transpose_placement placement, std::vector<size_t> lengths, std::vector<size_t> in_stride, std::vector<size_t> out_stride, size_t in_dist, size_t out_dist, size_t batch_size)
{
    return rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_single, array_type, placement,
                                 lengths.size(), lengths.data(), in_stride.data(), out_stride.data(), in_dist, out_dist, batch_size, NULL);
}

template<>
rocfft_transpose_status create_transpose_plan_test<double2>(rocfft_transpose_plan &plan, rocfft_transpose_array_type array_type, rocfft_transpose_placement placement, std::vector<size_t> lengths, std::vector<size_t> in_stride, std::vector<size_t> out_stride, size_t in_dist, size_t out_dist, size_t batch_size)
{
    return rocfft_transpose_plan_create(&plan, rocfft_transpose_precision_double, array_type, placement,
                                 lengths.size(), lengths.data(), in_stride.data(), out_stride.data(), in_dist, out_dist, batch_size, NULL);
}


template<typename T>
void real_transpose_test(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size, T *input_matrix, T *output_matrix)
{
    size_t output_row_size = input_col_size;
    size_t output_col_size = input_row_size;
   
    hipError_t err;
    //create device memory
    T *input_matrix_device, *output_matrix_device;

    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_leading_dim_size * sizeof(T));
    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(T));
    //copy data to device
    err = hipMemcpy(input_matrix_device, input_matrix, batch_size * input_row_size * input_leading_dim_size * sizeof(T), hipMemcpyHostToDevice);
    
    //create transpose only plan
    rocfft_transpose_status status;
    rocfft_transpose_plan plan = NULL;
    std::vector<size_t> lengths = {(size_t)input_col_size, (size_t)input_row_size};
    std::vector<size_t> in_stride = {1, input_leading_dim_size};
    std::vector<size_t> out_stride = {1, output_leading_dim_size};
    size_t in_dist = input_row_size * input_leading_dim_size;
    size_t out_dist = input_col_size * output_leading_dim_size;
    
    status = create_transpose_plan_test<T>(plan, rocfft_transpose_array_type_real_to_real, rocfft_transpose_placement_notinplace, lengths, in_stride, out_stride, in_dist, out_dist, batch_size);
    status = rocfft_transpose_execute(plan, (void**)&input_matrix_device, (void**)&output_matrix_device, NULL);
    
    //destroy plan
    status = rocfft_transpose_plan_destroy(plan);

    //copy data from device to host
    err = hipMemcpy(output_matrix, output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(T), hipMemcpyDeviceToHost);

    hipFree(input_matrix_device);
    hipFree(output_matrix_device);
}

//writting a seperate function for complex data type (both planar and interleaved)
template<typename T, rocfft_transpose_array_type array_type>
void complex_transpose_test(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size, T *input_matrix, T *output_matrix)
{
    size_t output_row_size = input_col_size;
    size_t output_col_size = input_row_size;

    hipError_t err;
    //create device memory
    double2 *input_matrix_device, *output_matrix_device;

    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_leading_dim_size * sizeof(double2));
    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(double2));
    //copy data to device
    err = hipMemcpy(input_matrix_device, input_matrix, batch_size * input_row_size * input_leading_dim_size * sizeof(double2), hipMemcpyHostToDevice);

    //create transpose only plan
    rocfft_transpose_status status;
    rocfft_transpose_plan plan = NULL;
    std::vector<size_t> lengths = {(size_t)input_col_size, (size_t)input_row_size};
    std::vector<size_t> in_stride = {1, input_leading_dim_size};
    std::vector<size_t> out_stride = {1, output_leading_dim_size};
    size_t in_dist = input_row_size * input_leading_dim_size;
    size_t out_dist = output_row_size * output_leading_dim_size;

    status = create_transpose_plan_test<double2>(plan, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved, rocfft_transpose_placement_notinplace, lengths, in_stride, out_stride, in_dist, out_dist, batch_size);
    status = rocfft_transpose_execute(plan, (void**)&input_matrix_device, (void**)&output_matrix_device, NULL);

    //destroy plan
    status = rocfft_transpose_plan_destroy(plan);

    //copy data from device to host
    err = hipMemcpy(output_matrix, output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(double2), hipMemcpyDeviceToHost);

    hipFree(input_matrix_device);
    hipFree(output_matrix_device);
}

template<>
void complex_transpose_test<std::complex<float>, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved>(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size, std::complex<float> *input_matrix, std::complex<float> *output_matrix)
{
    size_t output_row_size = input_col_size;
    size_t output_col_size = input_row_size;

    hipError_t err;
    //create device memory
    float2 *input_matrix_device, *output_matrix_device;

    err = hipMalloc(&input_matrix_device, batch_size * input_row_size * input_leading_dim_size * sizeof(float2));
    err = hipMalloc(&output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(float2));
    //copy data to device
    err = hipMemcpy(input_matrix_device, input_matrix, batch_size * input_row_size * input_leading_dim_size * sizeof(float2), hipMemcpyHostToDevice);

    //create transpose only plan
    rocfft_transpose_status status;
    rocfft_transpose_plan plan = NULL;
    std::vector<size_t> lengths = {(size_t)input_col_size, (size_t)input_row_size};
    std::vector<size_t> in_stride = {1, input_leading_dim_size};
    std::vector<size_t> out_stride = {1, output_leading_dim_size};
    size_t in_dist = input_row_size * input_leading_dim_size;
    size_t out_dist = output_row_size * output_leading_dim_size;

    status = create_transpose_plan_test<float2>(plan, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved, rocfft_transpose_placement_notinplace, lengths, in_stride, out_stride, in_dist, out_dist, batch_size);
    status = rocfft_transpose_execute(plan, (void**)&input_matrix_device, (void**)&output_matrix_device, NULL);

    //destroy plan
    status = rocfft_transpose_plan_destroy(plan);

    //copy data from device to host
    err = hipMemcpy(output_matrix, output_matrix_device, batch_size * output_row_size * output_leading_dim_size * sizeof(float2), hipMemcpyDeviceToHost);

    hipFree(input_matrix_device);
    hipFree(output_matrix_device);
}
#endif
