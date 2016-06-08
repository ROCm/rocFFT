#include <gtest/gtest.h>
#include <math.h>
#include "accuracy_test_common.h"
#include "test_exception.h"
#include <iostream>
#include <vector>

namespace transpose_test_namespace
{

const int matrix_size_small_pow2_range[] = {128, 256};
const int batch_size_small_range[] = {1, 3};
const int matrix_size_middle_pow2_range[] = {512, 1024};
const int matrix_size_middle_range[] = {520, 999};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_transpose_single : public ::testing::Test {
protected:
	accuracy_test_transpose_single(){}
	virtual ~accuracy_test_transpose_single(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_transpose_double : public ::testing::Test {
protected:
	accuracy_test_transpose_double(){}
	virtual ~accuracy_test_transpose_double(){}
	virtual void SetUp(){}
	virtual void TearDown(){
	}
};

template<typename T>
void normal_2d_out_place_real_to_real(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{
    if(input_row_size < 1 || input_col_size < 1 || batch_size < 1)
    {
        throw std::runtime_error("matrix size and batch size cannot be smaller than 1");
    }
    //allocate host memory
    std::vector<T> input_matrix(input_row_size * input_leading_dim_size * batch_size);
    std::vector<T> output_matrix(output_leading_dim_size * input_col_size * batch_size, 0);
    std::vector<T> reference_output_matrix(output_leading_dim_size * input_col_size * batch_size, 1);
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] = 
                static_cast<T>(b * input_row_size * input_col_size + i * input_col_size +j);
            } 
        }
    }
    
    real_transpose_test<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), output_matrix.data());
    transpose_reference<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), reference_output_matrix.data());
    EXPECT_EQ(reference_output_matrix, output_matrix);
}

template<typename T, rocfft_transpose_array_type array_type>
void normal_2d_out_place_complex_to_complex(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{
    if(input_row_size < 1 || input_col_size < 1 || batch_size < 1)
    {
        throw std::runtime_error("matrix size and batch size cannot be smaller than 1");
    }
    //allocate host memory in row major
    std::vector<std::complex<T> > input_matrix(input_row_size * input_leading_dim_size * batch_size, std::complex<T>(0, 0));
    std::vector<std::complex<T> > output_matrix(output_leading_dim_size * input_col_size * batch_size, std::complex<T>(0, 0));
    std::vector<std::complex<T> > reference_output_matrix(output_leading_dim_size * input_col_size * batch_size, std::complex<T>(1, 1));
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                std::complex<T>(b * input_row_size * input_col_size + i * input_col_size + j, b * input_row_size * input_col_size + i * input_col_size + j + 1);
            }
        }
    }
    
    complex_transpose_test<std::complex<T>, array_type>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), output_matrix.data());
    transpose_reference<std::complex<T> >(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), reference_output_matrix.data());
    EXPECT_EQ(reference_output_matrix, output_matrix);
}

template<typename T>
void normal_2d_out_place_complex_planar_to_complex_planar(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{

    if(input_row_size < 1 || input_col_size < 1 || batch_size < 1)
    {
        throw std::runtime_error("matrix size and batch size cannot be smaller than 1");
    }
    //allocate host memory in row major
    std::vector<T> input_matrix_real(input_row_size * input_leading_dim_size * batch_size, 0);
    std::vector<T> input_matrix_imag(input_row_size * input_leading_dim_size * batch_size, 0);

    std::vector<T> output_matrix_real(output_leading_dim_size * input_col_size * batch_size, 0);
    std::vector<T> output_matrix_imag(output_leading_dim_size * input_col_size * batch_size, 0);

    std::vector<T> reference_output_matrix_real(output_leading_dim_size * input_col_size * batch_size, 1);
    std::vector<T> reference_output_matrix_imag(output_leading_dim_size * input_col_size * batch_size, 1);
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix_real[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                static_cast<T>(b * input_row_size * input_col_size + i * input_col_size +j);
                input_matrix_imag[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                static_cast<T>(b * input_row_size * input_col_size + i * input_col_size +j + 1);
            }
        }
    }

    complex_planar_to_planar_transpose_test<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix_real.data(), input_matrix_imag.data(), output_matrix_real.data(), output_matrix_imag.data());
    transpose_reference<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix_real.data(), reference_output_matrix_real.data());
    transpose_reference<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix_imag.data(), reference_output_matrix_imag.data());
    
    EXPECT_EQ(reference_output_matrix_real, output_matrix_real);
    EXPECT_EQ(reference_output_matrix_imag, output_matrix_imag);

}

template<typename T>
void normal_2d_out_place_complex_planar_to_complex_interleaved(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{
    if(input_row_size < 1 || input_col_size < 1 || batch_size < 1)
    {
        throw std::runtime_error("matrix size and batch size cannot be smaller than 1");
    }
    //allocate host memory in row major
    std::vector<T> input_matrix_real(input_row_size * input_leading_dim_size * batch_size, 0);
    std::vector<T> input_matrix_imag(input_row_size * input_leading_dim_size * batch_size, 0);

    std::vector<std::complex<T>> output_matrix(output_leading_dim_size * input_col_size * batch_size, std::complex<T>(0,0));
    std::vector<std::complex<T>> reference_output_matrix(output_leading_dim_size * input_col_size * batch_size, std::complex<T>(1,1));
    
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix_real[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                static_cast<T>(b * input_row_size * input_col_size + i * input_col_size +j);
                input_matrix_imag[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                static_cast<T>(b * input_row_size * input_col_size + i * input_col_size +j + 1);
            }
        }
    }
    
    complex_planar_to_interleaved_transpose_test<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix_real.data(), input_matrix_imag.data(), output_matrix.data());
    transpose_complex_planar_to_complex_interleaved_reference<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix_real.data(), input_matrix_imag.data(), reference_output_matrix.data());
    
    EXPECT_EQ(reference_output_matrix, output_matrix);
}

template<typename T>
void normal_2d_out_place_complex_interleaved_to_complex_planar(size_t input_row_size, size_t input_col_size, size_t input_leading_dim_size, size_t output_leading_dim_size, size_t batch_size)
{
    if(input_row_size < 1 || input_col_size < 1 || batch_size < 1)
    {
        throw std::runtime_error("matrix size and batch size cannot be smaller than 1");
    }
    //allocate host memory in row major
    std::vector<std::complex<T>> input_matrix(input_row_size * input_leading_dim_size * batch_size, std::complex<T>(0,0));
    
    std::vector<T> output_matrix_real(output_leading_dim_size * input_col_size * batch_size, 0);
    std::vector<T> output_matrix_imag(output_leading_dim_size * input_col_size * batch_size, 0);

    std::vector<T> reference_output_matrix_real(output_leading_dim_size * input_col_size * batch_size, 0);
    std::vector<T> reference_output_matrix_imag(output_leading_dim_size * input_col_size * batch_size, 0);

    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_leading_dim_size + i * input_leading_dim_size + j] =
                std::complex<T>(b * input_row_size * input_col_size + i * input_col_size + j, b * input_row_size * input_col_size + i * input_col_size + j + 1);
            }
        }
    }
    
    complex_interleaved_to_planar_transpose_test<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), output_matrix_real.data(), output_matrix_imag.data());
    transpose_complex_interleaved_to_complex_planar_reference<T>(input_row_size, input_col_size, input_leading_dim_size, output_leading_dim_size, batch_size, input_matrix.data(), reference_output_matrix_real.data(), reference_output_matrix_imag.data());
    
    EXPECT_EQ(reference_output_matrix_real, output_matrix_real);
    EXPECT_EQ(reference_output_matrix_imag, output_matrix_imag);
}

typedef struct TestParams{
size_t row_size;
size_t column_size;
size_t batch_size;
}TestParams;

class transpose_test : public testing::TestWithParam<
        std::tuple<
        int, //input row size
        int, //input column size
        int  //batch size
        > > 
{
public:
    void getParams(TestParams *params)
    {
        memset(params, 0, sizeof(TestParams));
        params->row_size = row_size;
        params->column_size = column_size;
        params->batch_size = batch_size;
    }
protected:
    virtual void SetUp()
    {
        row_size = std::get<0>(GetParam());
        column_size = std::get<1>(GetParam());
        batch_size = std::get<2>(GetParam());
    }
private:
    size_t row_size;
    size_t column_size;
    size_t batch_size;
};

INSTANTIATE_TEST_CASE_P(accuracy_test_transpose_small_pow2_range, transpose_test, testing::Combine(
    testing::ValuesIn(matrix_size_small_pow2_range), testing::ValuesIn(matrix_size_small_pow2_range), testing::ValuesIn(batch_size_small_range)));

INSTANTIATE_TEST_CASE_P(accuracy_test_transpose_middle_pow2_range, transpose_test, testing::Combine(
    testing::ValuesIn(matrix_size_middle_pow2_range), testing::ValuesIn(matrix_size_middle_pow2_range), testing::ValuesIn(batch_size_small_range)));

INSTANTIATE_TEST_CASE_P(accuracy_test_transpose_middle_range, transpose_test, testing::Combine(
    testing::ValuesIn(matrix_size_middle_range), testing::ValuesIn(matrix_size_middle_range), testing::ValuesIn(batch_size_small_range)));

// real to real single tests
TEST_P(transpose_test, outplace_transpose_single)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_real_to_real<float>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}
// real to real double tests
TEST_P(transpose_test, outplace_transpose_double)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_real_to_real<double>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}
// complex interleaved to complex interleaved single tests
TEST_P(transpose_test, outplace_transpose_single_complex_interleaved_to_interleaved)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_to_complex<float, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}
// complex interleaved to complex interleaved double tests
TEST_P(transpose_test, outplace_transpose_double_complex_interleaved_to_interleaved)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_to_complex<double, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}

// complex planar to complex planar single tests
TEST_P(transpose_test, outplace_transpose_single_complex_planar_to_planar)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_planar_to_complex_planar<float>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}
// complex planar to complex planar double tests
TEST_P(transpose_test, outplace_transpose_double_complex_planar_to_planar)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_planar_to_complex_planar<double>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}

//complex planar to complex interleaved single tests
TEST_P(transpose_test, outplace_transpose_single_complex_planar_to_interleaved)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_planar_to_complex_interleaved<float>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}
//complex planar to complex interleaved double tests
TEST_P(transpose_test, outplace_transpose_double_complex_planar_to_interleaved)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_planar_to_complex_interleaved<double>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}

//complex interleaved to complex planar single tests
TEST_P(transpose_test, outplace_transpose_single_complex_interleaved_to_complex_planar)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_interleaved_to_complex_planar<float>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}

TEST_P(transpose_test, outplace_transpose_double_complex_interleaved_to_complex_planar)
{
    TestParams params;
    getParams(&params);
    try{ normal_2d_out_place_complex_interleaved_to_complex_planar<double>(params.row_size, params.column_size, params.column_size, params.row_size, params.batch_size); }
    catch(const std::exception &err) { handle_exception(err); }
}


//add some special cases if needed
/*
TEST_F(accuracy_test_transpose_single, normal_2d_outplace_real_to_real_192_192_193_194_1)
{
    try{ normal_2d_out_place_real_to_real<float>(192, 192, 193, 194, 1); }
    catch(const std::exception &err) { handle_exception(err); }
}

TEST_F(accuracy_test_transpose_single, normal_2d_outplace_complex_interleaved_to_complex_interleaved_192_192_193_194_1)
{
    try{ normal_2d_out_place_complex_to_complex<float, rocfft_transpose_array_type_complex_interleaved_to_complex_interleaved>(192, 192, 193, 194, 1); }
    catch(const std::exception &err) { handle_exception(err); }
}
*/

}
