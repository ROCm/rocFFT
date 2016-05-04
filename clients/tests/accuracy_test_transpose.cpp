#include <gtest/gtest.h>
#include <math.h>
#include "accuracy_test_common.h"
#include <iostream>

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

namespace transpose_test_namespace
{

template<typename T>
void normal_2d_out_place_real_to_real(size_t input_row_size, size_t input_col_size, size_t batch_size)
{
    //allocate host memory
    std::vector<T> input_matrix(input_row_size * input_col_size * batch_size);
    std::vector<T> output_matrix(input_row_size * input_col_size * batch_size, 0);
    std::vector<T> reference_output_matrix(input_row_size * input_col_size * batch_size, 0);
    //init the input matrix
    for(int b = 0; b < batch_size; b++)
    {
        for(int i = 0; i < input_row_size; i++)
        {
            for(int j = 0; j < input_col_size; j++)
            {
                input_matrix[b * input_row_size * input_col_size + i * input_col_size + j] = 
                (T)(b * input_row_size * input_col_size + i * input_col_size +j);
            } 
        }
    }
    
    real_transpose_test<T>(input_row_size, input_col_size, batch_size, input_matrix.data(), output_matrix.data());
    real_transpose_reference<T>(input_row_size, input_col_size, batch_size, input_matrix.data(), reference_output_matrix.data());
    EXPECT_EQ(reference_output_matrix, output_matrix);
}

TEST_F(accuracy_test_transpose_single, normal_2D_outplace_real_to_real_1024_1024_1)
{
    normal_2d_out_place_real_to_real<float>(1024, 1024, 1);
    //catch(const std::exception& err) {handle_exception(err);}
}

TEST_F(accuracy_test_transpose_double, normal_2D_outplace_real_to_real_1024_1024_1)
{
    normal_2d_out_place_real_to_real<double>(1024, 1024, 1);
    //catch(const std::exception& err) {handle_exception(err);}
}

}
