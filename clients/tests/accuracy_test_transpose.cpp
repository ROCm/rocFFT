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
void normal_2d_out_place_real_to_real()
{
    std::cout << "inside normal 2d" << std::endl;
    EXPECT_EQ(1.0f, 1.0f);
}

TEST_F(accuracy_test_transpose_single, normal_2D_outplace_real_to_real)
{
    normal_2d_out_place_real_to_real<float>();
    //catch(const std::exception& err) {handle_exception(err);}
}

}
