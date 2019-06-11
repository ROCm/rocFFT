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

#include "testing_transpose.hpp"
#include <functional>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <tuple>
#include <vector>

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,
// std::tuple is good enough;

typedef std::tuple<vector<size_t>, size_t> transpose_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one
which invalidates the matrix.
Yet, the goal of this file is to verify result correctness not
argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not
necessary
=================================================================== */

// small sizes

// vector of vector, each triple is a {rows, cols, lda, ldb};
// add/delete this list in pairs, like {3, 4, 4, 5}

const vector<vector<size_t>> size_range = {{3, 3, 3, 3},
                                           {3, 3, 4, 4},
                                           {10, 10, 10, 10},
                                           {100, 100, 102, 101},
                                           {1024, 1024, 1024, 1024},
                                           {1119, 111, 2000, 111},
                                           {5000, 5000, 6000, 7000}

};

static size_t batch_range[] = {1, 100};

/* ===============Google Unit
 * Test==================================================== */

class transpose_gtest : public ::TestWithParam<transpose_tuple>
{
protected:
    transpose_gtest() {}
    virtual ~transpose_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*
TEST_P(transpose_gtest, float2)
{

    transpose_tuple tup = GetParam();

    vector<size_t> size = std::get<0>(tup);

    size_t rows = size[0];
    size_t cols = size[1];

    size_t lda = size[2];
    size_t ldb = size[3];

    size_t batch_count = std::get<1>(tup);

    rocfft_status status = testing_transpose<float2>( rows, cols, lda, ldb,
batch_count);

    // if not success, then the input argument is problematic, so detect the
error message
    if(status != rocfft_status_success){
        if(lda < rows){
            EXPECT_EQ(rocfft_status_invalid_dimensions, status);
        }
        else if(ldb < cols){
            EXPECT_EQ(rocfft_status_invalid_dimensions, status);
        }
    }
}




//notice we are using vector of vector
//so each elment in xxx_range is a avector,
//ValuesIn take each element (a vector) and combine them and feed them to test_p


INSTANTIATE_TEST_CASE_P(rocfft_transpose,
                        transpose_gtest,
                        Combine(
                            ValuesIn(size_range), ValuesIn(batch_range)
                        )
                       );
*/
