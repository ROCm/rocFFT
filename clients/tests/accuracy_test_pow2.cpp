
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "rocfft.h"
#include "test_constants.h"
#include "rocfft_against_fftw.h"
#include "fftw_transform.h"



/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow2_single : public ::testing::Test {
protected:
    accuracy_test_pow2_single(){}
    virtual ~accuracy_test_pow2_single(){}
    virtual void SetUp(){}
    virtual void TearDown(){
    }
};

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
class accuracy_test_pow2_double : public ::testing::Test {
protected:
    accuracy_test_pow2_double(){}
    virtual ~accuracy_test_pow2_double(){}
    virtual void SetUp(){}
    virtual void TearDown(){
    }
};

namespace power2
{
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ normal 1D ^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************


// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( normal2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_forward;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void len65536_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( 65536 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_inverse;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, len65536_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}

// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( normal2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_inverse;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { normal_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************


// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************



// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ small 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void small_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( small2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_forward;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { small_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void small_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( small2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_inverse;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, small_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { small_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************



// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ large 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^ huge 1D ^^^^^^^^^^^^^^^^^^^^^^^ //
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ //

// *****************************************************
// *****************************************************
//TESTS disabled by default since they take a long time to execute
//TO enable this tests
//1. make sure ENV CLFFT_REQUEST_LIB_NOMEMALLOC=1
//2. pass --gtest_also_run_disabled_tests to TEST.exe

#define CLFFT_TEST_HUGE
#ifdef CLFFT_TEST_HUGE

#define HUGE_TEST_MAKE(test_name, len, bat) \
template< class T, class fftw_T > \
void test_name() \
{ \
    std::vector<size_t> lengths; \
    lengths.push_back( len ); \
    size_t batch = bat; \
\
    std::vector<size_t> input_strides; \
    std::vector<size_t> output_strides; \
    size_t input_distance = 0; \
    size_t output_distance = 0; \
    rocfft_array_type in_array_type = rocfft_array_type_complex_planar; \
    rocfft_array_type out_array_type = rocfft_array_type_complex_planar; \
    rocfft_result_placement placeness = rocfft_placement_inplace; \
    rocfft_transform_type transform_type = rocfft_transform_type_complex_forward; \
\
    data_pattern pattern = sawtooth; \
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness ); \
}

#define SP_HUGE_TEST(test_name, len, bat) \
\
    HUGE_TEST_MAKE(test_name, len, bat) \
\
    TEST_F(accuracy_test_pow2_single, test_name) \
    { \
        try { test_name< float,  fftwf_complex >(); } \
        catch( const std::exception& err ) { handle_exception(err);    } \
    }

#define DP_HUGE_TEST(test_name, len, bat) \
\
    HUGE_TEST_MAKE(test_name, len, bat) \
\
    TEST_F(accuracy_test_pow2_double, test_name) \
    { \
        try { test_name< double,  fftw_complex >(); } \
        catch( const std::exception& err ) { handle_exception(err);    } \
    }

SP_HUGE_TEST( DISABLED_huge_sp_test_1, 1048576,    11 )
SP_HUGE_TEST( DISABLED_huge_sp_test_2, 1048576*2,  7  )
SP_HUGE_TEST( DISABLED_huge_sp_test_3, 1048576*4,  3  )
SP_HUGE_TEST( DISABLED_huge_sp_test_4, 1048576*8,  5  )
SP_HUGE_TEST( DISABLED_huge_sp_test_5, 1048576*16, 3  )
SP_HUGE_TEST( DISABLED_huge_sp_test_6, 1048576*32, 2  )
SP_HUGE_TEST( DISABLED_huge_sp_test_7, 1048576*64, 1  )

DP_HUGE_TEST( DISABLED_huge_dp_test_1, 524288,    11 )
DP_HUGE_TEST( DISABLED_huge_dp_test_2, 524288*2,  7  )
DP_HUGE_TEST( DISABLED_huge_dp_test_3, 524288*4,  3  )
DP_HUGE_TEST( DISABLED_huge_dp_test_4, 524288*8,  5  )
DP_HUGE_TEST( DISABLED_huge_dp_test_5, 524288*16, 3  )
DP_HUGE_TEST( DISABLED_huge_dp_test_6, 524288*32, 2  )
DP_HUGE_TEST( DISABLED_huge_dp_test_7, 524288*64, 1  )

SP_HUGE_TEST( DISABLED_large_sp_test_1, 8192,    11 )
SP_HUGE_TEST( DISABLED_large_sp_test_2, 8192*2,  7  )
SP_HUGE_TEST( DISABLED_large_sp_test_3, 8192*4,  3  )
SP_HUGE_TEST( DISABLED_large_sp_test_4, 8192*8,  5  )
SP_HUGE_TEST( DISABLED_large_sp_test_5, 8192*16, 3  )
SP_HUGE_TEST( DISABLED_large_sp_test_6, 8192*32, 21  )
SP_HUGE_TEST( DISABLED_large_sp_test_7, 8192*64, 17  )

DP_HUGE_TEST( DISABLED_large_dp_test_1, 4096,    11 )
DP_HUGE_TEST( DISABLED_large_dp_test_2, 4096*2,  7  )
DP_HUGE_TEST( DISABLED_large_dp_test_3, 4096*4,  3  )
DP_HUGE_TEST( DISABLED_large_dp_test_4, 4096*8,  5  )
DP_HUGE_TEST( DISABLED_large_dp_test_5, 4096*16, 3  )
DP_HUGE_TEST( DISABLED_large_dp_test_6, 4096*32, 21  )
DP_HUGE_TEST( DISABLED_large_dp_test_7, 4096*64, 17  )

#endif

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void large_1D_forward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( large2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_forward;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_forward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_1D_forward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************
template< class T, class fftw_T >
void large_1D_backward_in_place_complex_interleaved_to_complex_interleaved()
{
    std::vector<size_t> lengths;
    lengths.push_back( large2 );
    size_t batch = 1;
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    size_t input_distance = 0;
    size_t output_distance = 0;
    rocfft_array_type in_array_type = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;
    rocfft_result_placement placeness = rocfft_placement_inplace;
    rocfft_transform_type transform_type = rocfft_transform_type_complex_inverse;

    data_pattern pattern = sawtooth;
    complex_to_complex<T, fftw_T>( pattern, transform_type, lengths, batch, input_strides, output_strides, input_distance, output_distance, in_array_type, out_array_type, placeness );
}

TEST_F(accuracy_test_pow2_single, large_1D_backward_in_place_complex_interleaved_to_complex_interleaved)
{
    try { large_1D_backward_in_place_complex_interleaved_to_complex_interleaved< float,  fftwf_complex >(); }
    catch( const std::exception& err ) { handle_exception(err);    }
}


// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************


// *****************************************************
// *****************************************************


// *****************************************************
// *****************************************************


// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************

// *****************************************************
// *****************************************************
} //namespace
