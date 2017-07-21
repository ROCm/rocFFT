
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include <gtest/gtest.h>
#include<math.h>
#include <stdexcept>
#include <vector>

#include "rocfft.h"
#include "test_constants.h"
#include "rocfft_transform.h"
#include "fftw_transform.h"




/*****************************************************
           Complex to Complex
*****************************************************/
// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
template< class T, class fftw_T >
void complex_to_complex( data_pattern pattern, rocfft_transform_type transform_type,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	rocfft_array_type  in_array_type , rocfft_array_type  out_array_type ,
	rocfft_result_placement placeness,
	T scale = 1.0f )
{

	rocfft<T> test_fft( lengths, batch,
		input_strides,
		output_strides,
		input_distance, output_distance,
		in_array_type, out_array_type,
		placeness, transform_type, scale );


	fftw<T, fftw_T> reference( lengths, batch, input_strides, output_strides, placeness, c2c );


	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f, 2.5f );
		reference.set_all_data_to_value( 2.0f, 2.5f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random(); //TODO 
		reference.set_data_to_random(); //TODO
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in complex_to_complex()" );
	}

	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

    // scale is already set in plan create called in constructor of class rocfft
	if( transform_type  == rocfft_transform_type_complex_forward )
	{
		reference.set_forward_transform();
		reference.forward_scale( scale );
	}
	else if( transform_type  == rocfft_transform_type_complex_inverse )
	{
		reference.set_backward_transform();
		reference.backward_scale( scale );
	}
	else{
		throw std::runtime_error( "invalid transform_type  in complex_to_complex()" );
	}

	reference.transform();
	test_fft.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );

}

/*****************************************************
           Real to Complex
*****************************************************/

// dimension is inferred from lengths.size()
// tightly packed is inferred from strides.empty()
// input layout is always real
template< class T, class fftw_T >
void real_to_complex( data_pattern pattern, rocfft_transform_type transform_type,
	std::vector<size_t> lengths, size_t batch,
	std::vector<size_t> input_strides, std::vector<size_t> output_strides,
	size_t input_distance, size_t output_distance,
	rocfft_array_type  in_array_type , rocfft_array_type  out_array_type ,
	rocfft_result_placement placeness,
	T scale = 1.0f )
{

	rocfft<T> test_fft( lengths, batch,
		input_strides,
		output_strides,
		input_distance, output_distance,
		in_array_type, out_array_type,
		placeness, transform_type, scale );

	fftw<T, fftw_T> reference( lengths, batch, input_strides, output_strides, placeness, r2c );

	if( pattern == sawtooth )
	{
		test_fft.set_input_to_sawtooth( 1.0f );
		reference.set_data_to_sawtooth( 1.0f );
	}
	else if( pattern == value )
	{
		test_fft.set_input_to_value( 2.0f );
		reference.set_all_data_to_value( 2.0f );
	}
	else if( pattern == impulse )
	{
		test_fft.set_input_to_impulse();
		reference.set_data_to_impulse();
	}
	else if( pattern == erratic )
	{
		test_fft.set_input_to_random(); //TODO 
		reference.set_data_to_random(); //TODO
	}
	else
	{
		throw std::runtime_error( "invalid pattern type in real_to_complex()" );
	}


	// if we're starting with unequal data, we're destined for failure
	EXPECT_EQ( true, test_fft.input_buffer() == reference.input_buffer() );

	//test_fft.forward_scale( scale );//TODO may not need
	reference.forward_scale( scale );

	test_fft.transform();
	reference.transform();

	EXPECT_EQ( true, test_fft.result() == reference.result() );
}

