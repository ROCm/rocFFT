/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

/// @file
/// @brief googletest based unit tester for rocfft
///

#include <iostream>
#include <gtest/gtest.h>
// #include <boost/program_options.hpp>

#include <hip_runtime.h>
#include "rocfft.h"
#include "test_constants.h"
// namespace po = boost::program_options;

size_t number_of_random_tests;
time_t random_test_parameter_seed;
float tolerance;
double rmse_tolerance;
bool verbose;

// global for test use
bool suppress_output = false;
bool comparison_type = root_mean_square;

int main( int argc, char* argv[] )
{
	// // Declare the supported options.
	// po::options_description desc( "rocfft-test command line options" );
	// desc.add_options()
	// 	( "help,h",		"print help message" )
	// 	;
	//
	// //	Parse the command line options, ignore unrecognized options and collect them into a vector of strings
	// po::variables_map vm;
	// po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );
	// po::store( parsed, vm );
	// po::notify( vm );
	// std::vector< std::string > to_pass_further = po::collect_unrecognized( parsed.options, po::include_positional );
	//
	// std::cout << std::endl;
	//
	// if( vm.count( "help" ) )
	// {
	// 	std::cout << desc << std::endl;
	// 	return 0;
	// }

	::testing::InitGoogleTest( &argc, argv );

	return RUN_ALL_TESTS();

}
