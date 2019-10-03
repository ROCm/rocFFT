// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// @file
/// @brief googletest based unit tester for rocfft
///

#include <gtest/gtest.h>
#include <iostream>

#include "rocfft.h"
#include "test_constants.h"

//#include <boost/program_options.hpp>
// namespace po = boost::program_options;

int main(int argc, char* argv[])
{

#if 0
    // Declare the supported options.
    po::options_description desc( "rocFFT Runtime Test command line options" );
    desc.add_options()
        ( "help,h",             "produces this help message" )
        ( "verbose,v",          "print out detailed information for the tests" )
        ( "noVersion",          "Don't print version information from the rocFFT library" )
        ( "pointwise,p",        "Do a pointwise comparison to determine test correctness (default: use root mean square)" )
        ( "tolerance,t",        po::value< float >( &tolerance )->default_value( 0.001f ),   "tolerance level to use when determining test pass/fail" )
        ( "numRandom,r",        po::value< size_t >( &number_of_random_tests )->default_value( 2000 ),   "number of random tests to run" )
        ;

    //    Parse the command line options, ignore unrecognized options and collect them into a vector of strings
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );
    po::store( parsed, vm );
    po::notify( vm );
    std::vector< std::string > to_pass_further = po::collect_unrecognized( parsed.options, po::include_positional );

    std::cout << std::endl;

    if( vm.count( "help" ) )
    {
        std::cout << desc << std::endl;
        return 0;
    }

    //    Create a new argc,argv to pass to InitGoogleTest
    //    First parameter of course is the name of this program
    std::vector< const char* > myArgv;

    //    Push back a pointer to the executable name
    if( argc > 0 )
        myArgv.push_back( *argv );

    int myArgc    = static_cast< int >( myArgv.size( ) );

#endif

    char v[256];
    rocfft_get_version_string(v, 256);
    std::cout << "rocFFT version: " << v << std::endl;

    //::testing::InitGoogleTest( &myArgc, const_cast< char** >( &myArgv[ 0 ] ) );

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
