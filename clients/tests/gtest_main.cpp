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

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int verbose = 0;

int main(int argc, char* argv[])
{
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly; no need to jump through hoops for
    // boost::program_options.
    ::testing::InitGoogleTest(&argc, argv);

    // Declare the supported options.
    po::options_description desc("rocFFT Runtime Test command line options");
    desc.add_options()("help,h", "produces this help message")(
        "verbose,v",
        po::value<int>()->default_value(0)->implicit_value(1),
        "print out detailed information for the tests.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    verbose = vm["verbose"].as<int>();

    char v[256];
    rocfft_get_version_string(v, 256);
    std::cout << "rocFFT version: " << v << std::endl;

    return RUN_ALL_TESTS();
}
