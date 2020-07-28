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

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <streambuf>
#include <string>

#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"
#include "test_constants.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Control output verbosity:
int verbose = 0;

// Parameters for single manual test:

// Transform type parameters:
rocfft_transform_type   transformType;
rocfft_array_type       itype;
rocfft_array_type       otype;
size_t                  nbatch = 1;
rocfft_result_placement place;
rocfft_precision        precision;
std::vector<size_t>     length;
size_t                  istride0;
size_t                  ostride0;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

int main(int argc, char* argv[])
{
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly; no need to jump through hoops for
    // boost::program_options.
    ::testing::InitGoogleTest(&argc, argv);

    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;

    // Declare the supported options.
    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("rocFFT Runtime Test command line options");
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("verbose,v",  po::value<int>()->default_value(0),
        "print out detailed information for the tests.")
        ("transformType,t", po::value<rocfft_transform_type>(&transformType)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ( "itype", po::value<rocfft_array_type>(&itype)
          ->default_value(rocfft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&otype)
          ->default_value(rocfft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<size_t>>(&length)->multitoken(), "Lengths.")
        ( "batchSize,b", po::value<size_t>(&nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "istride0", po::value<size_t>(&istride0)->default_value(1),
          "Input stride ")
        ( "ostride0", po::value<size_t>(&ostride0)->default_value(1),
          "Output stride ")
        ("wise,w", "use FFTW wisdom")
        ("wisdomfile,W",
         po::value<std::string>(&fftw_wisdom_filename)->default_value("wisdom3.txt"),
         "FFTW3 wisdom filename");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }
    place     = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    precision = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

    verbose = vm["verbose"].as<int>();

    if(vm.count("wise"))
    {
        use_fftw_wisdom = true;
    }

    if(length.size() == 0)
    {
        length.push_back(8);
        // TODO: add random size?
    }

    rocfft_setup();
    char v[256];
    rocfft_get_version_string(v, 256);
    std::cout << "rocFFT version: " << v << std::endl;

    if(use_fftw_wisdom)
    {
        if(verbose)
        {
            std::cout << "Using " << fftw_wisdom_filename << " wisdom file\n";
        }
        std::ifstream fftw_wisdom_file(fftw_wisdom_filename);
        std::string   allwisdom = std::string(std::istreambuf_iterator<char>(fftw_wisdom_file),
                                            std::istreambuf_iterator<char>());

        std::string fftw_wisdom;
        std::string fftwf_wisdom;

        bool               load_wisdom  = false;
        bool               load_fwisdom = false;
        std::istringstream input;
        input.str(allwisdom);
        // Separate the single-precision and double-precision wisdom:
        for(std::string line; std::getline(input, line);)
        {
            if(line.rfind("(fftw", 0) == 0 && line.find("fftw_wisdom") != std::string::npos)
            {
                load_wisdom = true;
            }
            if(line.rfind("(fftw", 0) == 0 && line.find("fftwf_wisdom") != std::string::npos)
            {
                load_fwisdom = true;
            }
            if(load_wisdom)
            {
                fftw_wisdom.append(line + "\n");
            }
            if(load_fwisdom)
            {
                fftwf_wisdom.append(line + "\n");
            }
            if(line.rfind(")", 0) == 0)
            {
                load_wisdom  = false;
                load_fwisdom = false;
            }
        }
        fftw_import_wisdom_from_string(fftw_wisdom.c_str());
        fftwf_import_wisdom_from_string(fftwf_wisdom.c_str());
    }

    auto retval = RUN_ALL_TESTS();

    if(use_fftw_wisdom)
    {
        std::string fftw_wisdom  = std::string(fftw_export_wisdom_to_string());
        std::string fftwf_wisdom = std::string(fftwf_export_wisdom_to_string());
        fftw_wisdom.append(std::string(fftwf_export_wisdom_to_string()));
        std::ofstream fftw_wisdom_file(fftw_wisdom_filename);
        fftw_wisdom_file << fftw_wisdom;
        fftw_wisdom_file << fftwf_wisdom;
        fftw_wisdom_file.close();
    }

    rocfft_cleanup();
    return retval;
}

TEST(manual, vs_fftw)
{
    // Run an individual test using the provided command-line parameters.

    std::cout << "Manual test:" << std::endl;
    check_set_iotypes(place, transformType, itype, otype);
    print_params(length, istride0, istride0, nbatch, place, precision, transformType, itype, otype);

    const size_t dim = length.size();

    // Input cpu parameters:
    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    const auto cpu_istride = compute_stride(ilength, 1);
    const auto cpu_itype   = contiguous_itype(transformType);
    const auto cpu_idist
        = set_idist(rocfft_placement_notinplace, transformType, length, cpu_istride);

    // Output cpu parameters:
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    const auto cpu_ostride = compute_stride(olength, 1);
    const auto cpu_odist
        = set_odist(rocfft_placement_notinplace, transformType, length, cpu_ostride);
    auto cpu_otype = contiguous_otype(transformType);
    // Generate the data:
    auto cpu_input = compute_input<fftwAllocator<char>>(
        precision, cpu_itype, length, cpu_istride, cpu_idist, nbatch);
    auto cpu_input_copy = cpu_input; // copy of input (might get overwritten by FFTW).

    // Compute the Linfinity and L2 norm of the CPU output:
    auto cpu_input_L2Linfnorm
        = LinfL2norm(cpu_input, ilength, nbatch, precision, cpu_itype, cpu_istride, cpu_idist);
    if(verbose > 2)
    {
        std::cout << "CPU Input Linf norm:  " << cpu_input_L2Linfnorm.first << "\n";
        std::cout << "CPU Input L2 norm:    " << cpu_input_L2Linfnorm.second << "\n";
    }
    ASSERT_TRUE(std::isfinite(cpu_input_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(cpu_input_L2Linfnorm.second));

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        printbuffer(precision, cpu_itype, cpu_input, ilength, cpu_istride, nbatch, cpu_idist);
    }

    // FFTW computation
    // NB: FFTW may overwrite input, even for out-of-place transforms.
    auto cpu_output = fftw_via_rocfft(length,
                                      cpu_istride,
                                      cpu_ostride,
                                      nbatch,
                                      cpu_idist,
                                      cpu_odist,
                                      precision,
                                      transformType,
                                      cpu_input);

    // Compute the Linfinity and L2 norm of the CPU output:
    auto cpu_output_L2Linfnorm
        = LinfL2norm(cpu_output, olength, nbatch, precision, cpu_otype, cpu_ostride, cpu_odist);
    if(verbose > 2)
    {
        std::cout << "CPU Output Linf norm: " << cpu_output_L2Linfnorm.first << "\n";
        std::cout << "CPU Output L2 norm:   " << cpu_output_L2Linfnorm.second << "\n";
    }
    if(verbose > 3)
    {
        std::cout << "CPU output:\n";
        printbuffer(precision, cpu_otype, cpu_output, olength, cpu_ostride, nbatch, cpu_odist);
    }
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.second));

    rocfft_transform(length,
                     istride0,
                     ostride0,
                     nbatch,
                     precision,
                     transformType,
                     itype,
                     otype,
                     place,
                     cpu_istride,
                     cpu_ostride,
                     cpu_idist,
                     cpu_odist,
                     cpu_itype,
                     cpu_otype,
                     cpu_input_copy,
                     cpu_output,
                     cpu_output_L2Linfnorm);
}
