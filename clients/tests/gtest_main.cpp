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
#include "test_params.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Control output verbosity:
int verbose;

// Transform parameters for manual test:
rocfft_params manual_params;

// Ram limitation for tests (GB).
size_t ramgb;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

// Cache the last cpu fft that was requested - the tuple members
// correspond to the input and output of compute_cpu_fft.
std::tuple<std::vector<size_t>, size_t, rocfft_transform_type, accuracy_test::cpu_fft_params>
    last_cpu_fft;

int main(int argc, char* argv[])
{
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly; no need to jump through hoops for
    // boost::program_options.
    ::testing::InitGoogleTest(&argc, argv);

    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;

    po::options_description opdesc(
        "\n"
        "rocFFT Runtime Test command line options\n"
        "NB: input parameters are row-major.\n"
        "\n"
        "FFTW accuracy test cases are named using these identifiers:\n"
        "\n"
        "  len_<n>: problem dimensions, row-major\n"
        "  single,double: precision\n"
        "  ip,op: in-place or out-of-place\n"
        "  batch_<n>: batch size\n"
        "  istride_<n>_<format>: input stride (ostride for output stride), format may be:\n"
        "      CI - complex interleaved\n"
        "      CP - complex planar\n"
        "      R  - real\n"
        "      HI - hermitian interleaved\n"
        "      HP - hermitian planar\n"
        "\n"
        "Usage");
    // Declare the supported options.
    // clang-format doesn't handle boost program options very well:
    // clang-format off
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("verbose,v",  po::value<int>()->default_value(0),
        "print out detailed information for the tests.")
        ("transformType,t", po::value<rocfft_transform_type>(&manual_params.transform_type)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ( "itype", po::value<rocfft_array_type>(&manual_params.itype)
          ->default_value(rocfft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&manual_params.otype)
          ->default_value(rocfft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<size_t>>(&manual_params.length)->multitoken(), "Lengths.")
        ( "batchSize,b", po::value<size_t>(&manual_params.nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ("istride",  po::value<std::vector<size_t>>(&manual_params.istride)->multitoken(),
         "Input stride.")
        ("ostride",  po::value<std::vector<size_t>>(&manual_params.ostride)->multitoken(),
         "Output stride.")
        ("idist", po::value<size_t>(&manual_params.idist)->default_value(0),
         "Logical distance between input batches.")
        ("odist", po::value<size_t>(&manual_params.odist)->default_value(0),
         "Logical distance between output batches.")
        ("ioffset", po::value<std::vector<size_t>>(&manual_params.ioffset)->multitoken(),
         "Input offset.")
        ("ooffset", po::value<std::vector<size_t>>(&manual_params.ooffset)->multitoken(),
         "Output offset.")
        ("isize", po::value<std::vector<size_t>>(&manual_params.isize)->multitoken(),
         "Logical size of input buffer.")
        ("osize", po::value<std::vector<size_t>>(&manual_params.osize)->multitoken(),
         "Logical size of output.")
        ("R", po::value<size_t>(&ramgb)->default_value(0), "Ram limit in GB for tests.")
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
    manual_params.placement
        = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    manual_params.precision
        = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

    verbose = vm["verbose"].as<int>();

    if(vm.count("wise"))
    {
        use_fftw_wisdom = true;
    }

    if(manual_params.length.empty())
    {
        manual_params.length.push_back(8);
        // TODO: add random size?
    }

    if(manual_params.istride.empty())
    {
        manual_params.istride.push_back(1);
        // TODO: add random size?
    }

    if(manual_params.ostride.empty())
    {
        manual_params.ostride.push_back(1);
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

    manual_params.istride
        = compute_stride(manual_params.ilength(),
                         manual_params.istride,
                         manual_params.placement == rocfft_placement_inplace
                             && manual_params.transform_type == rocfft_transform_type_real_forward);

    manual_params.ostride
        = compute_stride(manual_params.olength(),
                         manual_params.ostride,
                         manual_params.placement == rocfft_placement_inplace
                             && manual_params.transform_type == rocfft_transform_type_real_inverse);

    if(manual_params.idist == 0)
    {
        manual_params.idist = set_idist(manual_params.placement,
                                        manual_params.transform_type,
                                        manual_params.length,
                                        manual_params.istride);
    }
    if(manual_params.odist == 0)
    {
        manual_params.odist = set_odist(manual_params.placement,
                                        manual_params.transform_type,
                                        manual_params.length,
                                        manual_params.ostride);
    }

    check_set_iotypes(manual_params.placement,
                      manual_params.transform_type,
                      manual_params.itype,
                      manual_params.otype);

    if(manual_params.isize.empty())
    {
        for(int i = 0; i < manual_params.nibuffer(); ++i)
        {
            manual_params.isize.push_back(manual_params.compute_isize());
        }
    }

    if(manual_params.osize.empty())
    {
        for(int i = 0; i < manual_params.nobuffer(); ++i)
        {
            manual_params.osize.push_back(manual_params.compute_osize());
        }
    }

    std::cout << "\t" << manual_params.str("\n\t") << std::endl;
    auto cpu = accuracy_test::compute_cpu_fft(manual_params);

    accuracy_test::cpu_fft_params cpu_params(manual_params);

    rocfft_transform(manual_params, cpu, ramgb);

    auto cpu_input_norm = cpu.input_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_inf));

    auto cpu_output_norm = cpu.output_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));
}
