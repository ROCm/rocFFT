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
std::vector<size_t>     istride;
std::vector<size_t>     ostride;

// Ram limitation for tests (GB).
size_t ramgb = 0;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

// cache the last cpu fft that was requested - the tuple members
// correspond to the input and output of compute_cpu_fft
std::tuple<std::vector<size_t>,
           size_t,
           rocfft_precision,
           rocfft_transform_type,
           accuracy_test::cpu_fft_data>
    last_cpu_fft;

accuracy_test::cpu_fft_data accuracy_test::compute_cpu_fft(const std::vector<size_t>& length,
                                                           size_t                     nbatch,
                                                           rocfft_precision           precision,
                                                           rocfft_transform_type      transformType)
{
    // check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  smaller batch runs can
    // compare against the larger data.
    if(std::get<0>(last_cpu_fft) == length && std::get<2>(last_cpu_fft) == precision
       && std::get<3>(last_cpu_fft) == transformType)
    {
        if(std::get<1>(last_cpu_fft) >= nbatch)
        {
            return std::get<4>(last_cpu_fft);
        }
        else
            // something's unexpected with our test order - we should have
            // generated the bigger batch first.  batch ranges provided to
            // the test suites need to be in descending order
            abort();
    }

    const size_t dim = length.size();

    // Input cpu parameters:
    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    auto istride = compute_stride(ilength);
    auto itype   = contiguous_itype(transformType);
    auto idist   = set_idist(rocfft_placement_notinplace, transformType, length, istride);

    // Output cpu parameters:
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    auto ostride = compute_stride(olength);
    auto odist   = set_odist(rocfft_placement_notinplace, transformType, length, ostride);
    auto otype   = contiguous_otype(transformType);

    if(verbose > 3)
    {
        std::cout << "CPU  params:\n";
        std::cout << "\tilength:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_istride:";
        for(auto i : istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_idist: " << idist << std::endl;

        std::cout << "\tolength:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_ostride:";
        for(auto i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_odist: " << odist << std::endl;
    }

    // hook up the futures
    std::shared_future<fftw_data_t> input = std::async(std::launch::async, [=]() {
        return compute_input<fftwAllocator<char>>(precision, itype, length, istride, idist, nbatch);
    });

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        printbuffer(precision, itype, input.get(), ilength, istride, nbatch, idist);
    }

    auto input_norm = std::async(std::launch::async, [=]() {
        auto ret_norm = norm(input.get(), ilength, nbatch, precision, itype, istride, idist);
        if(verbose > 2)
        {
            std::cout << "CPU Input Linf norm:  " << ret_norm.l_inf << "\n";
            std::cout << "CPU Input L2 norm:    " << ret_norm.l_2 << "\n";
        }
        return ret_norm;
    });

    std::shared_future<fftw_data_t> output      = std::async(std::launch::async, [=]() {
        // copy input, as FFTW may overwrite it
        auto input_copy = input.get();
        auto output     = fftw_via_rocfft(
            length, istride, ostride, nbatch, idist, odist, precision, transformType, input_copy);
        if(verbose > 3)
        {
            std::cout << "CPU output:\n";
            printbuffer(precision, otype, output, olength, ostride, nbatch, odist);
        }
        return std::move(output);
    });
    std::shared_future<VectorNorms> output_norm = std::async(std::launch::async, [=]() {
        auto ret_norm = norm(output.get(), olength, nbatch, precision, otype, ostride, odist);
        if(verbose > 2)
        {
            std::cout << "CPU Output Linf norm: " << ret_norm.l_inf << "\n";
            std::cout << "CPU Output L2 norm:   " << ret_norm.l_2 << "\n";
        }
        return ret_norm;
    });

    cpu_fft_data ret;
    ret.ilength = ilength;
    ret.istride = istride;
    ret.itype   = itype;
    ret.idist   = idist;
    ret.olength = olength;
    ret.ostride = ostride;
    ret.otype   = otype;
    ret.odist   = odist;

    ret.input       = std::move(input);
    ret.input_norm  = std::move(input_norm);
    ret.output      = std::move(output);
    ret.output_norm = std::move(output_norm);

    // cache our result
    std::get<0>(last_cpu_fft) = length;
    std::get<1>(last_cpu_fft) = nbatch;
    std::get<2>(last_cpu_fft) = precision;
    std::get<3>(last_cpu_fft) = transformType;
    std::get<4>(last_cpu_fft) = ret;

    return std::move(ret);
}

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
    po::options_description opdesc("rocFFT Runtime Test command line options\nNB: input parameters are row-major.");
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
        ("istride",  po::value<std::vector<size_t>>(&istride)->multitoken(), "Input stride.")
        ("ostride",  po::value<std::vector<size_t>>(&ostride)->multitoken(), "Output stride.")
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

    if(istride.size() == 0)
    {
        istride.push_back(1);
        // TODO: add random size?
    }

    if(ostride.size() == 0)
    {
        ostride.push_back(1);
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
    print_params(length, istride, istride, nbatch, place, precision, transformType, itype, otype);

    auto cpu = accuracy_test::compute_cpu_fft(length, nbatch, precision, transformType);

    rocfft_transform(length,
                     istride,
                     ostride,
                     nbatch,
                     precision,
                     transformType,
                     itype,
                     otype,
                     place,
                     cpu.istride,
                     cpu.ostride,
                     cpu.idist,
                     cpu.odist,
                     cpu.itype,
                     cpu.otype,
                     cpu.input,
                     cpu.output,
                     ramgb,
                     cpu.output_norm);

    auto cpu_input_norm = cpu.input_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_inf));

    auto cpu_output_norm = cpu.output_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));
}
