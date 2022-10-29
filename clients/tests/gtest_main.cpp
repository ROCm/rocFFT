// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <streambuf>
#include <string>
#include <thread>

#include "../../shared/concurrency.h"
#include "../../shared/environment.h"
#include "../../shared/work_queue.h"
#include "../rocfft_params.h"
#include "rocfft.h"
#include "rocfft_accuracy_test.h"
#include "test_params.h"

#ifdef WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Control output verbosity:
int verbose;

// Transform parameters for manual test:
fft_params manual_params;

// Ram limitation for tests (GiB):
size_t ramgb;

// Manually specified precision cutoffs:
double single_epsilon;
double double_epsilon;

// Measured precision cutoffs:
double max_linf_eps_double = 0.0;
double max_l2_eps_double   = 0.0;
double max_linf_eps_single = 0.0;
double max_l2_eps_single   = 0.0;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

// Cache the last cpu fft that was requested
last_cpu_fft_cache last_cpu_fft_data;

static size_t get_system_memory_GiB()
{
    // system memory often has a little chunk carved out for other
    // stuff, so round up to nearest GiB.
#ifdef WIN32
    MEMORYSTATUSEX info;
    info.dwLength = sizeof(info);
    if(!GlobalMemoryStatusEx(&info))
        return 0;
    return (info.ullTotalPhys + ONE_GiB - 1) / ONE_GiB;
#else
    struct sysinfo info;
    if(sysinfo(&info) != 0)
        return 0;
    return (info.totalram * info.mem_unit + ONE_GiB - 1) / ONE_GiB;
#endif
}

void precompile_test_kernels()
{
    puts("precompiling test kernels...");
    WorkQueue<std::string> tokenQueue;

    std::vector<std::string> tokens;
    auto                     ut = testing::UnitTest::GetInstance();
    for(int ts_index = 0; ts_index < ut->total_test_suite_count(); ++ts_index)
    {
        const auto ts = ut->GetTestSuite(ts_index);
        // skip disabled suites
        if(strncmp(ts->name(), "DISABLED", 8) == 0)
            continue;
        for(int ti_index = 0; ti_index < ts->total_test_count(); ++ti_index)
        {
            const auto  ti   = ts->GetTestInfo(ti_index);
            std::string name = ti->name();
            // only care about accuracy tests
            if(name.find("vs_fftw/") != std::string::npos
               && name.find("_batch_1_") != std::string::npos)
            {
                name.erase(0, 8);
                tokens.emplace_back(std::move(name));
            }
        }
    }

    std::random_device dev;
    std::mt19937       dist(dev());
    std::shuffle(tokens.begin(), tokens.end(), dist);
    auto precompile_begin = std::chrono::steady_clock::now();
    printf("precompiling %zu FFT plans...\n", tokens.size());

    for(auto&& t : tokens)
        tokenQueue.push(std::move(t));

    EnvironmentSetTemp       env_twid{"ROCFFT_INTERNAL_COMPILE_ONLY", "1"};
    const size_t             NUM_THREADS = rocfft_concurrency();
    std::vector<std::thread> threads;
    for(size_t i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&tokenQueue]() {
            for(;;)
            {
                std::string token{tokenQueue.pop()};
                if(token.empty())
                    break;
                rocfft_params params;
                params.from_token(token);
                params.validate();
                params.setup_structs();
            }
        });
        // insert empty tokens to tell threads to stop
        tokenQueue.push({});
    }
    for(auto& t : threads)
        t.join();

    auto                                      precompile_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> precompile_ms  = precompile_end - precompile_begin;
    printf("done precompiling FFT plans in %.2f ms\n", precompile_ms.count());
}

int main(int argc, char* argv[])
{
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly; no need to jump through hoops for
    // boost::program_options.
    ::testing::InitGoogleTest(&argc, argv);

    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;

    // Token string to fully specify fft params for the manual test.
    std::string test_token;

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
        ("transformType,t", po::value<fft_transform_type>(&manual_params.transform_type)
         ->default_value(fft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("callback", "Inject load/store callbacks")
        ("checkstride", "Check that data is not written outside of output strides")
        ("double", "Double precision transform (default: single)")
        ( "itype", po::value<fft_array_type>(&manual_params.itype)
          ->default_value(fft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<fft_array_type>(&manual_params.otype)
          ->default_value(fft_array_type_unset),
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
        ("R", po::value<size_t>(&ramgb)->default_value(get_system_memory_GiB()), "Ram limit in GiB for tests.")
        ("single_epsilon",  po::value<double>(&single_epsilon)->default_value(3.75e-5)) 
	("double_epsilon",  po::value<double>(&double_epsilon)->default_value(1e-15))
        ("wise,w", "use FFTW wisdom")
        ("wisdomfile,W",
         po::value<std::string>(&fftw_wisdom_filename)->default_value("wisdom3.txt"),
         "FFTW3 wisdom filename")
        ("scalefactor", po::value<double>(&manual_params.scale_factor), "Scale factor to apply to output.")
        ("token", po::value<std::string>(&test_token)->default_value(""), "Test token name for manual test")
        ("precompile", "Precompile kernels for all test cases before running tests");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }
    verbose = vm["verbose"].as<int>();

    std::cout << "single epsilon: " << single_epsilon << "\tdouble epsilon: " << double_epsilon
              << std::endl;

    if(vm.count("wise"))
    {
        use_fftw_wisdom = true;
    }

    rocfft_setup();
    char v[256];
    rocfft_get_version_string(v, 256);
    std::cout << "rocFFT version: " << v << std::endl;

#ifdef FFTW_MULTITHREAD
    fftw_init_threads();
    fftwf_init_threads();
    fftw_plan_with_nthreads(rocfft_concurrency());
    fftwf_plan_with_nthreads(rocfft_concurrency());
#endif

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

    if(test_token != "")
    {
        std::cout << "Reading fft params from token:\n" << test_token << std::endl;

        try
        {
            manual_params.from_token(test_token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return 1;
        }
    }
    else
    {
        if(manual_params.length.empty())
        {
            manual_params.length.push_back(8);
            // TODO: add random size?
        }

        manual_params.placement
            = vm.count("notInPlace") ? fft_placement_notinplace : fft_placement_inplace;
        manual_params.precision = vm.count("double") ? fft_precision_double : fft_precision_single;

        if(vm.count("callback"))
        {
            manual_params.run_callbacks = true;
        }

        if(vm.count("checkstride"))
        {
            manual_params.check_output_strides = true;
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
    }

    if(vm.count("precompile"))
        precompile_test_kernels();

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

    std::cout << "single precision max l-inf epsilon: " << max_linf_eps_single << std::endl;
    std::cout << "single precision max l2 epsilon:     " << max_l2_eps_single << std::endl;
    std::cout << "double precision max l-inf epsilon: " << max_linf_eps_double << std::endl;
    std::cout << "double precision max l2 epsilon:     " << max_l2_eps_double << std::endl;

    return retval;
}

TEST(manual, vs_fftw)
{
    // Run an individual test using the provided command-line parameters.

    std::cout << "Manual test:"
              << "\n\t" << manual_params.str("\n\t") << std::endl;

    manual_params.validate();

    std::cout << "Token: " << manual_params.token() << std::endl;

    rocfft_params params(manual_params);
    fft_vs_reference(params);
}
