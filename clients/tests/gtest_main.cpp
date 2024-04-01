// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <random>
#include <streambuf>
#include <string>
#include <thread>

#include "../../shared/concurrency.h"
#include "../../shared/environment.h"
#include "../../shared/rocfft_accuracy_test.h"
#include "../../shared/test_params.h"
#include "../../shared/work_queue.h"
#include "rocfft/rocfft.h"

#ifdef WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Control output verbosity:
int verbose;

// User-defined random seed
size_t random_seed;
// Probability of running individual planar FFTs
double planar_prob;
// Probability of running individual callback FFTs
double callback_prob;
// Number of random tests per suite
size_t n_random_tests = 0;

// Transform parameters for manual test:
fft_params manual_params;

// Display 32/64-bit identifier for input
// and output buffers of a manual FFT test.
bool print_hash = false;

// Host memory limitation for tests (GiB):
size_t ramgb;

// Device memory limitation for tests (GiB):
size_t vramgb;

// Allow skipping tests if there is a runtime error
bool skip_runtime_fails;
// But count the number of failures
int n_hip_failures = 0;

// Manually specified precision cutoffs:
double half_epsilon;
double single_epsilon;
double double_epsilon;

// Measured precision cutoffs:
double max_linf_eps_double = 0.0;
double max_l2_eps_double   = 0.0;
double max_linf_eps_single = 0.0;
double max_l2_eps_single   = 0.0;
double max_linf_eps_half   = 0.0;
double max_l2_eps_half     = 0.0;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

// Compare results against FFTW in accuracy tests
bool fftw_compare = true;

// Cache the last cpu fft that was requested
last_cpu_fft_cache last_cpu_fft_data;

// Number of devices to distribute the FFT to for manual tests
int manual_devices = 1;

system_memory get_system_memory()
{
    system_memory memory_data;
#ifdef WIN32
    MEMORYSTATUSEX info;
    info.dwLength = sizeof(info);
    if(!GlobalMemoryStatusEx(&info))
        return memory_data;
    memory_data.total_bytes = info.ullTotalPhys;
    memory_data.free_bytes  = info.ullAvailPhys;
#else
    struct sysinfo info;
    if(sysinfo(&info) != 0)
        return memory_data;
    memory_data.total_bytes = info.totalram * info.mem_unit;
    memory_data.free_bytes  = info.freeram * info.mem_unit;
#endif
    return memory_data;
}

system_memory start_memory = get_system_memory();

void precompile_test_kernels(const std::string& precompile_file)
{
    std::cout << "precompiling test kernels...\n";
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
            if(name.find("vs_fftw/") != std::string::npos)
            {
                name.erase(0, 8);

                // Run any problem that uses brick decomposition
                // without touching batch.  Bricks are specified with
                // batch indexes, so arbitrarily changing batch to 1
                // can break those cases.
                if(name.find("_brick_") != std::string::npos)
                {
                    tokens.emplace_back(std::move(name));
                    continue;
                }

                // change batch to 1, so we don't waste time creating
                // multiple plans that differ only by batch
                auto idx = name.find("_batch_");
                if(idx == std::string::npos)
                    continue;
                // advance idx to batch number
                idx += 7;
                auto end = name.find('_', idx);
                if(end == std::string::npos)
                    continue;
                name.replace(idx, end - idx, "1");

                tokens.emplace_back(std::move(name));
            }
        }
    }

    std::random_device dev;
    std::mt19937       dist(dev());
    std::shuffle(tokens.begin(), tokens.end(), dist);
    auto precompile_begin = std::chrono::steady_clock::now();
    std::cout << "precompiling " << tokens.size() << " FFT plans...\n";

    for(auto&& t : tokens)
        tokenQueue.push(std::move(t));

    EnvironmentSetTemp       env_compile_only{"ROCFFT_INTERNAL_COMPILE_ONLY", "1"};
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

                try
                {
                    rocfft_params params_forward;
                    params_forward.from_token(token);
                    params_forward.validate();
                    params_forward.setup_structs();

                    params_forward.free();

                    rocfft_params params_inverse;
                    params_inverse.inverse_from_forward(params_forward);
                    params_inverse.validate();
                    params_inverse.setup_structs();
                }
                catch(std::exception& e)
                {
                    // failed to create a plan, abort
                    //
                    // we could continue on, but the test should just
                    // fail later anyway in the same way.  so report
                    // which token failed early and get out
                    throw std::runtime_error(token + " plan creation failure: " + e.what());
                }
            }
        });
        // insert empty tokens to tell threads to stop
        tokenQueue.push({});
    }
    for(auto& t : threads)
        t.join();

    auto                                      precompile_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> precompile_ms  = precompile_end - precompile_begin;
    std::cout << "done precompiling FFT plans in " << static_cast<size_t>(precompile_ms.count())
              << " ms\n";
}

int main(int argc, char* argv[])
{
    // We would like to parse a few arguments before initiating gtest.
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
    // clang-format off
    opdesc.add_options()
        ("verbose,v",
         po::value<int>()->default_value(0),
         "print out detailed information for the tests.")
        ("seed", po::value<size_t>(&random_seed),
         "Random seed; if unset, use an actual random seed.")
        ("nrand", po::value<size_t>(&n_random_tests)->default_value(0),
         "Number of extra randomized tests.")
        ("planar_prob", po::value<double>(&planar_prob)->default_value(0.1),
        "Probability of running individual planar transforms")
        ("callback", "Inject load/store callbacks")
        ("callback_prob", po::value<double>(&callback_prob)->default_value(0.1),
         "Probability of running individual callback transforms")
        ("fftw_compare",
	 po::value<bool>(&fftw_compare)->default_value(true), "Compare to FFTW in accuracy tests");
    // clang-format on
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opdesc).allow_unregistered().run(), vm);
    po::notify(vm);

    verbose = vm["verbose"].as<int>();

    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly; no need to jump through hoops for
    // boost::program_options.
    ::testing::InitGoogleTest(&argc, argv);

    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;

    // Token string to fully specify fft params for the manual test.
    std::string test_token;

    // Filename for precompiled kernels to be written to
    std::string precompile_file;

    // Declare the supported options.
    // clang-format doesn't handle boost program options very well:
    // clang-format off
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("skip_runtime_fails",  po::value<bool>(&skip_runtime_fails)->default_value(true),
        "Skip the test if there is a runtime failure.")
        ("version", "Print queryable version information from the rocfft library and exit")
        ("transformType,t", po::value<fft_transform_type>(&manual_params.transform_type)
         ->default_value(fft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("checkstride", "Check that data is not written outside of output strides")
        ("double", "Double precision transform (deprecated: use --precision double)")
        ("precision", po::value<fft_precision>(&manual_params.precision),
         "Transform precision: single (default), double, half")
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
        ("R", po::value<size_t>(&ramgb)->default_value((start_memory.total_bytes + ONE_GiB - 1) / ONE_GiB), "Ram limit in GiB for tests.")
        ("V", po::value<size_t>(&vramgb)->default_value(0), "vram limit in GiB for tests.")
        ("half_epsilon",  po::value<double>(&half_epsilon)->default_value(9.77e-4))
        ("single_epsilon",  po::value<double>(&single_epsilon)->default_value(3.75e-5))
        ("double_epsilon",  po::value<double>(&double_epsilon)->default_value(1e-15))
        ("wise,w", "use FFTW wisdom")
        ("wisdomfile,W",
         po::value<std::string>(&fftw_wisdom_filename)->default_value("wisdom3.txt"),
         "FFTW3 wisdom filename")
        ("manual_devices", po::value<int>(&manual_devices)->default_value(1),
	  "Distribute manual test case among this many devices")
        ("scalefactor", po::value<double>(&manual_params.scale_factor), "Scale factor to apply to output.")
        ("token", po::value<std::string>(&test_token)->default_value(""), "Test token name for manual test")
        ("hash", po::value<bool>(&print_hash)->default_value(false), "Display input/output buffer identifier for manual test")
        ("precompile",  po::value<std::string>(&precompile_file), "Precompile kernels to a file for all test cases before running tests");
    // clang-format on

    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << "\n";
        return 0;
    }

    if(vm.count("version"))
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "version " << v << "\n";
        return EXIT_SUCCESS;
    }

    std::cout << "half epsilon: " << half_epsilon << "\tsingle epsilon: " << single_epsilon
              << "\tdouble epsilon: " << double_epsilon << "\n";

    if(!vm.count("seed"))
    {
        std::random_device dev;
        random_seed = dev();
    }
    std::cout << "Random seed: " << random_seed << "\n";

    if(vm.count("wise"))
    {
        use_fftw_wisdom = true;
    }

    // if precompiling, tell rocFFT to use the specified cache file
    // to write kernels to
    //
    // but if our environment already has a cache file for RTC, then
    // we should just use that
    std::unique_ptr<EnvironmentSetTemp> env_precompile;
    if(!precompile_file.empty() && rocfft_getenv("ROCFFT_RTC_CACHE_PATH").empty())
    {
        env_precompile = std::make_unique<EnvironmentSetTemp>("ROCFFT_RTC_CACHE_PATH",
                                                              precompile_file.c_str());
    }

    rocfft_setup();
    char v[256];
    rocfft_get_version_string(v, 256);
    std::cout << "rocFFT version: " << v << "\n";

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
        std::cout << "Reading fft params from token:\n" << test_token << "\n";

        try
        {
            manual_params.from_token(test_token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token.\n";
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
        if(vm.count("double"))
            manual_params.precision = fft_precision_double;

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
        precompile_test_kernels(precompile_file);

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

    std::cout << "Random seed: " << random_seed << "\n";
    std::cout << "half precision max l-inf epsilon: " << max_linf_eps_half << "\n";
    std::cout << "half precision max l2 epsilon:     " << max_l2_eps_half << "\n";
    std::cout << "single precision max l-inf epsilon: " << max_linf_eps_single << "\n";
    std::cout << "single precision max l2 epsilon:     " << max_l2_eps_single << "\n";
    std::cout << "double precision max l-inf epsilon: " << max_linf_eps_double << "\n";
    std::cout << "double precision max l2 epsilon:     " << max_l2_eps_double << "\n";
    std::cout << "Number of runtime issues: " << n_hip_failures << "\n";

    return retval;
}

TEST(manual, vs_fftw) // MANUAL TESTS HERE
{
    rocfft_params params(manual_params);

    if(manual_devices > 1)
    {
        // just distribute along the slowest FFT dimension
        std::vector<unsigned int> deviceGrid(params.length.size() + 1, 1);
        deviceGrid[1] = manual_devices;

        params.distribute_input(deviceGrid);
        params.distribute_output(deviceGrid);
    }

    // Run an individual test using the provided command-line parameters.
    params.validate();

    std::cout << "Manual test:"
              << "\n\t" << params.str("\n\t") << "\n";

    std::cout << "Token: " << params.token() << "\n";

    if(!params.valid(verbose + 2))
    {
        std::cout << "manual params are not valid\n";
    }

    fft_vs_reference(params, print_hash);
}
