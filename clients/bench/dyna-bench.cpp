// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

// This file allows one to run tests multiple different rocFFT libraries at the same time.
// This allows one to randomize the execution order for better a better experimental setup
// which produces fewer type 1 errors where one incorrectly rejects the null hypothesis.

#include <algorithm>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <math.h>
#include <vector>

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#include "../../shared/CLI11.hpp"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "bench.h"
#include "rocfft/rocfft.h"

#ifdef WIN32
typedef HMODULE ROCFFT_LIB;
#else
typedef void* ROCFFT_LIB;
#endif

// Load the rocfft library
ROCFFT_LIB rocfft_lib_load(const std::string& path)
{
#ifdef WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

// Return a string describing the error loading rocfft
const char* rocfft_lib_load_error()
{
#ifdef WIN32
    // just return the error number
    static std::string error_str;
    error_str = std::to_string(GetLastError());
    return error_str.c_str();
#else
    return dlerror();
#endif
}

// Get symbol from rocfft lib
void* rocfft_lib_symbol(ROCFFT_LIB libhandle, const char* sym)
{
#ifdef WIN32
    return reinterpret_cast<void*>(GetProcAddress(libhandle, sym));
#else
    return dlsym(libhandle, sym);
#endif
}

void rocfft_lib_close(ROCFFT_LIB libhandle)
{
#ifdef WIN32
    FreeLibrary(libhandle);
#else
    dlclose(libhandle);
#endif
}

// Given a libhandle from dload, return a plan to a rocFFT plan with the given parameters.
rocfft_plan make_plan(ROCFFT_LIB libhandle, const fft_params& params)
{
    auto procfft_setup = (decltype(&rocfft_setup))rocfft_lib_symbol(libhandle, "rocfft_setup");
    if(procfft_setup == NULL)
        exit(1);
    auto procfft_plan_description_create
        = (decltype(&rocfft_plan_description_create))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_create");
    auto procfft_plan_description_destroy
        = (decltype(&rocfft_plan_description_destroy))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_destroy");
    auto procfft_plan_description_set_data_layout
        = (decltype(&rocfft_plan_description_set_data_layout))rocfft_lib_symbol(
            libhandle, "rocfft_plan_description_set_data_layout");
    auto procfft_plan_create
        = (decltype(&rocfft_plan_create))rocfft_lib_symbol(libhandle, "rocfft_plan_create");

    procfft_setup();

    rocfft_plan_description desc = NULL;
    LIB_V_THROW(procfft_plan_description_create(&desc), "rocfft_plan_description_create failed");
    LIB_V_THROW(
        procfft_plan_description_set_data_layout(desc,
                                                 rocfft_array_type_from_fftparams(params.itype),
                                                 rocfft_array_type_from_fftparams(params.otype),
                                                 params.ioffset.data(),
                                                 params.ooffset.data(),
                                                 params.istride.size(),
                                                 params.istride.data(),
                                                 params.idist,
                                                 params.ostride.size(),
                                                 params.ostride.data(),
                                                 params.odist),
        "rocfft_plan_description_data_layout failed");
    rocfft_plan plan = NULL;

    LIB_V_THROW(procfft_plan_create(&plan,
                                    rocfft_result_placement_from_fftparams(params.placement),
                                    rocfft_transform_type_from_fftparams(params.transform_type),
                                    rocfft_precision_from_fftparams(params.precision),
                                    params.length.size(),
                                    params.length.data(),
                                    params.nbatch,
                                    desc),
                "rocfft_plan_create failed");

    LIB_V_THROW(procfft_plan_description_destroy(desc), "rocfft_plan_description_destroy failed");

    return plan;
}

// Given a libhandle from dload and a rocFFT plan, destroy the plan.
void destroy_plan(ROCFFT_LIB libhandle, rocfft_plan& plan)
{
    auto procfft_plan_destroy
        = (decltype(&rocfft_plan_destroy))rocfft_lib_symbol(libhandle, "rocfft_plan_destroy");

    LIB_V_THROW(procfft_plan_destroy(plan), "rocfft_plan_destroy failed");

    auto procfft_cleanup
        = (decltype(&rocfft_cleanup))rocfft_lib_symbol(libhandle, "rocfft_cleanup");
    if(procfft_cleanup)
        LIB_V_THROW(procfft_cleanup(), "rocfft_cleanup failed");
}

// Given a libhandle from dload and a rocFFT execution info structure, destroy the info.
void destroy_info(ROCFFT_LIB libhandle, rocfft_execution_info& info)
{
    auto procfft_execution_info_destroy
        = (decltype(&rocfft_execution_info_destroy))rocfft_lib_symbol(
            libhandle, "rocfft_execution_info_destroy");
    LIB_V_THROW(procfft_execution_info_destroy(info), "rocfft_execution_info_destroy failed");
}

// Given a libhandle from dload, and a corresponding rocFFT plan, return how much work
// buffer is required.
size_t get_wbuffersize(ROCFFT_LIB libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_work_buffer_size
        = (decltype(&rocfft_plan_get_work_buffer_size))rocfft_lib_symbol(
            libhandle, "rocfft_plan_get_work_buffer_size");

    // Get the buffersize
    size_t workBufferSize = 0;
    LIB_V_THROW(procfft_plan_get_work_buffer_size(plan, &workBufferSize),
                "rocfft_plan_get_work_buffer_size failed");

    return workBufferSize;
}

// Given a libhandle from dload and a corresponding rocFFT plan, print the plan information.
void show_plan(ROCFFT_LIB libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_print
        = (decltype(&rocfft_plan_get_print))rocfft_lib_symbol(libhandle, "rocfft_plan_get_print");

    LIB_V_THROW(procfft_plan_get_print(plan), "rocfft_plan_get_print failed");
}

// FIXME: doc
rocfft_execution_info make_execinfo(ROCFFT_LIB libhandle)
{
    auto procfft_execution_info_create = (decltype(&rocfft_execution_info_create))rocfft_lib_symbol(
        libhandle, "rocfft_execution_info_create");
    rocfft_execution_info info = NULL;
    LIB_V_THROW(procfft_execution_info_create(&info), "rocfft_execution_info_create failed");
    return info;
}

// FIXME: doc
void set_work_buffer(const ROCFFT_LIB&      libhandle,
                     rocfft_execution_info& info,
                     const size_t           wbuffersize,
                     void*                  wbuffer)
{
    if(wbuffersize > 0 && wbuffer != NULL)
    {
        auto procfft_execution_info_set_work_buffer
            = (decltype(&rocfft_execution_info_set_work_buffer))rocfft_lib_symbol(
                libhandle, "rocfft_execution_info_set_work_buffer");
        LIB_V_THROW(procfft_execution_info_set_work_buffer(info, wbuffer, wbuffersize),
                    "rocfft_execution_info_set_work_buffer failed");
    }
}

// Given a libhandle from dload and a corresponding rocFFT plan and execution info,
// execute a transform on the given input and output buffers and return the kernel
// execution time.
float run_plan(
    ROCFFT_LIB libhandle, rocfft_plan plan, rocfft_execution_info info, void** in, void** out)
{
    auto procfft_execute
        = (decltype(&rocfft_execute))rocfft_lib_symbol(libhandle, "rocfft_execute");

    hipEvent_wrapper_t start, stop;
    start.alloc();
    stop.alloc();

    HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

    auto rcfft = procfft_execute(plan, in, out, info);

    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

    if(rcfft != rocfft_status_success)
    {
        throw std::runtime_error("execution failed");
    }

    float time;
    HIP_V_THROW(hipEventElapsedTime(&time, start, stop), "hipEventElapsedTime failed");
    return time;
}

std::pair<ROCFFT_LIB, rocfft_plan> create_handleplan(const std::string& libstring,
                                                     const fft_params&  params)
{
    auto libhandle = rocfft_lib_load(libstring);
    if(libhandle == NULL)
    {
        std::stringstream ss;
        ss << "Failed to open " << libstring << ", error: " << rocfft_lib_load_error();
        throw std::runtime_error(ss.str());
    }

    auto plan = make_plan(libhandle, params);

    return std::make_pair(libhandle, plan);
}

int main(int argc, char* argv[])
{
    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples:
    int ntrial{};

    // Bool to specify whether the libs are loaded in forward or forward+reverse order.
    int reverse{};

    // Test sequence choice:
    int test_sequence{};

    // Vector of test target libraries
    std::vector<std::string> lib_strings;

    // FFT parameters:
    fft_params params;

    // Token string to fully specify fft params.
    std::string token;

    CLI::App app{"dyna-rocfft-bench command line options"};

    // Declare supported pure flags.
    // FIXME: version needs to be implemented
    app.add_flag("--version",
                 "Print queryable version information from the rocfft library and exit");
    CLI::Option* opt_double = app.add_flag(
        "--double", "Double precision transform (deprecated: use --precision double)");
    CLI::Option* opt_not_in_place
        = app.add_flag("-o, --notInPlace", "Not in-place FFT transform (default: in-place)");

    // Declare the supported options. Some option pointers are declared to track passed opts.
    app.add_option("--device", deviceId, "Select a specific device id")->default_val(0);
    app.add_option("--verbose", verbose, "Control output verbosity")->default_val(0);
    CLI::Option* opt_ntrial
        = app.add_option("-N, --ntrial", ntrial, "Trial size for the problem")->default_val(1);
    app.add_option("--reverse", reverse, "Load libs in forward and reverse order")->default_val(1);
    app.add_option(
           "--sequence", test_sequence, "Test sequence:\n0) random\n1) alternating\n2) sequential")
        ->default_val(0);
    app.add_option(
        "--precision", params.precision, "Transform precision: single (default), double, half");
    app.add_option("-g, --inputGen",
                   params.igen,
                   "Input data generation:\n0) PRNG sequence (device)\n"
                   "1) PRNG sequence (host)\n"
                   "2) linearly-spaced sequence (device)\n"
                   "3) linearly-spaced sequence (host)")
        ->default_val(fft_input_random_generator_device);
    app.add_option("-t, --transformType",
                   params.transform_type,
                   "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                   "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward);
    app.add_option("-b, --batchSize",
                   params.nbatch,
                   "If this value is greater than one, arrays will be used")
        ->default_val(1);
    app.add_option("--itype",
                   params.itype,
                   "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
                   "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    app.add_option("--otype",
                   params.otype,
                   "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
                   "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    app.add_option("--lib", lib_strings, "Set test target library full path (appendable)");
    CLI::Option* opt_length  = app.add_option("--length", params.length, "Lengths");
    CLI::Option* opt_istride = app.add_option("--istride", params.istride, "Input strides");
    CLI::Option* opt_ostride = app.add_option("--ostride", params.ostride, "Output strides");
    app.add_option("--idist", params.idist, "Logical distance between input batches")
        ->default_val(0);
    app.add_option("--odist", params.odist, "Logical distance between output batches")
        ->default_val(0);
    app.add_option("--isize", params.isize, "Logical size of input buffer");
    app.add_option("--osize", params.osize, "Logical size of output buffer");
    CLI::Option* opt_ioffset = app.add_option("--ioffset", params.ioffset, "Input offsets");
    CLI::Option* opt_ooffset = app.add_option("--ooffset", params.ooffset, "Output offsets");
    app.add_option("--scalefactor", params.scale_factor, "Scale factor to apply to output");
    app.add_option("--token", token);

    // Parse args and catch any errors here
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Check if all the provided libraries are actually there:
    for(const auto& lib_string : lib_strings)
    {
        if(!std::filesystem::exists(lib_string))
        {
            std::cerr << "Error: lib " << lib_string << " does not exist\n";
            exit(1);
        }
    }

    if(*opt_not_in_place)
    {
        std::cout << "out-of-place\n";
    }
    else
    {
        std::cout << "in-place\n";
    }

    if(*opt_ntrial)
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(!token.empty())
    {
        std::cout << "Reading fft params from token:\n" << token << std::endl;

        try
        {
            params.from_token(token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return 1;
        }
    }
    else
    {
        if(!*opt_length)
        {
            std::cout << "Please specify transform length!" << std::endl;
            std::cout << app.help() << std::endl;
            return EXIT_SUCCESS;
        }

        params.placement = *opt_not_in_place ? fft_placement_notinplace : fft_placement_inplace;
        if(*opt_double)
            params.precision = fft_precision_double;

        if(*opt_length)
        {
            std::cout << "length:";
            for(auto& i : params.length)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(*opt_istride)
        {
            std::cout << "istride:";
            for(auto& i : params.istride)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(*opt_ostride)
        {
            std::cout << "ostride:";
            for(auto& i : params.ostride)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(params.idist > 0)
        {
            std::cout << "idist: " << params.idist << "\n";
        }
        if(params.odist > 0)
        {
            std::cout << "odist: " << params.odist << "\n";
        }

        if(*opt_ioffset)
        {
            std::cout << "ioffset:";
            for(auto& i : params.ioffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(*opt_ooffset)
        {
            std::cout << "ooffset:";
            for(auto& i : params.ooffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
    }
    std::cout << std::flush;

    // Set GPU for single-device FFT computation
    rocfft_scoped_device dev(deviceId);

    params.validate();

    if(!params.valid(verbose))
    {
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    }

    std::cout << "Token: " << params.token() << std::endl;
    if(verbose)
    {
        std::cout << params.str() << std::endl;
    }

    // Check free and total available memory:
    size_t free  = 0;
    size_t total = 0;
    HIP_V_THROW(hipMemGetInfo(&free, &total), "hipMemGetInfo failed");

    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);
    if(!vram_fits_problem(raw_vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << raw_vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    // GPU input buffer:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        HIP_V_THROW(ibuffer[i].alloc(ibuffer_sizes[i]), "Creating input Buffer failed");
        pibuffer[i] = ibuffer[i].data();
    }

    // CPU input buffer
    std::vector<hostbuf> ibuffer_cpu;

    auto is_device_gen = (params.igen == fft_input_generator_device
                          || params.igen == fft_input_random_generator_device);
    auto is_host_gen   = (params.igen == fft_input_generator_host
                        || params.igen == fft_input_random_generator_host);

    if(is_device_gen)
    {
        // Input data:
        params.compute_input(ibuffer);

        if(verbose > 1)
        {
            // Copy input to CPU
            ibuffer_cpu = allocate_host_buffer(params.precision, params.itype, params.isize);
            for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
            {
                HIP_V_THROW(hipMemcpy(ibuffer_cpu.at(idx).data(),
                                      ibuffer[idx].data(),
                                      ibuffer_sizes[idx],
                                      hipMemcpyDeviceToHost),
                            "hipMemcpy failed");
            }

            std::cout << "GPU input:\n";
            params.print_ibuffer(ibuffer_cpu);
        }
    }

    if(is_host_gen)
    {
        // Input data:
        ibuffer_cpu = allocate_host_buffer(params.precision, params.itype, params.isize);
        params.compute_input(ibuffer_cpu);

        if(verbose > 1)
        {
            std::cout << "GPU input:\n";
            params.print_ibuffer(ibuffer_cpu);
        }

        for(unsigned int idx = 0; idx < ibuffer_cpu.size(); ++idx)
        {
            HIP_V_THROW(hipMemcpy(pibuffer[idx],
                                  ibuffer_cpu[idx].data(),
                                  ibuffer_cpu[idx].size(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
        }
    }

    // GPU output buffer:
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            HIP_V_THROW(obuffer_data[i].alloc(obuffer_sizes[i]), "Creating output Buffer failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // Execution times for loaded libraries:
    std::vector<std::vector<double>> time(lib_strings.size());

    // If we are doing a reverse-run, then we need two ntrials; otherwise, just one.
    std::vector<int> ntrial_runs;
    if(reverse == 0)
    {
        ntrial_runs.push_back(ntrial);
    }
    else
    {
        ntrial_runs.push_back((ntrial + 1) / 2);
        ntrial_runs.push_back(ntrial / 2);
    }

    for(size_t ridx = 0; ridx < ntrial_runs.size(); ++ridx)
    {

        std::vector<size_t> timeindex;
        for(size_t i = 0; i < lib_strings.size(); ++i)
        {
            timeindex.push_back(i);
        }
        if(ridx == 1)
        {
            std::reverse(lib_strings.begin(), lib_strings.end());
            std::reverse(timeindex.begin(), timeindex.end());
        }

        // Create the handles to the libs and the associated fft plans.
        std::vector<ROCFFT_LIB>  handle;
        std::vector<rocfft_plan> plan;
        // Allocate the work buffer: just one, big enough for any dloaded library.
        std::vector<rocfft_execution_info> info;
        size_t                             wbuffer_size = 0;
        for(unsigned int idx = 0; idx < lib_strings.size(); ++idx)
        {
            std::cout << idx << ": " << lib_strings[idx] << "\n";
            auto libhandle = rocfft_lib_load(lib_strings[idx]);
            if(libhandle == NULL)
            {
                std::cout << "Failed to open " << lib_strings[idx]
                          << ", error: " << rocfft_lib_load_error() << "\n";
                return 1;
            }
            handle.push_back(libhandle);
            plan.push_back(make_plan(handle[idx], params));
            show_plan(handle[idx], plan[idx]);
            wbuffer_size = std::max(wbuffer_size, get_wbuffersize(handle[idx], plan[idx]));
            info.push_back(make_execinfo(handle[idx]));
        }

        std::cout << "Work buffer size: " << wbuffer_size << std::endl;
        gpubuf wbuffer;
        if(wbuffer_size)
        {
            HIP_V_THROW(wbuffer.alloc(wbuffer_size), "Creating intermediate Buffer failed");
        }

        // Associate the work buffer to the individual libraries:
        for(unsigned int idx = 0; idx < lib_strings.size(); ++idx)
        {
            set_work_buffer(handle[idx], info[idx], wbuffer_size, wbuffer.data());
        }

        // Run the plan using its associated rocFFT library:
        for(unsigned int idx = 0; idx < handle.size(); ++idx)
        {
            run_plan(handle[idx], plan[idx], info[idx], pibuffer.data(), pobuffer.data());
        }

        std::vector<int> testcase(ntrial_runs[ridx] * lib_strings.size());

        switch(test_sequence)
        {
        case 0:
        {
            // Random order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < lib_strings.size(); ++ilib)
                {
                    testcase[lib_strings.size() * itrial + ilib] = ilib;
                }
            }
            std::random_device rd;
            std::mt19937       g(rd());
            std::shuffle(testcase.begin(), testcase.end(), g);
            break;
        }
        case 1:
            // Alternating order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < lib_strings.size(); ++ilib)
                {
                    testcase[lib_strings.size() * itrial + ilib] = ilib;
                }
            }
            break;
        case 2:
            // Sequential order:
            for(int itrial = 0; itrial < ntrial_runs[ridx]; ++itrial)
            {
                for(size_t ilib = 0; ilib < lib_strings.size(); ++ilib)
                {
                    testcase[ilib * ntrial + itrial] = ilib;
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid test sequence choice.");
        }

        if(verbose > 3)
        {
            std::cout << "Test case order:";
            for(const auto val : testcase)
                std::cout << " " << val;
            std::cout << "\n";
        }

        std::cout << "Running the tests...\n";

        for(size_t itest = 0; itest < testcase.size(); ++itest)
        {
            const int tidx = testcase[itest];

            if(verbose > 3)
            {
                std::cout << "running test case " << tidx << "\n";
            }

            if(is_device_gen)
            {
                params.compute_input(ibuffer);
            }
            if(is_host_gen)
            {
                for(unsigned int bidx = 0; bidx < ibuffer_cpu.size(); ++bidx)
                {
                    HIP_V_THROW(hipMemcpy(pibuffer[bidx],
                                          ibuffer_cpu[bidx].data(),
                                          ibuffer_cpu[bidx].size(),
                                          hipMemcpyHostToDevice),
                                "hipMemcpy failed");
                }
            }

            // Run the plan using its associated rocFFT library:
            time[tidx].push_back(
                run_plan(handle[tidx], plan[tidx], info[tidx], pibuffer.data(), pobuffer.data()));

            if(verbose > 2)
            {
                auto output = allocate_host_buffer(params.precision, params.otype, params.osize);
                for(unsigned int iout = 0; iout < output.size(); ++iout)
                {
                    HIP_V_THROW(hipMemcpy(output[iout].data(),
                                          pobuffer[iout],
                                          output[iout].size(),
                                          hipMemcpyDeviceToHost),
                                "hipMemcpy failed");
                }
                std::cout << "GPU output:\n";
                params.print_obuffer(output);
            }
        }

        // Clean up:
        for(unsigned int hidx = 0; hidx < handle.size(); ++hidx)
        {
            destroy_info(handle[hidx], info[hidx]);
            destroy_plan(handle[hidx], plan[hidx]);
            rocfft_lib_close(handle[hidx]);
        }
    }

    std::cout << "Execution times in ms:\n";
    for(unsigned int idx = 0; idx < time.size(); ++idx)
    {
        std::cout << "\nExecution gpu time:";
        for(auto& i : time[idx])
        {
            std::cout << " " << i;
        }
        std::cout << " ms" << std::endl;
    }

    return EXIT_SUCCESS;
}
