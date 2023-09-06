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
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <math.h>
#include <vector>

#ifdef WIN32
#include <windows.h>

#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "bench.h"
#include "rocfft/rocfft.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

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

// Return true if rocfft_device is loaded, which indicates that the
// library was not built with -DSINGLELIB=ON.
bool rocfft_lib_device_loaded(ROCFFT_LIB libhandle)
{
#ifdef WIN32
    DWORD arraySize = 0;
    EnumProcessModules(GetCurrentProcess(), NULL, 0, &arraySize);
    std::vector<HMODULE> modules(arraySize);
    if(EnumProcessModules(GetCurrentProcess(), modules.data(), modules.size(), &arraySize))
    {
        for(auto& mod : modules)
        {
            char name[MAX_PATH] = {0};
            GetModuleFileNameA(mod, name, MAX_PATH);
            // poor man's stristr on windows
            std::transform(name, name + strlen(name), name, [](char c) { return std::tolower(c); });
            if(strstr(name, "rocfft-device.dll"))
                return true;
        }
    }
    return false;
#else
    struct link_map* link = nullptr;
    dlinfo(libhandle, RTLD_DI_LINKMAP, &link);
    for(; link != nullptr; link = link->l_next)
    {
        if(strstr(link->l_name, "librocfft-device") != nullptr)
        {
            return true;
        }
    }
    return false;
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
rocfft_plan make_plan(ROCFFT_LIB                    libhandle,
                      const rocfft_result_placement place,
                      const fft_transform_type      transform_type,
                      const std::vector<size_t>&    length,
                      const std::vector<size_t>&    istride,
                      const std::vector<size_t>&    ostride,
                      const size_t                  idist,
                      const size_t                  odist,
                      const std::vector<size_t>&    ioffset,
                      const std::vector<size_t>&    ooffset,
                      const size_t                  nbatch,
                      const rocfft_precision        precision,
                      const rocfft_array_type       itype,
                      const rocfft_array_type       otype)
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
    LIB_V_THROW(procfft_plan_description_set_data_layout(desc,
                                                         itype,
                                                         otype,
                                                         ioffset.data(),
                                                         ooffset.data(),
                                                         istride.size(),
                                                         istride.data(),
                                                         idist,
                                                         ostride.size(),
                                                         ostride.data(),
                                                         odist),
                "rocfft_plan_description_data_layout failed");
    rocfft_plan plan = NULL;

    LIB_V_THROW(procfft_plan_create(&plan,
                                    place,
                                    rocfft_transform_type_from_fftparams(transform_type),
                                    precision,
                                    length.size(),
                                    length.data(),
                                    nbatch,
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

// Given a libhandle from dload and a corresponding rocFFT plan, a work buffer size and an
// allocated work buffer, return a rocFFT execution info for the plan.
rocfft_execution_info make_execinfo(ROCFFT_LIB libhandle, const size_t wbuffersize, void* wbuffer)
{
    auto procfft_execution_info_create = (decltype(&rocfft_execution_info_create))rocfft_lib_symbol(
        libhandle, "rocfft_execution_info_create");
    auto procfft_execution_info_set_work_buffer
        = (decltype(&rocfft_execution_info_set_work_buffer))rocfft_lib_symbol(
            libhandle, "rocfft_execution_info_set_work_buffer");

    rocfft_execution_info info = NULL;
    LIB_V_THROW(procfft_execution_info_create(&info), "rocfft_execution_info_create failed");
    if(wbuffer != NULL)
    {
        LIB_V_THROW(procfft_execution_info_set_work_buffer(info, wbuffer, wbuffersize),
                    "rocfft_execution_info_set_work_buffer failed");
    }

    return info;
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

    procfft_execute(plan, in, out, info);

    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

    float time;
    HIP_V_THROW(hipEventElapsedTime(&time, start, stop), "hipEventElapsedTime failed");
    return time;
}

// Load python library with RTLD_GLOBAL so that rocfft is free to
// import python modules that need all of the symbols in libpython.
// Normally, dyna-bench will want to dlopen rocfft's with RTLD_LOCAL.
// If libpython is brought in this way, python modules might not be
// able to find the symbols they need and import will fail.
#ifndef WIN32
static void* python_dl = nullptr;
void         load_python(const std::vector<std::string>& libs)
{
    // dlopen each lib, taking note of the python library that it needs
    std::string pythonlib;
    for(const auto& lib : libs)
    {
        void* handle = dlopen(lib.c_str(), RTLD_LAZY);
        if(handle)
        {
            // look through the link map to see what libpython it needs (if any)
            struct link_map* map;
            if(dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0)
            {
                for(struct link_map* ptr = map; ptr != nullptr; ptr = ptr->l_next)
                {
                    std::string libname = ptr->l_name;
                    if(libname.find("/libpython3.") != std::string::npos)
                    {
                        if(!pythonlib.empty() && pythonlib != libname)
                            throw std::runtime_error("multiple distinct libpythons required");
                        pythonlib = libname;
                    }
                }
            }
        }
        dlclose(handle);
    }

    if(!pythonlib.empty())
    {
        // explicitly dlopen python with RTLD_GLOBAL
        python_dl = dlopen(pythonlib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    }
}
#endif

int main(int argc, char* argv[])
{
    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples:
    int ntrial{};

    // Test sequence choice:
    int test_sequence{};

    // Vector of test target libraries
    std::vector<std::string> libs;

    // FFT parameters:
    fft_params params;

    // Token string to fully specify fft params.
    std::string token;

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("dyna-rocfft-bench command line options");
    opdesc.add_options()("help,h", "Produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("sequence", po::value<int>(&test_sequence)->default_value(0),
         "Test sequence: random(0), alternating(1) sequential(2)")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (deprecated: use --precision double)")
        ("precision", po::value<fft_precision>(&params.precision), "Transform precision: single (default), double, half")
        ("inputGen,g", po::value<fft_input_generator>(&params.igen)
        ->default_value(fft_input_random_generator_device),
        "Input data generation:\n0) PRNG sequence (device)\n"
        "1) PRNG sequence (host)\n"
        "2) linearly-spaced sequence (device)\n"
        "3) linearly-spaced sequence (host)")
        ("transformType,t", po::value<fft_transform_type>(&params.transform_type)
         ->default_value(fft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "batchSize,b", po::value<size_t>(&params.nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<fft_array_type>(&params.itype)
          ->default_value(fft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<fft_array_type>(&params.otype)
          ->default_value(fft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("lib",  po::value<std::vector<std::string>>(&libs)->multitoken(),
         "Set test target library full path(appendable).")
        ("length",  po::value<std::vector<size_t>>(&params.length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<size_t>>(&params.istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<size_t>>(&params.ostride)->multitoken(), "Output strides.")
        ("idist", po::value<size_t>(&params.idist)->default_value(0),
         "Logical distance between input batches.")
        ("odist", po::value<size_t>(&params.odist)->default_value(0),
         "Logical distance between output batches.")
        ("isize", po::value<std::vector<size_t>>(&params.isize)->multitoken(),
         "Logical size of input buffer.")
        ("osize", po::value<std::vector<size_t>>(&params.osize)->multitoken(),
         "Logical size of output.")
        ("ioffset", po::value<std::vector<size_t>>(&params.ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<size_t>>(&params.ooffset)->multitoken(), "Output offsets.")
        ("scalefactor", po::value<double>(&params.scale_factor), "Scale factor to apply to output.")
        ("token", po::value<std::string>(&token));
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    if(vm.count("notInPlace"))
    {
        std::cout << "out-of-place\n";
    }
    else
    {
        std::cout << "in-place\n";
    }

    if(vm.count("ntrial"))
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(token != "")
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
        if(!vm.count("length"))
        {
            std::cout << "Please specify transform length!" << std::endl;
            std::cout << opdesc << std::endl;
            return EXIT_SUCCESS;
        }

        params.placement
            = vm.count("notInPlace") ? fft_placement_notinplace : fft_placement_inplace;
        if(vm.count("double"))
            params.precision = fft_precision_double;

        if(vm.count("notInPlace"))
        {
            std::cout << "out-of-place\n";
        }
        else
        {
            std::cout << "in-place\n";
        }

        if(vm.count("length"))
        {
            std::cout << "length:";
            for(auto& i : params.length)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(vm.count("istride"))
        {
            std::cout << "istride:";
            for(auto& i : params.istride)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(vm.count("ostride"))
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

        if(vm.count("ioffset"))
        {
            std::cout << "ioffset:";
            for(auto& i : params.ioffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(vm.count("ooffset"))
        {
            std::cout << "ooffset:";
            for(auto& i : params.ooffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    // HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

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

    std::vector<rocfft_plan> plan;

    size_t wbuffer_size = 0;

#ifndef WIN32
    load_python(libs);
#endif

    // Set up shared object handles
    std::vector<ROCFFT_LIB> handles;
    for(unsigned int idx = 0; idx < libs.size(); ++idx)
    {
        auto libhandle = rocfft_lib_load(libs[idx]);
        if(libhandle == NULL)
        {
            std::cout << "Failed to open " << libs[idx] << ", error: " << rocfft_lib_load_error()
                      << std::endl;
            return 1;
        }
        if(rocfft_lib_device_loaded(libhandle))
        {
            std::cerr << "Error: Library " << libs[idx] << " depends on librocfft-device.\n";
            std::cerr << "All libraries need to be built with -DSINGLELIB=on.\n";
            return 1;
        }
        handles.push_back(libhandle);
    }

    // Set up plans:
    for(unsigned int idx = 0; idx < libs.size(); ++idx)
    {
        std::cout << idx << ": " << libs[idx] << std::endl;
        plan.push_back(make_plan(handles[idx],
                                 rocfft_result_placement_from_fftparams(params.placement),
                                 params.transform_type,
                                 params.length_cm(),
                                 params.istride_cm(),
                                 params.ostride_cm(),
                                 params.idist,
                                 params.odist,
                                 params.ioffset,
                                 params.ooffset,
                                 params.nbatch,
                                 rocfft_precision_from_fftparams(params.precision),
                                 rocfft_array_type_from_fftparams(params.itype),
                                 rocfft_array_type_from_fftparams(params.otype)));
        show_plan(handles[idx], plan[idx]);
        wbuffer_size = std::max(wbuffer_size, get_wbuffersize(handles[idx], plan[idx]));
    }

    std::cout << "Work buffer size: " << wbuffer_size << std::endl;

    // Allocate the work buffer: just one, big enough for any dloaded library.
    gpubuf wbuffer;
    if(wbuffer_size)
    {
        HIP_V_THROW(wbuffer.alloc(wbuffer_size), "Creating intermediate Buffer failed");
    }

    // Associate the work buffer to the invidual libraries:
    std::vector<rocfft_execution_info> info;
    for(unsigned int idx = 0; idx < libs.size(); ++idx)
    {
        info.push_back(make_execinfo(handles[idx], wbuffer_size, wbuffer.data()));
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

    if(handles.size())
    {
        // Run the plan using its associated rocFFT library:
        for(unsigned int idx = 0; idx < handles.size(); ++idx)
        {
            run_plan(handles[idx], plan[idx], info[idx], pibuffer.data(), pobuffer.data());
        }
    }

    // Execution times for loaded libraries:
    std::vector<std::vector<double>> time(libs.size());

    std::vector<int> testcase(ntrial * libs.size());
    switch(test_sequence)
    {
    case 0:
    {
        // Random order:
        for(int itrial = 0; itrial < ntrial; ++itrial)
        {
            for(size_t ilib = 0; ilib < libs.size(); ++ilib)
            {
                testcase[libs.size() * itrial + ilib] = ilib;
            }
        }

        std::random_device rd;
        std::mt19937       g(rd());
        std::shuffle(testcase.begin(), testcase.end(), g);
        break;
    }
    case 1:
        // Alternating order:
        for(int itrial = 0; itrial < ntrial; ++itrial)
        {
            for(size_t ilib = 0; ilib < libs.size(); ++ilib)
            {
                testcase[libs.size() * itrial + ilib] = ilib;
            }
        }
        break;
    case 2:
        // Sequential order:
        for(int itrial = 0; itrial < ntrial; ++itrial)
        {
            for(size_t ilib = 0; ilib < libs.size(); ++ilib)
            {
                testcase[ilib * ntrial + itrial] = ilib;
            }
        }

        break;
    default:
        throw std::runtime_error("Invalid test sequence choice.");
    }

    std::cout << "test case:";
    for(const auto i : testcase)
        std::cout << " " << i;
    std::cout << "\n";

    // Run the FFTs from the different libraries in random order until they all have at
    // least ntrial times.
    std::vector<int> ndone(libs.size());
    std::fill(ndone.begin(), ndone.end(), 0);
    for(size_t itest = 0; itest < testcase.size(); ++itest)
    {
        const int idx = testcase[itest];

        if(is_device_gen)
        {
            params.compute_input(ibuffer);
        }

        if(is_host_gen)
        {
            for(unsigned int idx = 0; idx < ibuffer_cpu.size(); ++idx)
            {
                HIP_V_THROW(hipMemcpy(pibuffer[idx],
                                      ibuffer_cpu[idx].data(),
                                      ibuffer_cpu[idx].size(),
                                      hipMemcpyHostToDevice),
                            "hipMemcpy failed");
            }
        }

        // Run the plan using its associated rocFFT library:
        time[idx].push_back(
            run_plan(handles[idx], plan[idx], info[idx], pibuffer.data(), pobuffer.data()));

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

    // Clean up:
    for(unsigned int idx = 0; idx < handles.size(); ++idx)
    {
        destroy_info(handles[idx], info[idx]);
        destroy_plan(handles[idx], plan[idx]);
        rocfft_lib_close(handles[idx]);
    }

#ifndef WIN32
    if(python_dl)
        dlclose(python_dl);
#endif

    return EXIT_SUCCESS;
}
