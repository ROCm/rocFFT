// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <math.h>
#include <vector>

#include <dlfcn.h>
#include <link.h>

#include "rider.h"
#include "rocfft.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Given a libhandle from dload, return a plan to a rocFFT plan with the given parameters.
rocfft_plan make_plan(void*                         libhandle,
                      const rocfft_result_placement place,
                      const rocfft_transform_type   transformType,
                      const std::vector<size_t>&    length,
                      const std::vector<size_t>&    istride,
                      const std::vector<size_t>&    ostride,
                      const size_t                  idist,
                      const size_t                  odist,
                      const std::vector<size_t>     ioffset,
                      const std::vector<size_t>     ooffset,
                      const size_t                  nbatch,
                      const rocfft_precision        precision,
                      const rocfft_array_type       itype,
                      const rocfft_array_type       otype)
{
    auto procfft_setup = (decltype(&rocfft_setup))dlsym(libhandle, "rocfft_setup");
    if(procfft_setup == NULL)
        exit(1);
    auto procfft_plan_description_create = (decltype(&rocfft_plan_description_create))dlsym(
        libhandle, "rocfft_plan_description_create");
    auto procfft_plan_description_destroy = (decltype(&rocfft_plan_description_destroy))dlsym(
        libhandle, "rocfft_plan_description_destroy");
    auto procfft_plan_description_set_data_layout
        = (decltype(&rocfft_plan_description_set_data_layout))dlsym(
            libhandle, "rocfft_plan_description_set_data_layout");
    auto procfft_plan_create
        = (decltype(&rocfft_plan_create))dlsym(libhandle, "rocfft_plan_create");
    auto procfft_execute = (decltype(&rocfft_execute))dlsym(libhandle, "rocfft_execute");

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

    procfft_plan_create(
        &plan, place, transformType, precision, length.size(), length.data(), nbatch, desc);

    LIB_V_THROW(procfft_plan_description_destroy(desc), "rocfft_plan_description_destroy failed");

    return plan;
}

// Given a libhandle from dload and a rocFFT plan, destroy the plan.
void destroy_plan(void* libhandle, rocfft_plan& plan)
{
    auto procfft_plan_destroy
        = (decltype(&rocfft_plan_destroy))dlsym(libhandle, "rocfft_plan_destroy");
    procfft_plan_destroy(plan);
    auto procfft_cleanup = (decltype(&rocfft_cleanup))dlsym(libhandle, "rocfft_cleanup");
    if(procfft_cleanup)
        procfft_cleanup();
}

// Given a libhandle from dload and a rocFFT execution info structure, destroy the info.
void destroy_info(void* libhandle, rocfft_execution_info& info)
{
    auto procfft_execution_info_destroy = (decltype(&rocfft_execution_info_destroy))dlsym(
        libhandle, "rocfft_execution_info_destroy");
    procfft_execution_info_destroy(info);
}

// Given a libhandle from dload, and a corresponding rocFFT plan, return how much work
// buffer is required.
size_t get_wbuffersize(void* libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_work_buffer_size = (decltype(&rocfft_plan_get_work_buffer_size))dlsym(
        libhandle, "rocfft_plan_get_work_buffer_size");

    // Get the buffersize
    size_t workBufferSize = 0;
    LIB_V_THROW(procfft_plan_get_work_buffer_size(plan, &workBufferSize),
                "rocfft_plan_get_work_buffer_size failed");

    return workBufferSize;
}

// Given a libhandle from dload and a corresponding rocFFT plan, print the plan information.
void show_plan(void* libhandle, const rocfft_plan& plan)
{
    auto procfft_plan_get_print
        = (decltype(&rocfft_plan_get_print))dlsym(libhandle, "rocfft_plan_get_print");

    LIB_V_THROW(procfft_plan_get_print(plan), "rocfft_plan_get_print failed");
}

// Given a libhandle from dload and a corresponding rocFFT plan, a work buffer size and an
// allocated work buffer, return a rocFFT execution info for the plan.
rocfft_execution_info make_execinfo(void* libhandle, const size_t wbuffersize, void* wbuffer)
{
    auto procfft_execution_info_create
        = (decltype(&rocfft_execution_info_create))dlsym(libhandle, "rocfft_execution_info_create");
    auto procfft_execution_info_set_work_buffer
        = (decltype(&rocfft_execution_info_set_work_buffer))dlsym(
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
float run_plan(void* libhandle, rocfft_plan plan, rocfft_execution_info info, void** in, void** out)
{
    auto procfft_execute = (decltype(&rocfft_execute))dlsym(libhandle, "rocfft_execute");

    hipEvent_t start, stop;
    HIP_V_THROW(hipEventCreate(&start), "hipEventCreate failed");
    HIP_V_THROW(hipEventCreate(&stop), "hipEventCreate failed");

    HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

    procfft_execute(plan, in, out, info);

    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

    float time;
    hipEventElapsedTime(&time, start, stop);
    return time;
}

int main(int argc, char* argv[])
{
    // Control output verbosity:
    int verbose;

    // hip Device number for running tests:
    int deviceId;

    // Transform type parameters:
    rocfft_transform_type transformType;
    rocfft_array_type     itype;
    rocfft_array_type     otype;

    // Number of performance trial samples
    int ntrial;

    // Number of batches:
    size_t nbatch;

    // TODO: enable when enabled in rocFFT
    // // Scale for transform
    // double scale = 1.0;

    // Transform length:
    std::vector<size_t> length;

    // Transform input and output strides:
    std::vector<size_t> istride;
    std::vector<size_t> ostride;

    // Offset to start of buffer (or buffers, for planar format):
    std::vector<size_t> ioffset;
    std::vector<size_t> ooffset;

    // Input and output distances:
    size_t idist;
    size_t odist;

    // Vector of directories which contain librocfft.so
    std::vector<std::string> libdir;

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("rocfft rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ("transformType,t", po::value<rocfft_transform_type>(&transformType)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "idist", po::value<size_t>(&idist)->default_value(0),
          "input distance between successive members when batch size > 1")
        ( "odist", po::value<size_t>(&odist)->default_value(0),
          "output distance between successive members when batch size > 1")
        // ("scale", po::value<double>(&scale)->default_value(1.0), "Specify the scaling factor ")
        ( "batchSize,b", po::value<size_t>(&nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<rocfft_array_type>(&itype)
          ->default_value(rocfft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&otype)
          ->default_value(rocfft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("lib",  po::value<std::vector<std::string>>(&libdir)->multitoken(), "libdirs.")
        ("length",  po::value<std::vector<size_t>>(&length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<size_t>>(&istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<size_t>>(&ostride)->multitoken(), "Output strides.")
        ("ioffset", po::value<std::vector<size_t>>(&ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<size_t>>(&ooffset)->multitoken(), "Output offsets.");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << opdesc << std::endl;
        return 0;
    }

    const rocfft_result_placement place
        = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    const rocfft_precision precision
        = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

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

    if(vm.count("length"))
    {
        std::cout << "length:";
        for(auto& i : length)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("istride"))
    {
        std::cout << "istride:";
        for(auto& i : istride)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ostride"))
    {
        std::cout << "ostride:";
        for(auto& i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(idist > 0)
    {
        std::cout << "idist: " << idist << "\n";
    }
    if(odist > 0)
    {
        std::cout << "odist: " << odist << "\n";
    }

    if(vm.count("ioffset"))
    {
        std::cout << "ioffset:";
        for(auto& i : ioffset)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ooffset"))
    {
        std::cout << "ooffset:";
        for(auto& i : ooffset)
            std::cout << " " << i;
        std::cout << "\n";
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    // HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    // Set default data formats if not yet specified:
    const size_t dim     = length.size();
    auto         ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
    {
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    }
    if(istride.size() == 0)
    {
        istride = compute_stride(ilength,
                                 1,
                                 place == rocfft_placement_inplace
                                     && transformType == rocfft_transform_type_real_forward);
    }
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
    {
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    }
    if(ostride.size() == 0)
    {
        ostride = compute_stride(olength,
                                 1,
                                 place == rocfft_placement_inplace
                                     && transformType == rocfft_transform_type_real_inverse);
    }
    check_set_iotypes(place, transformType, itype, otype);
    if(idist == 0)
    {
        idist = set_idist(place, transformType, length, istride);
    }
    if(odist == 0)
    {
        odist = set_odist(place, transformType, length, ostride);
    }

    if(verbose > 0)
    {
        std::cout << "FFT  params:\n";
        std::cout << "\tilength:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tistride:";
        for(auto i : istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tidist: " << idist << std::endl;

        std::cout << "\tolength:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tostride:";
        for(auto i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\todist: " << odist << std::endl;
    }

    std::vector<rocfft_plan> plan;

    size_t wbuffer_size = 0;

    // Set up shared object handles
    std::vector<void*> handles;
    for(int idx = 0; idx < libdir.size(); ++idx)
    {
        void* libhandle = dlopen((libdir[idx] + "/librocfft.so").c_str(), RTLD_LAZY);
        if(libhandle == NULL)
        {
            std::cout << "Failed to open " << libdir[idx] << std::endl;
            exit(1);
        }
        struct link_map* link = nullptr;
        dlinfo(libhandle, RTLD_DI_LINKMAP, &link);
        for(; link != nullptr; link = link->l_next)
        {
            if(strstr(link->l_name, "librocfft-device") != nullptr)
            {
                std::cerr << "Error: Library " << libdir[idx] << " depends on librocfft-device.\n";
                std::cerr << "All libraries need to be built with -DSINGLELIB=on.\n";
                exit(1);
            }
        }
        handles.push_back(libhandle);
    }

    // Set up plans:
    for(int idx = 0; idx < libdir.size(); ++idx)
    {
        // Create column-major parameters for rocFFT:
        auto length_cm  = length;
        auto istride_cm = istride;
        auto ostride_cm = ostride;
        for(int idx = 0; idx < dim / 2; ++idx)
        {
            const auto toidx = dim - idx - 1;
            std::swap(istride_cm[idx], istride_cm[toidx]);
            std::swap(ostride_cm[idx], ostride_cm[toidx]);
            std::swap(length_cm[idx], length_cm[toidx]);
        }

        std::cout << idx << ": " << libdir[idx] << std::endl;
        plan.push_back(make_plan(handles[idx],
                                 place,
                                 transformType,
                                 length_cm,
                                 istride_cm,
                                 ostride_cm,
                                 idist,
                                 odist,
                                 ioffset,
                                 ooffset,
                                 nbatch,
                                 precision,
                                 itype,
                                 otype));
        show_plan(handles[idx], plan[idx]);
        wbuffer_size = std::max(wbuffer_size, get_wbuffersize(handles[idx], plan[idx]));
    }

    std::cout << "Work buffer size: " << wbuffer_size << std::endl;

    // Allocate the work buffer: just one, big enough for any dloaded library.
    void* wbuffer = NULL;
    if(wbuffer_size)
    {
        HIP_V_THROW(hipMalloc(&wbuffer, wbuffer_size), "Creating intermediate Buffer failed");
    }

    // Associate the work buffer to the invidual libraries:
    std::vector<rocfft_execution_info> info;
    for(int idx = 0; idx < libdir.size(); ++idx)
    {
        info.push_back(make_execinfo(handles[idx], wbuffer_size, wbuffer));
    }

    // Input data:
    const auto input = compute_input(precision, itype, length, istride, idist, nbatch);

    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        printbuffer(precision, itype, input, ilength, istride, nbatch, idist);
    }

    // GPU input and output buffers:
    auto               ibuffer_sizes = buffer_sizes(precision, itype, idist, nbatch);
    std::vector<void*> ibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        HIP_V_THROW(hipMalloc(&ibuffer[i], ibuffer_sizes[i]), "Creating input Buffer failed");
    }

    std::vector<void*> obuffer;
    if(place == rocfft_placement_inplace)
    {
        obuffer = ibuffer;
    }
    else
    {
        auto obuffer_sizes = buffer_sizes(precision, otype, odist, nbatch);
        obuffer.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer.size(); ++i)
        {
            HIP_V_THROW(hipMalloc(&obuffer[i], obuffer_sizes[i]), "Creating output Buffer failed");
        }
    }

    if(handles.size())
    {
        // Run a kernel once to load the instructions on the GPU:

        // Copy the input data to the GPU:
        for(int idx = 0; idx < input.size(); ++idx)
        {
            HIP_V_THROW(
                hipMemcpy(ibuffer[0], input[0].data(), input[0].size(), hipMemcpyHostToDevice),
                "hipMemcpy failed");
        }
        // Run the plan using its associated rocFFT library:
        for(int idx = 0; idx < handles.size(); ++idx)
        {
            run_plan(handles[idx], plan[idx], info[idx], ibuffer.data(), obuffer.data());
        }
    }

    // Execution times for loaded libraries:
    std::vector<std::vector<double>> time(libdir.size());

    // Run the FFTs from the different libraries in random order until they all have at
    // least ntrial times.
    std::vector<int> ndone(libdir.size());
    std::fill(ndone.begin(), ndone.end(), 0);
    while(!std::all_of(ndone.begin(), ndone.end(), [&ntrial](int i) { return i >= ntrial; }))
    {
        const int idx = rand() % ndone.size();
        ndone[idx]++;

        // We can optionally require that all runs have exactly ntrial, but it may be more
        // iid to just let things run:
        // if(ndone[idx] > ntrial)
        //     continue;

        // Copy the input data to the GPU:
        for(int idx = 0; idx < input.size(); ++idx)
        {
            HIP_V_THROW(
                hipMemcpy(
                    ibuffer[idx], input[idx].data(), input[idx].size(), hipMemcpyHostToDevice),
                "hipMemcpy failed");
        }

        // Run the plan using its associated rocFFT library:
        time[idx].push_back(
            run_plan(handles[idx], plan[idx], info[idx], ibuffer.data(), obuffer.data()));

        if(verbose > 2)
        {
            auto output = allocate_host_buffer(precision, otype, olength, ostride, odist, nbatch);
            for(int idx = 0; idx < output.size(); ++idx)
            {
                hipMemcpy(
                    output[idx].data(), obuffer[idx], output[idx].size(), hipMemcpyDeviceToHost);
            }
            std::cout << "GPU output:\n";
            printbuffer(precision, otype, output, olength, ostride, nbatch, odist);
        }
    }

    std::cout << "Execution times in ms:\n";
    for(int idx = 0; idx < time.size(); ++idx)
    {
        std::cout << "\nExecution gpu time:";
        for(auto& i : time[idx])
        {
            std::cout << " " << i;
        }
        std::cout << " ms" << std::endl;
    }

    // Clean up:
    for(int idx = 0; idx < handles.size(); ++idx)
    {
        destroy_info(handles[idx], info[idx]);
        destroy_plan(handles[idx], plan[idx]);
        dlclose(handles[idx]);
    }
    hipFree(wbuffer);
    for(auto& buf : ibuffer)
        hipFree(buf);
    for(auto& buf : obuffer)
        hipFree(buf);

    return 0;
}
