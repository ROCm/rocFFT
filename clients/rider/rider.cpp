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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>

#include "./rider.h"
#include "rocfft.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

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
    size_t nbatch = 1;

    // Scale for transform
    double scale = 1.0;

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

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("rocfft rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->implicit_value(0), "Control output verbosity")
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
        ("scale", po::value<double>(&scale)->default_value(1.0), "Specify the scaling factor ")
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

    if(vm.count("version"))
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "version " << v << std::endl;
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

    // Create FFT description
    rocfft_plan_description desc = NULL;
    LIB_V_THROW(rocfft_plan_description_create(&desc), "rocfft_plan_description_create failed");
    LIB_V_THROW(rocfft_plan_description_set_data_layout(desc,
                                                        itype,
                                                        otype,
                                                        ioffset.data(),
                                                        ooffset.data(),
                                                        istride_cm.size(),
                                                        istride_cm.data(),
                                                        idist,
                                                        ostride_cm.size(),
                                                        ostride_cm.data(),
                                                        odist),
                "rocfft_plan_description_data_layout failed");

    // Create the plan
    rocfft_plan plan = NULL;
    rocfft_plan_create(
        &plan, place, transformType, precision, length_cm.size(), length_cm.data(), nbatch, desc);

    // Get work buffer size and allocated info-associated work buffer is necessary
    size_t workBufferSize = 0;
    LIB_V_THROW(rocfft_plan_get_work_buffer_size(plan, &workBufferSize),
                "rocfft_plan_get_work_buffer_size failed");
    rocfft_execution_info info = NULL;
    LIB_V_THROW(rocfft_execution_info_create(&info), "rocfft_execution_info_create failed");
    void* wbuffer = NULL;
    {
        HIP_V_THROW(hipMalloc(&wbuffer, workBufferSize), "Creating intermediate Buffer failed");
        LIB_V_THROW(rocfft_execution_info_set_work_buffer(info, wbuffer, workBufferSize),
                    "rocfft_execution_info_set_work_buffer failed");
    }

    // Input data:
    const auto input = compute_input(precision, itype, length, istride, idist, nbatch);

    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        printbuffer(precision, itype, input, ilength, istride, nbatch, idist);
    }

    // GPU input and output buffers:
    std::vector<void*> ibuffer = alloc_buffer(precision, itype, idist, nbatch);
    std::vector<void*> obuffer = (place == rocfft_placement_inplace)
                                     ? ibuffer
                                     : alloc_buffer(precision, otype, odist, nbatch);

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    HIP_V_THROW(hipEventCreate(&start), "hipEventCreate failed");
    HIP_V_THROW(hipEventCreate(&stop), "hipEventCreate failed");
    for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {

        // Copy the input data to the GPU:
        for(int idx = 0; idx < input.size(); ++idx)
        {
            HIP_V_THROW(
                hipMemcpy(
                    ibuffer[idx], input[idx].data(), input[idx].size(), hipMemcpyHostToDevice),
                "hipMemcpy failed");
        }

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        rocfft_execute(plan, ibuffer.data(), obuffer.data(), info);

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        hipEventElapsedTime(&time, start, stop);
        gpu_time[itrial] = time;

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

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;

    std::cout << "Execution gflops:  ";
    const double totsize
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    const double k
        = ((itype == rocfft_array_type_real) || (otype == rocfft_array_type_real)) ? 2.5 : 5.0;
    const double opscount = (double)nbatch * k * totsize * log(totsize) / log(2.0);
    for(const auto& i : gpu_time)
    {
        std::cout << " " << opscount / (1e6 * i);
    }
    std::cout << std::endl;

    // Clean up:
    rocfft_plan_description_destroy(desc);
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    hipFree(wbuffer);
    for(auto& buf : ibuffer)
        hipFree(buf);
    for(auto& buf : obuffer)
        hipFree(buf);
}
