// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

#include <stdexcept>

int main(int argc, char* argv[])
{
    std::cout << "rocfft single-node multi-gpu complex-to-complex 3D FFT example\n";

    // Length of transform, first dimension must be greather than number of GPU devices
    std::vector<size_t> length = {8, 8, 8};

    // Gpu device ids:
    std::vector<size_t> devices = {0, 1};

    // Command-line options:
    // clang-format off
    po::options_description desc("rocfft sample command line options");
    desc.add_options()("help,h", "Produces this help message")
        ("length", po::value<std::vector<size_t>>(&length)->multitoken(),
         "3-D FFT size (eg: --length 256 256 256).")
        ("devices", po::value<std::vector<size_t>>(&devices)->multitoken(), 
        "List of devices to use separated by spaces (eg: --devices 1 3)");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    int    deviceCount = devices.size();
    size_t fftSize     = length[0] * length[1] * length[2]; // must evenly divide deviceCount
    int    nDevices;
    (void)hipGetDeviceCount(&nDevices);

    if(length.size() != 3 || deviceCount != 2)
        throw std::runtime_error("This example is designed to run on two devices with 3-D inputs!");

    std::cout << "Number of available GPUs: " << nDevices << " \n";
    if(nDevices <= static_cast<int>(*std::max_element(devices.begin(), devices.end())))
        throw std::runtime_error("device ID greater than number of available devices");

    // Placeness for the transform
    auto rc = rocfft_status_success;
    rc      = rocfft_setup();
    if(rc != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    const rocfft_result_placement place = rocfft_placement_notinplace;

    // Direction of transform
    const rocfft_transform_type direction = rocfft_transform_type_complex_forward;

    rocfft_plan_description description = nullptr;
    rocfft_plan_description_create(&description);
    // Do not set stride information via the descriptor, they are to be defined during field creation below
    rocfft_plan_description_set_data_layout(description,
                                            rocfft_array_type_complex_interleaved,
                                            rocfft_array_type_complex_interleaved,
                                            nullptr,
                                            nullptr,
                                            0,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            0);

    // Define infield geometry
    // first entry of upper dimension is the batch size
    std::vector<size_t> inbrick0_lower = {0, 0, 0, 0};
    std::vector<size_t> inbrick0_upper = {1, length[0] / deviceCount, length[1], length[2]};
    std::vector<size_t> inbrick1_lower = {0, length[0] / deviceCount, 0, 0};
    std::vector<size_t> inbrick1_upper = {1, length[0], length[1], length[2]};

    // Row-major stride for brick data layout in memory
    size_t              idist        = fftSize; // distance between batches
    std::vector<size_t> brick_stride = {idist, length[0] * length[1], length[0], 1};

    rocfft_field infield = nullptr;
    rocfft_field_create(&infield);
    rocfft_field_add_brick(infield,
                           inbrick0_lower.data(),
                           inbrick0_upper.data(),
                           brick_stride.data(),
                           inbrick0_lower.size(),
                           devices[0]); // device id
    rocfft_field_add_brick(infield,
                           inbrick1_lower.data(),
                           inbrick1_upper.data(),
                           brick_stride.data(),
                           inbrick1_lower.size(),
                           devices[1]); // device id
    rocfft_plan_description_add_infield(description, infield);

    // Allocate and initialize GPU input
    std::vector<void*>                gpu_in(2);
    size_t                            bufferSize = fftSize / deviceCount;
    std::vector<std::complex<double>> input(bufferSize, 0.1); // host test input
    size_t                            memSize = sizeof(std::complex<double>) * bufferSize;

    if(hipSetDevice(devices[0]) != hipSuccess)
        throw std::runtime_error("hipSetDevice failed");
    if(hipMalloc(&gpu_in[0], memSize) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    if(hipMemcpy(gpu_in[0], input.data(), memSize, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    if(hipSetDevice(devices[1]) != hipSuccess)
        throw std::runtime_error("hipSetDevice failed");
    if(hipMalloc(&gpu_in[1], memSize) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    if(hipMemcpy(gpu_in[1], input.data(), memSize, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

    // Data decomposition for output
    rocfft_field outfield = nullptr;
    rocfft_field_create(&outfield);

    std::vector<void*>  gpu_out(2);
    std::vector<size_t> outbrick0_lower = {0, 0, 0, 0};
    std::vector<size_t> outbrick0_upper = {1, length[0] / deviceCount, length[1], length[2]};
    std::vector<size_t> outbrick1_lower = {0, length[0] / deviceCount, 0, 0};
    std::vector<size_t> outbrick1_upper = {1, length[0], length[1], length[2]};

    rocfft_field_add_brick(outfield,
                           outbrick0_lower.data(),
                           outbrick0_upper.data(),
                           brick_stride.data(),
                           outbrick0_lower.size(),
                           devices[0]); // device id

    rocfft_field_add_brick(outfield,
                           outbrick1_lower.data(),
                           outbrick1_upper.data(),
                           brick_stride.data(),
                           outbrick1_lower.size(),
                           devices[1]); // device id
    rc = rocfft_plan_description_add_outfield(description, outfield);

    // Allocate GPU output
    (void)hipSetDevice(devices[0]);
    if(hipMalloc(&gpu_out[0], memSize) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    (void)hipSetDevice(devices[1]);
    if(hipMalloc(&gpu_out[1], memSize) != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Create a multi-gpu plan
    (void)hipSetDevice(devices[0]);
    rocfft_plan gpu_plan = nullptr;
    rc                   = rocfft_plan_create(&gpu_plan,
                            place,
                            direction,
                            rocfft_precision_double,
                            length.size(), // Dimension
                            length.data(), // lengths
                            1, // Number of transforms
                            description); // Description
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create plan");

    // Get execution information and allocate work buffer
    rocfft_execution_info planinfo      = nullptr;
    size_t                work_buf_size = 0;
    if(rocfft_plan_get_work_buffer_size(gpu_plan, &work_buf_size) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");
    void* work_buf = nullptr;

    if(work_buf_size)
    {
        if(rocfft_execution_info_create(&planinfo) != rocfft_status_success)
            throw std::runtime_error("failed to create execution info");
        if(hipMalloc(&work_buf, work_buf_size) != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
        if(rocfft_execution_info_set_work_buffer(planinfo, work_buf, work_buf_size)
           != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_work_buffer failed.");
    }

    // Execute plan
    rc = rocfft_execute(gpu_plan, (void**)gpu_in.data(), (void**)gpu_out.data(), planinfo);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to execute.");

    // Destroy plan
    if(rocfft_execution_info_destroy(planinfo) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_destroy failed.");
    planinfo = nullptr;
    if(rocfft_plan_description_destroy(description) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_destroy failed.");
    description = nullptr;
    if(rocfft_plan_destroy(gpu_plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    gpu_plan = nullptr;

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");

    for(unsigned int i = 0; i < 2; ++i)
    {
        (void)hipFree(gpu_in[i]);
        (void)hipFree(gpu_out[i]);
    }

    return 0;
}
