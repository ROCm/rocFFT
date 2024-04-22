// Copyright (C) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "bench.h"
#include "rocfft/rocfft.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples
    int ntrial{};

    // FFT parameters:
    rocfft_params params;

    // Token string to fully specify fft params.
    std::string token;

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("rocfft-bench command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
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
         "Logical size of output buffer.")
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

    if(vm.count("version"))
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "version " << v << std::endl;
        return EXIT_SUCCESS;
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

    rocfft_setup();

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
        std::cout << params.str(" ") << std::endl;
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

    auto ret = params.create_plan();
    if(ret != fft_status_success)
        LIB_V_THROW(rocfft_status_failure, "Plan creation failed");

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

    // Scatter input out to other devices and adjust I/O buffers to match requested transform
    params.multi_gpu_prepare(ibuffer, pibuffer, pobuffer);

    // Execute a warm-up call
    params.execute(pibuffer.data(), pobuffer.data());

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_wrapper_t start, stop;
    start.alloc();
    stop.alloc();
    for(unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        // Create input at every iteration to avoid overflow
        if(params.ifields.empty())
        {
            // Compute input on default device
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

            // Scatter input out to other devices if this is a multi-GPU test
            params.multi_gpu_prepare(ibuffer, pibuffer, pobuffer);
        }

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        params.execute(pibuffer.data(), pobuffer.data());

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        HIP_V_THROW(hipEventElapsedTime(&time, start, stop), "hipEventElapsedTime failed");
        gpu_time[itrial] = time;

        // Print result after FFT transform
        if(verbose > 2)
        {
            // Gather data to default GPU if this is a multi-GPU test
            params.multi_gpu_finalize(*obuffer, pobuffer);

            auto output = allocate_host_buffer(params.precision, params.otype, params.osize);
            for(unsigned int idx = 0; idx < output.size(); ++idx)
            {
                HIP_V_THROW(hipMemcpy(output[idx].data(),
                                      pobuffer.at(idx),
                                      output[idx].size(),
                                      hipMemcpyDeviceToHost),
                            "hipMemcpy failed");
            }
            std::cout << "GPU output:\n";
            params.print_obuffer(output);
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
        = std::accumulate(params.length.begin(), params.length.end(), 1, std::multiplies<size_t>());
    const double k
        = ((params.itype == fft_array_type_real) || (params.otype == fft_array_type_real)) ? 2.5
                                                                                           : 5.0;
    const double opscount = (double)params.nbatch * k * totsize * log(totsize) / log(2.0);
    for(const auto& i : gpu_time)
    {
        std::cout << " " << opscount / (1e6 * i);
    }
    std::cout << std::endl;

    rocfft_cleanup();
}
