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

#include <boost/scope_exit.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../shared/gpubuf.h"
#include "../client_utils.h"
#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

accuracy_test::cpu_fft_params accuracy_test::compute_cpu_fft(const rocfft_params& params)
{
    // Check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  Smaller batch runs can
    // compare against the larger data.
    if(std::get<0>(last_cpu_fft) == params.length
       && std::get<2>(last_cpu_fft) == params.transform_type)
    {
        if(std::get<1>(last_cpu_fft) >= params.nbatch)
        {
            auto& ret = std::get<3>(last_cpu_fft);
            if(params.precision != ret.precision)
            {
                // Tests should be ordered so we do double first, then float.
                if(ret.precision == rocfft_precision_double)
                {
                    // convert double input/output to float in-place so
                    // we don't need extra memory
                    auto convert_to_single = [](fftw_data_t& data) {
                        for(auto& arr : data)
                        {
                            const double* readPtr  = reinterpret_cast<const double*>(arr.data());
                            const double* readEnd  = readPtr + (arr.size() / sizeof(double));
                            float*        writePtr = reinterpret_cast<float*>(arr.data());
                            std::copy(readPtr, readEnd, writePtr);
                            arr.resize(arr.size() / 2);
                        }
                        return data;
                    };
                    // HACK: the future only gives us const
                    // fftw_data.  But at this point we're in between
                    // test cases and can safely mutate the data to
                    // convert it.
                    //
                    // Steal the data from the future
                    fftw_data_t input;
                    fftw_data_t output;
                    input.swap(const_cast<fftw_data_t&>(ret.input.get()));
                    output.swap(const_cast<fftw_data_t&>(ret.output.get()));
                    // async convert the input/output to single-precision
                    std::shared_future<fftw_data_t> input_future = std::async(
                        std::launch::async, [=, &ret, input{std::move(input)}]() mutable {
                            return convert_to_single(input);
                        });
                    std::shared_future<fftw_data_t> output_future = std::async(
                        std::launch::async, [=, &ret, output{std::move(output)}]() mutable {
                            return convert_to_single(output);
                        });
                    // replace the cached futures with these conversions
                    ret.input     = std::move(input_future);
                    ret.output    = std::move(output_future);
                    ret.precision = rocfft_precision_single;
                }
                else
                {
                    // Somehow we've done float first, then double?
                    // Tests are ordered wrong, and we don't want to
                    // lose precision
                    std::cerr << "Can't do float first then double: aborting." << std::endl;
                    abort();
                }
            }
            return ret;
        }
        else
            // Something's unexpected with our test order - we should have
            // generated the bigger batch first.  Batch ranges provided to
            // the test suites need to be in descending order.
            abort();
    }

    rocfft_params contiguous_params;
    contiguous_params.length         = params.length;
    contiguous_params.precision      = params.precision;
    contiguous_params.placement      = rocfft_placement_notinplace;
    contiguous_params.transform_type = params.transform_type;
    contiguous_params.nbatch         = params.nbatch;

    // Input cpu parameters:
    contiguous_params.istride = compute_stride(contiguous_params.ilength());
    contiguous_params.itype   = contiguous_itype(params.transform_type);
    contiguous_params.idist   = set_idist(rocfft_placement_notinplace,
                                        contiguous_params.transform_type,
                                        contiguous_params.length,
                                        contiguous_params.istride);
    contiguous_params.isize.push_back(contiguous_params.idist * contiguous_params.nbatch);

    // Output cpu parameters:
    contiguous_params.ostride = compute_stride(contiguous_params.olength());
    contiguous_params.odist   = set_odist(rocfft_placement_notinplace,
                                        contiguous_params.transform_type,
                                        contiguous_params.length,
                                        contiguous_params.ostride);
    contiguous_params.otype   = contiguous_otype(contiguous_params.transform_type);
    contiguous_params.osize.push_back(contiguous_params.odist * contiguous_params.nbatch);

    if(verbose > 3)
    {
        std::cout << "CPU  params:\n";
        std::cout << contiguous_params.str("\n\t") << std::endl;
    }

    // Hook up the futures
    std::shared_future<fftw_data_t> input = std::async(std::launch::async, [=]() {
        return compute_input<fftwAllocator<char>>(contiguous_params);
    });

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        printbuffer(params.precision,
                    contiguous_params.itype,
                    input.get(),
                    params.ilength(),
                    contiguous_params.istride,
                    params.nbatch,
                    contiguous_params.idist,
                    contiguous_params.ioffset);
    }

    auto input_norm = std::async(std::launch::async, [=]() {
        auto ret_norm = norm(input.get(),
                             contiguous_params.ilength(),
                             contiguous_params.nbatch,
                             contiguous_params.precision,
                             contiguous_params.itype,
                             contiguous_params.istride,
                             contiguous_params.idist,
                             contiguous_params.ioffset);
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
        auto output     = fftw_via_rocfft(contiguous_params.length,
                                      contiguous_params.istride,
                                      contiguous_params.ostride,
                                      contiguous_params.nbatch,
                                      contiguous_params.idist,
                                      contiguous_params.odist,
                                      contiguous_params.precision,
                                      contiguous_params.transform_type,
                                      input_copy);
        if(verbose > 3)
        {
            std::cout << "CPU output:\n";
            printbuffer(params.precision,
                        contiguous_params.otype,
                        output,
                        params.olength(),
                        contiguous_params.ostride,
                        params.nbatch,
                        contiguous_params.odist,
                        contiguous_params.ooffset);
        }
        return std::move(output);
    });
    std::shared_future<VectorNorms> output_norm = std::async(std::launch::async, [=]() {
        auto ret_norm = norm(output.get(),
                             params.olength(),
                             params.nbatch,
                             params.precision,
                             contiguous_params.otype,
                             contiguous_params.ostride,
                             contiguous_params.odist,
                             contiguous_params.ooffset);
        if(verbose > 2)
        {
            std::cout << "CPU Output Linf norm: " << ret_norm.l_inf << "\n";
            std::cout << "CPU Output L2 norm:   " << ret_norm.l_2 << "\n";
        }
        return ret_norm;
    });

    cpu_fft_params ret;
    ret.ilength   = params.ilength();
    ret.istride   = contiguous_params.istride;
    ret.itype     = contiguous_params.itype;
    ret.idist     = contiguous_params.idist;
    ret.precision = params.precision;
    ret.olength   = params.olength();
    ret.ostride   = contiguous_params.ostride;
    ret.otype     = contiguous_params.otype;
    ret.odist     = contiguous_params.odist;

    ret.input       = std::move(input);
    ret.input_norm  = std::move(input_norm);
    ret.output      = std::move(output);
    ret.output_norm = std::move(output_norm);

    // Cache our result
    std::get<0>(last_cpu_fft) = params.length;
    std::get<1>(last_cpu_fft) = params.nbatch;
    std::get<2>(last_cpu_fft) = params.transform_type;
    std::get<3>(last_cpu_fft) = ret;

    return std::move(ret);
}

// Compute a FFT using rocFFT and compare with the provided CPU reference computation.
void rocfft_transform(const rocfft_params&                 params,
                      const accuracy_test::cpu_fft_params& cpu,
                      const size_t                         ramgb)
{
    if(ramgb > 0 && params.needed_ram(verbose) > ramgb * 1e9)
    {
        if(verbose > 2)
        {
            std::cout << "skipped!" << std::endl;
        }
        return;
    }

    if(!params.valid(verbose))
    {
        // Invalid parameters; skip this test.
        return;
    }

    const size_t dim = params.length.size();

    rocfft_status fft_status = rocfft_status_success;

    // Create FFT description
    rocfft_plan_description desc = NULL;
    fft_status                   = rocfft_plan_description_create(&desc);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";
    fft_status = rocfft_plan_description_set_data_layout(desc,
                                                         params.itype,
                                                         params.otype,
                                                         params.ioffset.data(),
                                                         params.ooffset.data(),
                                                         params.istride_cm().size(),
                                                         params.istride_cm().data(),
                                                         params.idist,
                                                         params.ostride_cm().size(),
                                                         params.ostride_cm().data(),
                                                         params.odist);
    EXPECT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    // Create the plan
    rocfft_plan gpu_plan = NULL;
    fft_status           = rocfft_plan_create(&gpu_plan,
                                    params.placement,
                                    params.transform_type,
                                    params.precision,
                                    params.length_cm().size(),
                                    params.length_cm().data(),
                                    params.nbatch,
                                    desc);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // Create execution info
    rocfft_execution_info info = NULL;
    fft_status                 = rocfft_execution_info_create(&info);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";

    // Sizes of individual input and output variables
    const size_t isize_t = var_size<size_t>(params.precision, params.itype);
    const size_t osize_t = var_size<size_t>(params.precision, params.otype);

    // Numbers of input and output buffers:
    const int nibuffer = params.nibuffer();
    const int nobuffer = params.nobuffer();

    // Check if the problem fits on the device; if it doesn't skip it.
    if(!vram_fits_problem(std::accumulate(params.isize.begin(), params.isize.end(), 0) * isize_t,
                          (params.placement == rocfft_placement_inplace)
                              ? 0
                              : std::accumulate(params.osize.begin(), params.osize.end(), 0)
                                    * osize_t,
                          workbuffersize))
    {
        rocfft_plan_destroy(gpu_plan);
        rocfft_plan_description_destroy(desc);
        rocfft_execution_info_destroy(info);

        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped\n";
        }
        return;
    }

    hipError_t hip_status = hipSuccess;

    // Allocate work memory and associate with the execution info
    gpubuf wbuffer;
    if(workbuffersize > 0)
    {
        hip_status = wbuffer.alloc(workbuffersize);
        EXPECT_TRUE(hip_status == hipSuccess) << "hipMalloc failure for work buffer";
        fft_status = rocfft_execution_info_set_work_buffer(info, wbuffer.data(), workbuffersize);
        EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }

    // Formatted input data:
    auto gpu_input
        = allocate_host_buffer<fftwAllocator<char>>(params.precision, params.itype, params.isize);

    // Copy from contiguous_input to input.
    copy_buffers(cpu.input.get(),
                 gpu_input,
                 params.ilength(),
                 params.nbatch,
                 params.precision,
                 cpu.itype,
                 cpu.istride,
                 cpu.idist,
                 params.itype,
                 params.istride,
                 params.idist,
                 {0},
                 params.ioffset);

    if(verbose > 4)
    {
        std::cout << "GPU input:\n";
        printbuffer(params.precision,
                    params.itype,
                    gpu_input,
                    params.ilength(),
                    params.istride,
                    params.nbatch,
                    params.idist,
                    params.ioffset);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU input:\n";
        printbuffer_flat(params.precision, params.itype, gpu_input, params.isize, params.ioffset);
    }

    // GPU input and output buffers:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure for input buffer " << i
                                              << " size " << ibuffer_sizes[i] << params.str();
        pibuffer[i] = ibuffer[i].data();
    }

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(params.placement == rocfft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
            ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure for output buffer " << i
                                                  << " size " << obuffer_sizes[i] << params.str();
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // Copy the input data to the GPU:
    for(int idx = 0; idx < gpu_input.size(); ++idx)
    {
        hip_status = hipMemcpy(ibuffer[idx].data(),
                               gpu_input[idx].data(),
                               gpu_input[idx].size(),
                               hipMemcpyHostToDevice);
        EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    // Execute the transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)pibuffer.data(), // in buffers
                                (void**)pobuffer.data(), // out buffers
                                info); // execution info
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Copy the data back to the host:
    ASSERT_TRUE(!params.osize.empty()) << "Error: params osize is empty";
    auto gpu_output
        = allocate_host_buffer<fftwAllocator<char>>(params.precision, params.otype, params.osize);
    ASSERT_TRUE(gpu_output.size() > 0) << "Error: no output buffers.";
    for(int idx = 0; idx < gpu_output.size(); ++idx)
    {
        ASSERT_TRUE(gpu_output[idx].size() > 0)
            << "Error: output buffer index " << idx << " size is 0.";
        hip_status = hipMemcpy(gpu_output[idx].data(),
                               obuffer->at(idx).data(),
                               gpu_output[idx].size(),
                               hipMemcpyDeviceToHost);
        EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        printbuffer(params.precision,
                    params.otype,
                    gpu_output,
                    params.olength(),
                    params.ostride,
                    params.nbatch,
                    params.odist,
                    params.ooffset);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        printbuffer_flat(params.precision, params.otype, gpu_output, params.osize, params.ooffset);
    }

    // Compute the Linfinity and L2 norm of the GPU output:
    std::shared_future<VectorNorms> gpu_norm = std::async(std::launch::async, [&]() {
        return norm(gpu_output,
                    params.olength(),
                    params.nbatch,
                    params.precision,
                    params.otype,
                    params.ostride,
                    params.odist,
                    params.ooffset);
    });

    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length
        = std::accumulate(params.length.begin(), params.length.end(), 1, std::multiplies<size_t>());
    const double linf_cutoff
        = type_epsilon(params.precision) * cpu.output_norm.get().l_inf * log(total_length);
    auto diff = distance(cpu.output.get(),
                         gpu_output,
                         params.olength(),
                         params.nbatch,
                         params.precision,
                         cpu.otype,
                         cpu.ostride,
                         cpu.odist,
                         params.otype,
                         params.ostride,
                         params.odist,
                         linf_failures,
                         linf_cutoff,
                         {0},
                         params.ooffset);

    if(verbose > 1)
    {
        std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures.begin(), linf_failures.end());
        for(const auto& i : linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    // TODO: handle case where norm is zero?
    EXPECT_TRUE(diff.l_inf < linf_cutoff)
        << "Linf test failed.  Linf:" << diff.l_inf
        << "\tnormalized Linf: " << diff.l_inf / cpu.output_norm.get().l_inf
        << "\tcutoff: " << linf_cutoff << params.str();

    EXPECT_TRUE(diff.l_2 / cpu.output_norm.get().l_2
                < sqrt(log2(total_length)) * type_epsilon(params.precision))
        << "L2 test failed. L2: " << diff.l_2
        << "\tnormalized L2: " << diff.l_2 / cpu.output_norm.get().l_2
        << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
        << params.str();

    rocfft_plan_destroy(gpu_plan);
    gpu_plan = NULL;
    rocfft_plan_description_destroy(desc);
    desc = NULL;
    rocfft_execution_info_destroy(info);
    info = NULL;
}

// Test for comparison between FFTW and rocFFT.
TEST_P(accuracy_test, vs_fftw)
{
    rocfft_params params = GetParam();

    params.istride
        = compute_stride(params.ilength(),
                         params.istride,
                         params.placement == rocfft_placement_inplace
                             && params.transform_type == rocfft_transform_type_real_forward);
    params.ostride
        = compute_stride(params.olength(),
                         params.ostride,
                         params.placement == rocfft_placement_inplace
                             && params.transform_type == rocfft_transform_type_real_inverse);

    if(params.idist == 0)
    {
        params.idist
            = set_idist(params.placement, params.transform_type, params.length, params.istride);
    }
    if(params.odist == 0)
    {
        params.odist
            = set_odist(params.placement, params.transform_type, params.length, params.ostride);
    }

    if(params.isize.empty())
    {
        params.isize = params.compute_isize();
    }
    if(params.osize.empty())
    {
        params.osize = params.compute_osize();
    }

    if(ramgb > 0)
    {
        // Estimate the amount of memory needed, and skip if it's more than we allow.

        // Host input, output, and input copy, gpu input and output: 5 buffers.
        // This test assumes that all buffers are contiguous; other cases are dealt with when they
        // are called.
        // FFTW may require work memory; this is not accounted for.
        size_t needed_ram
            = 5
              * std::accumulate(
                  params.length.begin(), params.length.end(), 1, std::multiplies<size_t>());

        // Account for precision and data type:
        if(params.transform_type != rocfft_transform_type_real_forward
           || params.transform_type != rocfft_transform_type_real_inverse)
        {
            needed_ram *= 2;
        }
        switch(params.precision)
        {
        case rocfft_precision_single:
            needed_ram *= sizeof(float);
            break;
        case rocfft_precision_double:
            needed_ram *= sizeof(double);
            break;
        }

        if(needed_ram > ramgb * 1e9)
        {
            GTEST_SKIP();
            return;
        }
    }
    auto cpu = accuracy_test::compute_cpu_fft(params);

    // Set up GPU computations:
    if(verbose)
    {
        std::cout << "\t" << params.str("\n\t") << std::endl;
    }

    rocfft_transform(params, cpu, ramgb);

    auto cpu_input_norm = cpu.input_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_input_norm.l_inf));

    auto cpu_output_norm = cpu.output_norm.get();
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));

    SUCCEED();
}
