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
#include <thread>
#include <utility>
#include <vector>

#include "../client_utils.h"
#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

// Given an array type, return the name as a string.
std::string array_type_name(const rocfft_array_type type)
{
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
        return "rocfft_array_type_complex_interleaved";
    case rocfft_array_type_complex_planar:
        return "rocfft_array_type_complex_planar";
    case rocfft_array_type_real:
        return "rocfft_array_type_real";
    case rocfft_array_type_hermitian_interleaved:
        return "rocfft_array_type_hermitian_interleaved";
    case rocfft_array_type_hermitian_planar:
        return "rocfft_array_type_hermitian_planar";
    case rocfft_array_type_unset:
        return "rocfft_array_type_unset";
    }
    return "";
}

// Function to return a string with the gpu parameters.
std::string gpu_params(const std::vector<size_t>&    gpu_ilength_cm,
                       const std::vector<size_t>&    gpu_istride_cm,
                       const size_t                  gpu_idist,
                       const std::vector<size_t>&    gpu_ostride_cm,
                       const size_t                  gpu_odist,
                       const size_t                  nbatch,
                       const rocfft_precision        precision,
                       const rocfft_result_placement place,
                       const rocfft_array_type       itype,
                       const rocfft_array_type       otype)
{
    std::stringstream ss;
    ss << "\nGPU params:\n";
    ss << "\tgpu_ilength_cm:";
    for(auto i : gpu_ilength_cm)
        ss << " " << i;
    ss << "\n";
    ss << "\tgpu_istride_cm:";
    for(auto i : gpu_istride_cm)
        ss << " " << i;
    ss << "\n";
    ss << "\tgpu_idist: " << gpu_idist << "\n";

    ss << "\tgpu_ostride_cm:";
    for(auto i : gpu_ostride_cm)
        ss << " " << i;
    ss << "\n";
    ss << "\tgpu_odist: " << gpu_odist << "\n";

    ss << "\tbatch: " << nbatch << "\n";

    if(place == rocfft_placement_inplace)
        ss << "\tin-place\n";
    else
        ss << "\tout-of-place\n";
    ss << "\t" << array_type_name(itype) << " -> " << array_type_name(otype) << "\n";
    if(precision == rocfft_precision_single)
        ss << "\tsingle-precision\n";
    else
        ss << "\tdouble-precision\n";
    return ss.str();
}

// Compute a FFT using rocFFT and compare with the provided CPU reference computation.
void rocfft_transform(const std::vector<size_t>                                  length,
                      const size_t                                               istride0,
                      const size_t                                               ostride0,
                      const size_t                                               nbatch,
                      const rocfft_precision                                     precision,
                      const rocfft_transform_type                                transformType,
                      const rocfft_array_type                                    itype,
                      const rocfft_array_type                                    otype,
                      const rocfft_result_placement                              place,
                      const std::vector<size_t>&                                 cpu_istride,
                      const std::vector<size_t>&                                 cpu_ostride,
                      const size_t                                               cpu_idist,
                      const size_t                                               cpu_odist,
                      const rocfft_array_type                                    cpu_itype,
                      const rocfft_array_type                                    cpu_otype,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_input_copy,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_output,
                      const std::pair<double, double>& cpu_output_L2Linfnorm,
                      std::thread*                     cpu_output_thread)
{
    // Set up GPU computation:

    if(place == rocfft_placement_inplace)
    {
        if(istride0 != ostride0)
        {
            // In-place transforms require identical input and output strides.
            if(verbose)
            {
                std::cout << "istride0: " << istride0 << " ostride0: " << ostride0
                          << " differ; skipped for in-place transforms: skipping test" << std::endl;
            }
            // TODO: mark skipped
            return;
        }
        if((transformType == rocfft_transform_type_real_forward
            || transformType == rocfft_transform_type_real_inverse)
           && (istride0 != 1 || ostride0 != 1))
        {
            // In-place real/complex transforms require unit strides.
            if(verbose)
            {
                std::cout << "istride0: " << istride0 << " ostride0: " << ostride0
                          << " must be unitary for in-place real/complex transforms: skipping test"
                          << std::endl;
            }
            // TODO: mark skipped
            return;
        }

        if((itype == rocfft_array_type_complex_interleaved
            && otype == rocfft_array_type_complex_planar)
           || (itype == rocfft_array_type_complex_planar
               && otype == rocfft_array_type_complex_interleaved))
        {
            if(verbose)
            {
                std::cout << "In-place c2c transforms require identical io types; skipped.\n";
            }
            return;
        }

        if((itype == rocfft_array_type_real && otype == rocfft_array_type_hermitian_planar)
           || (itype == rocfft_array_type_hermitian_planar && otype == rocfft_array_type_real))
        {
            if(verbose)
            {
                std::cout << "In-place real/complex transforms cannot use planar types; skipped.\n";
            }
            return;
        }
    }

    const size_t dim = length.size();

    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;

    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;

    auto gpu_istride = compute_stride(ilength,
                                      istride0,
                                      place == rocfft_placement_inplace
                                          && transformType == rocfft_transform_type_real_forward);

    auto gpu_ostride = compute_stride(olength,
                                      ostride0,
                                      place == rocfft_placement_inplace
                                          && transformType == rocfft_transform_type_real_inverse);

    const auto gpu_idist = set_idist(place, transformType, length, gpu_istride);
    const auto gpu_odist = set_odist(place, transformType, length, gpu_ostride);

    rocfft_status fft_status = rocfft_status_success;
    // Transform parameters from row-major to column-major for rocFFT:
    auto gpu_length_cm  = length;
    auto gpu_ilength_cm = ilength;
    auto gpu_olength_cm = olength;
    auto gpu_istride_cm = gpu_istride;
    auto gpu_ostride_cm = gpu_ostride;
    for(int idx = 0; idx < dim / 2; ++idx)
    {
        const auto toidx = dim - idx - 1;
        std::swap(gpu_istride_cm[idx], gpu_istride_cm[toidx]);
        std::swap(gpu_ostride_cm[idx], gpu_ostride_cm[toidx]);
        std::swap(gpu_length_cm[idx], gpu_length_cm[toidx]);
        std::swap(gpu_ilength_cm[idx], gpu_ilength_cm[toidx]);
        std::swap(gpu_olength_cm[idx], gpu_olength_cm[toidx]);
    }

    if(verbose > 1)
    {
        std::cout << gpu_params(gpu_ilength_cm,
                                gpu_istride_cm,
                                gpu_idist,
                                gpu_ostride_cm,
                                gpu_odist,
                                nbatch,
                                precision,
                                place,
                                itype,
                                otype);
    }

    // Create FFT description
    rocfft_plan_description desc = NULL;
    fft_status                   = rocfft_plan_description_create(&desc);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";
    const std::vector<size_t> ioffset = {0, 0};
    const std::vector<size_t> ooffset = {0, 0};
    fft_status                        = rocfft_plan_description_set_data_layout(desc,
                                                         itype,
                                                         otype,
                                                         ioffset.data(),
                                                         ooffset.data(),
                                                         gpu_istride_cm.size(),
                                                         gpu_istride_cm.data(),
                                                         gpu_idist,
                                                         gpu_ostride_cm.size(),
                                                         gpu_ostride_cm.data(),
                                                         gpu_odist);
    EXPECT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    // Create the plan
    rocfft_plan gpu_plan = NULL;
    fft_status           = rocfft_plan_create(&gpu_plan,
                                    place,
                                    transformType,
                                    precision,
                                    gpu_length_cm.size(),
                                    gpu_length_cm.data(),
                                    nbatch,
                                    desc);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // Create execution info
    rocfft_execution_info info = NULL;
    fft_status                 = rocfft_execution_info_create(&info);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    EXPECT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";

    // Number of value in input and output variables.
    const size_t isize = nbatch * gpu_idist;
    const size_t osize = nbatch * gpu_odist;

    // Sizes of individual input and output variables
    const size_t isize_t = var_size<size_t>(precision, itype);
    const size_t osize_t = var_size<size_t>(precision, otype);

    // Numbers of input and output buffers:
    const int nibuffer
        = (itype == rocfft_array_type_complex_planar || itype == rocfft_array_type_hermitian_planar)
              ? 2
              : 1;
    const int nobuffer
        = (otype == rocfft_array_type_complex_planar || otype == rocfft_array_type_hermitian_planar)
              ? 2
              : 1;

    // Check if the problem fits on the device; if it doesn't skip it.
    if(!vram_fits_problem(nibuffer * isize * isize_t,
                          (place == rocfft_placement_inplace) ? 0 : nobuffer * osize * osize_t,
                          workbuffersize))
    {
        rocfft_plan_destroy(gpu_plan);
        rocfft_plan_description_destroy(desc);
        rocfft_execution_info_destroy(info);

        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped\n";
        }
        // TODO: mark as skipped via gtest.
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
    auto gpu_input = allocate_host_buffer<fftwAllocator<char>>(
        precision, itype, length, gpu_istride, gpu_idist, nbatch);

    // Copy from contiguous_input to input.
    copy_buffers(cpu_input_copy,
                 gpu_input,
                 ilength,
                 nbatch,
                 precision,
                 cpu_itype,
                 cpu_istride,
                 cpu_idist,
                 itype,
                 gpu_istride,
                 gpu_idist);

    if(verbose > 4)
    {
        std::cout << "GPU input:\n";
        printbuffer(precision, itype, gpu_input, ilength, gpu_istride, nbatch, gpu_idist);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU input:\n";
        printbuffer_flat(precision, itype, gpu_input, gpu_idist);
    }

    // GPU input and output buffers:
    auto                ibuffer_sizes = buffer_sizes(precision, itype, gpu_idist, nbatch);
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        ASSERT_TRUE(hip_status == hipSuccess)
            << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i]
            << gpu_params(gpu_ilength_cm,
                          gpu_istride_cm,
                          gpu_idist,
                          gpu_ostride_cm,
                          gpu_odist,
                          nbatch,
                          precision,
                          place,
                          itype,
                          otype);
        pibuffer[i] = ibuffer[i].data();
    }

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(place == rocfft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = buffer_sizes(precision, otype, gpu_odist, nbatch);
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
            ASSERT_TRUE(hip_status == hipSuccess)
                << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                << gpu_params(gpu_ilength_cm,
                              gpu_istride_cm,
                              gpu_idist,
                              gpu_ostride_cm,
                              gpu_odist,
                              nbatch,
                              precision,
                              place,
                              itype,
                              otype);
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
    auto gpu_output = allocate_host_buffer<fftwAllocator<char>>(
        precision, otype, olength, gpu_ostride, gpu_odist, nbatch);
    for(int idx = 0; idx < gpu_output.size(); ++idx)
    {
        hip_status = hipMemcpy(gpu_output[idx].data(),
                               obuffer->at(idx).data(),
                               gpu_output[idx].size(),
                               hipMemcpyDeviceToHost);
        EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        printbuffer(precision, otype, gpu_output, olength, gpu_ostride, nbatch, gpu_odist);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        printbuffer_flat(precision, otype, gpu_output, gpu_odist);
    }

    // Compute the Linfinity and L2 norm of the GPU output:
    std::pair<double, double> L2LinfnormGPU;
    std::thread               normthread([&]() {
        L2LinfnormGPU
            = LinfL2norm(gpu_output, olength, nbatch, precision, otype, gpu_ostride, gpu_odist);
    });
    if(cpu_output_thread && cpu_output_thread->joinable())
        cpu_output_thread->join();

    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    const double linf_cutoff
        = type_epsilon(precision) * cpu_output_L2Linfnorm.first * log(total_length);
    auto linfl2diff = LinfL2diff(cpu_output,
                                 gpu_output,
                                 olength,
                                 nbatch,
                                 precision,
                                 cpu_otype,
                                 cpu_ostride,
                                 cpu_odist,
                                 otype,
                                 gpu_ostride,
                                 gpu_odist,
                                 linf_failures,
                                 linf_cutoff);
    normthread.join();

    if(verbose > 1)
    {
        std::cout << "GPU output Linf norm: " << L2LinfnormGPU.first << "\n";
        std::cout << "GPU output L2 norm:   " << L2LinfnormGPU.second << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures.begin(), linf_failures.end());
        for(const auto& i : linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(std::isfinite(L2LinfnormGPU.first)) << gpu_params(gpu_ilength_cm,
                                                                  gpu_istride_cm,
                                                                  gpu_idist,
                                                                  gpu_ostride_cm,
                                                                  gpu_odist,
                                                                  nbatch,
                                                                  precision,
                                                                  place,
                                                                  itype,
                                                                  otype);
    EXPECT_TRUE(std::isfinite(L2LinfnormGPU.second)) << gpu_params(gpu_ilength_cm,
                                                                   gpu_istride_cm,
                                                                   gpu_idist,
                                                                   gpu_ostride_cm,
                                                                   gpu_odist,
                                                                   nbatch,
                                                                   precision,
                                                                   place,
                                                                   itype,
                                                                   otype);

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << linfl2diff.first << "\n";
        std::cout << "Linf diff: " << linfl2diff.second << "\n";
    }

    // TODO: handle case where norm is zero?
    EXPECT_TRUE(linfl2diff.first < linf_cutoff)
        << "Linf test failed.  Linf:" << linfl2diff.first << "\tnormalized Linf: "
        << linfl2diff.first / (cpu_output_L2Linfnorm.first * log(total_length))
        << "\tepsilon: " << type_epsilon(precision)
        << gpu_params(gpu_ilength_cm,
                      gpu_istride_cm,
                      gpu_idist,
                      gpu_ostride_cm,
                      gpu_odist,
                      nbatch,
                      precision,
                      place,
                      itype,
                      otype);

    EXPECT_TRUE(linfl2diff.second / (cpu_output_L2Linfnorm.second * sqrt(log(total_length)))
                < type_epsilon(precision))
        << "L2 test failed. L2: " << linfl2diff.second << "\tnormalized L2: "
        << linfl2diff.second / (cpu_output_L2Linfnorm.second * sqrt(log(total_length)))
        << "\tepsilon: " << type_epsilon(precision)
        << gpu_params(gpu_ilength_cm,
                      gpu_istride_cm,
                      gpu_idist,
                      gpu_ostride_cm,
                      gpu_odist,
                      nbatch,
                      precision,
                      place,
                      itype,
                      otype);

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
    const std::vector<size_t>                  length         = std::get<0>(GetParam());
    const std::vector<size_t>                  istride0_range = std::get<1>(GetParam());
    const std::vector<size_t>                  ostride0_range = std::get<2>(GetParam());
    const std::vector<size_t>                  batch_range    = std::get<3>(GetParam());
    const rocfft_precision                     precision      = std::get<4>(GetParam());
    const rocfft_transform_type                transformType  = std::get<5>(GetParam());
    const std::vector<rocfft_result_placement> place_range    = std::get<6>(GetParam());

    // NB: Input data is row-major.

    const size_t dim = length.size();

    // Input cpu parameters:
    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    const auto cpu_istride = compute_stride(ilength, 1);
    const auto cpu_itype   = contiguous_itype(transformType);
    const auto cpu_idist
        = set_idist(rocfft_placement_notinplace, transformType, length, cpu_istride);

    // Output cpu parameters:
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    const auto cpu_ostride = compute_stride(olength, 1);
    const auto cpu_odist
        = set_odist(rocfft_placement_notinplace, transformType, length, cpu_ostride);
    auto cpu_otype = contiguous_otype(transformType);
    if(verbose > 3)
    {
        std::cout << "CPU  params:\n";
        std::cout << "\tilength:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_istride:";
        for(auto i : cpu_istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_idist: " << cpu_idist << std::endl;

        std::cout << "\tolength:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_ostride:";
        for(auto i : cpu_ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_odist: " << cpu_odist << std::endl;
    }

    const size_t nbatch = *std::max_element(batch_range.begin(), batch_range.end());

    // Generate the data:
    auto cpu_input = compute_input<fftwAllocator<char>>(
        precision, cpu_itype, length, cpu_istride, cpu_idist, nbatch);
    auto cpu_input_copy = cpu_input; // copy of input (might get overwritten by FFTW).

    // Compute the Linfinity and L2 norm of the CPU output:
    std::pair<double, double> cpu_input_L2Linfnorm;
    std::thread               cpu_input_L2Linfnorm_thread([&]() {
        cpu_input_L2Linfnorm
            = LinfL2norm(cpu_input, ilength, nbatch, precision, cpu_itype, cpu_istride, cpu_idist);
        if(verbose > 2)
        {
            std::cout << "CPU Input Linf norm:  " << cpu_input_L2Linfnorm.first << "\n";
            std::cout << "CPU Input L2 norm:    " << cpu_input_L2Linfnorm.second << "\n";
        }
    });
    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        printbuffer(precision, cpu_itype, cpu_input, ilength, cpu_istride, nbatch, cpu_idist);
    }

    // FFTW computation
    // NB: FFTW may overwrite input, even for out-of-place transforms.
    decltype(cpu_input)       cpu_output;
    std::pair<double, double> cpu_output_L2Linfnorm;
    std::thread               cpu_output_thread([&]() {
        cpu_output = fftw_via_rocfft(length,
                                     cpu_istride,
                                     cpu_ostride,
                                     nbatch,
                                     cpu_idist,
                                     cpu_odist,
                                     precision,
                                     transformType,
                                     cpu_input);
        // Compute the Linfinity and L2 norm of the CPU output:
        cpu_output_L2Linfnorm
            = LinfL2norm(cpu_output, olength, nbatch, precision, cpu_otype, cpu_ostride, cpu_odist);
        if(verbose > 2)
        {
            std::cout << "CPU Output Linf norm: " << cpu_output_L2Linfnorm.first << "\n";
            std::cout << "CPU Output L2 norm:   " << cpu_output_L2Linfnorm.second << "\n";
        }
        if(verbose > 3)
        {
            std::cout << "CPU output:\n";
            printbuffer(precision, cpu_otype, cpu_output, olength, cpu_ostride, nbatch, cpu_odist);
        }
    });
    // clean up threads if transform throws
    BOOST_SCOPE_EXIT_ALL(&cpu_output_thread, &cpu_input_L2Linfnorm_thread)
    {
        if(cpu_output_thread.joinable())
            cpu_output_thread.join();
        if(cpu_input_L2Linfnorm_thread.joinable())
            cpu_input_L2Linfnorm_thread.join();
    };

    // Set up GPU computations:
    for(const auto nbatch : batch_range)
    {
        for(const auto place : place_range)
        {
            for(const auto iotype : iotypes(transformType, place))
            {
                const rocfft_array_type itype = iotype.first;
                const rocfft_array_type otype = iotype.second;
                for(const auto istride0 : istride0_range)
                {
                    for(const auto ostride0 : ostride0_range)
                    {
                        if(verbose)
                        {
                            print_params(length,
                                         istride0,
                                         ostride0,
                                         nbatch,
                                         place,
                                         precision,
                                         transformType,
                                         itype,
                                         otype);
                        }

                        rocfft_transform(length,
                                         istride0,
                                         ostride0,
                                         nbatch,
                                         precision,
                                         transformType,
                                         itype,
                                         otype,
                                         place,
                                         cpu_istride,
                                         cpu_ostride,
                                         cpu_idist,
                                         cpu_odist,
                                         cpu_itype,
                                         cpu_otype,
                                         cpu_input_copy,
                                         cpu_output,
                                         cpu_output_L2Linfnorm,
                                         &cpu_output_thread);
                    }
                }
            }
        }
    }

    cpu_input_L2Linfnorm_thread.join();
    ASSERT_TRUE(std::isfinite(cpu_input_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(cpu_input_L2Linfnorm.second));

    if(cpu_output_thread.joinable())
        cpu_output_thread.join();
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.second));

    SUCCEED();
}
