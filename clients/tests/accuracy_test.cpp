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

struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
    // base address of input, to ensure that each callback gets an offset from that base
    void* base;
};

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata load_callback(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    if(input == testdata->base)
        return input[offset] * testdata->scalar;
    // wrong base address passed, return something obviously wrong
    else
        return input[0];
}
__device__ auto load_callback_dev_float   = load_callback<float>;
__device__ auto load_callback_dev_float2  = load_callback<float2>;
__device__ auto load_callback_dev_double  = load_callback<double>;
__device__ auto load_callback_dev_double2 = load_callback<double2>;

void* get_load_callback_host(rocfft_array_type itype, rocfft_precision precision)
{
    void* load_callback_host = nullptr;
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case rocfft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_float2, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        case rocfft_precision_double:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_double2, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        }
    }
    case rocfft_array_type_real:
    {
        switch(precision)
        {
        case rocfft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_float, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        case rocfft_precision_double:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_double, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return load_callback_host;
    }
}

template <typename Tdata>
__host__ __device__ void
    store_callback(Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    if(output == testdata->base)
        output[offset] = element + testdata->scalar;
    // otherwise, wrong base address passed, just don't write
}
__device__ auto store_callback_dev_float   = store_callback<float>;
__device__ auto store_callback_dev_float2  = store_callback<float2>;
__device__ auto store_callback_dev_double  = store_callback<double>;
__device__ auto store_callback_dev_double2 = store_callback<double2>;

void* get_store_callback_host(rocfft_array_type otype, rocfft_precision precision)
{
    void* store_callback_host = nullptr;
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case rocfft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&store_callback_host, store_callback_dev_float2, sizeof(void*)),
                hipSuccess);
            return store_callback_host;
        case rocfft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &store_callback_host, store_callback_dev_double2, sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    case rocfft_array_type_real:
    {
        switch(precision)
        {
        case rocfft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&store_callback_host, store_callback_dev_float, sizeof(void*)),
                hipSuccess);
            return store_callback_host;
        case rocfft_precision_double:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&store_callback_host, store_callback_dev_double, sizeof(void*)),
                hipSuccess);
            return store_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return store_callback_host;
    }
}

// apply load callback if necessary
void apply_load_callback(const rocfft_params& params, fftw_data_t& input)
{
    if(!params.run_callbacks)
        return;
    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.load_cb_scalar;
    cbdata.base   = input.front().data();

    switch(params.itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case rocfft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float2*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        case rocfft_precision_double:
        {
            const size_t elem_size = 2 * sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double2*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    case rocfft_array_type_real:
    {
        switch(params.precision)
        {
        case rocfft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        case rocfft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

// apply store callback if necessary
void apply_store_callback(const rocfft_params& params, fftw_data_t& output)
{
    if(!params.run_callbacks)
        return;

    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.store_cb_scalar;
    cbdata.base   = output.front().data();

    switch(params.otype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case rocfft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float2*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case rocfft_precision_double:
        {
            const size_t elem_size = 2 * sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double2*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    case rocfft_array_type_real:
    {
        switch(params.precision)
        {
        case rocfft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case rocfft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

accuracy_test::cpu_fft_params accuracy_test::compute_cpu_fft(const rocfft_params& params)
{
    // Check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  Smaller batch runs can
    // compare against the larger data.
    if(std::get<0>(last_cpu_fft) == params.length
       && std::get<2>(last_cpu_fft) == params.transform_type
       && std::get<3>(last_cpu_fft) == params.run_callbacks)
    {
        if(std::get<1>(last_cpu_fft) >= params.nbatch)
        {
            auto& ret = std::get<4>(last_cpu_fft);
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
                    std::shared_future<fftw_data_t> input_future
                        = std::async(std::launch::async, [=, input{std::move(input)}]() mutable {
                              return convert_to_single(input);
                          });
                    std::shared_future<fftw_data_t> output_future
                        = std::async(std::launch::async, [=, output{std::move(output)}]() mutable {
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
        // If the last result has a smaller batch than the new
        // params, that might be a developer error - tests should be
        // ordered to generate the bigger batch first.  But if tests
        // got filtered or skipped due to insufficient memory, we
        // might never have tried to generate the bigger batch first.
        // So just fall through and redo the CPU FFT.
    }

    // Use steal+swap-with-empty idiom to deallocate previous input/output
    auto last_ret = std::get<4>(last_cpu_fft);
    if(last_ret.input.valid())
    {
        fftw_data_t z;
        z.swap(const_cast<fftw_data_t&>(last_ret.input.get()));
    }
    if(last_ret.output.valid())
    {
        fftw_data_t z;
        z.swap(const_cast<fftw_data_t&>(last_ret.output.get()));
    }

    rocfft_params contiguous_params;
    contiguous_params.length         = params.length;
    contiguous_params.precision      = params.precision;
    contiguous_params.placement      = rocfft_placement_notinplace;
    contiguous_params.transform_type = params.transform_type;
    contiguous_params.nbatch         = params.nbatch;
    contiguous_params.itype          = contiguous_itype(params.transform_type);
    contiguous_params.otype          = contiguous_otype(contiguous_params.transform_type);

    contiguous_params.validate();

    if(!contiguous_params.valid(verbose))
    {
        throw std::runtime_error("Invalid contiguous params");
    }

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

        apply_load_callback(params, input_copy);
        auto output = fftw_via_rocfft(contiguous_params.length,
                                      contiguous_params.istride,
                                      contiguous_params.ostride,
                                      contiguous_params.nbatch,
                                      contiguous_params.idist,
                                      contiguous_params.odist,
                                      contiguous_params.precision,
                                      contiguous_params.transform_type,
                                      input_copy);
        apply_store_callback(params, output);
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
        return output;
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
    std::get<3>(last_cpu_fft) = params.run_callbacks;
    std::get<4>(last_cpu_fft) = ret;

    return ret;
}

// Compute a FFT using rocFFT and compare with the provided CPU reference computation.
void rocfft_transform(const rocfft_params&                 params,
                      const accuracy_test::cpu_fft_params& cpu,
                      const size_t                         ramgb)
{
    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    if(ramgb > 0 && needed_ram(params, verbose) > ramgb * ONE_GiB)
    {
        if(verbose)
        {
            std::cout << "Problem exceeds memory limit; skipped [rocfft_transform]." << std::endl;
        }
        GTEST_SKIP();
        return;
    }

    rocfft_status fft_status = rocfft_status_success;

    // Create FFT description
    rocfft_plan_description desc = NULL;
    fft_status                   = rocfft_plan_description_create(&desc);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";
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
    ASSERT_TRUE(fft_status == rocfft_status_success)
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
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // Create execution info
    rocfft_execution_info info = NULL;
    fft_status                 = rocfft_execution_info_create(&info);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";

    // Check if the problem fits on the device; if it doesn't skip it.
    auto       ibuffer_sizes = params.ibuffer_sizes();
    auto       obuffer_sizes = params.obuffer_sizes();
    const auto vram_footprint
        = params.nibuffer() * ibuffer_sizes[0]
          + (params.placement == rocfft_placement_inplace ? 0
                                                          : params.nobuffer() * obuffer_sizes[0])
          + workbuffersize;
    if(verbose > 1)
    {
        size_t     free   = 0;
        size_t     total  = 0;
        hipError_t retval = hipMemGetInfo(&free, &total);
        if(retval != hipSuccess)
        {
            std::cerr << "hipMemGetInfo failed" << std::endl;
        }
        std::cout << "vram footprint: " << vram_footprint << " (" << (double)vram_footprint
                  << ") workbuffer: " << workbuffersize << " (" << (double)workbuffersize
                  << ") free: " << free << " (" << (double)free << ") total: " << total << " ("
                  << (double)total << ")\n";
    }
    if(!vram_fits_problem(vram_footprint))
    {
        rocfft_plan_destroy(gpu_plan);
        rocfft_plan_description_destroy(desc);
        rocfft_execution_info_destroy(info);

        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped [rocfft_transform]." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << vram_footprint << ") too large for device";
        return;
    }

    hipError_t hip_status = hipSuccess;

    // Allocate work memory and associate with the execution info
    gpubuf wbuffer;
    if(workbuffersize > 0)
    {
        hip_status = wbuffer.alloc(workbuffersize);
        if(hip_status != hipSuccess)
        {

            size_t     free   = 0;
            size_t     total  = 0;
            hipError_t retval = hipMemGetInfo(&free, &total);
            if(retval == hipSuccess)
            {
                std::cerr << "free vram: " << free << " total vram: " << total << std::endl;
            }
            else
            {
                std::cerr << "hipMemGetInfo also failed" << std::endl;
            }
        }
        ASSERT_TRUE(hip_status == hipSuccess)
            << "hipMalloc failure for work buffer of size " << workbuffersize << std::endl;

        fft_status = rocfft_execution_info_set_work_buffer(info, wbuffer.data(), workbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
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
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            size_t     free   = 0;
            size_t     total  = 0;
            hipError_t retval = hipMemGetInfo(&free, &total);
            if(retval != hipSuccess)
            {
                std::cerr << "hipMemGetInfo also failed" << std::endl;
            }
        }
        ASSERT_TRUE(hip_status == hipSuccess)
            << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i] << "\n"
            << params.str();
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
            if(hip_status != hipSuccess)
            {
                size_t     free   = 0;
                size_t     total  = 0;
                hipError_t retval = hipMemGetInfo(&free, &total);
                if(retval == hipSuccess)
                {
                    std::cerr << "free vram: " << free << " (" << (double)free
                              << ") total vram: " << total << " (" << (double)total << ")"
                              << std::endl;
                    if(free > obuffer_sizes[i])
                    {
                        std::cerr << "The system reports that there is enough space." << std::endl;
                    }
                }
                else
                {
                    std::cerr << "hipMemGetInfo also failed" << std::endl;
                }
            }
            ASSERT_TRUE(hip_status == hipSuccess)
                << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                << " (" << (double)obuffer_sizes[i] << ")\n"
                << params.str();
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    gpubuf_t<callback_test_data> load_cb_data_dev;
    gpubuf_t<callback_test_data> store_cb_data_dev;
    if(params.run_callbacks)
    {
        void* load_cb_host = get_load_callback_host(params.itype, params.precision);

        callback_test_data load_cb_data_host;
        load_cb_data_host.scalar = params.load_cb_scalar;
        load_cb_data_host.base   = ibuffer.front().data();

        ASSERT_TRUE(hipSuccess == load_cb_data_dev.alloc(sizeof(callback_test_data)));
        ASSERT_TRUE(hipSuccess
                    == hipMemcpy(load_cb_data_dev.data(),
                                 &load_cb_data_host,
                                 sizeof(callback_test_data),
                                 hipMemcpyHostToDevice));

        void* load_cb_data = load_cb_data_dev.data();
        fft_status = rocfft_execution_info_set_load_callback(info, &load_cb_host, &load_cb_data, 0);
        ASSERT_TRUE(fft_status == rocfft_status_success);

        void* store_cb_host = get_store_callback_host(params.otype, params.precision);

        callback_test_data store_cb_data_host;
        store_cb_data_host.scalar = params.store_cb_scalar;
        store_cb_data_host.base   = obuffer->front().data();

        ASSERT_TRUE(hipSuccess == store_cb_data_dev.alloc(sizeof(callback_test_data)));
        ASSERT_TRUE(hipSuccess
                    == hipMemcpy(store_cb_data_dev.data(),
                                 &store_cb_data_host,
                                 sizeof(callback_test_data),
                                 hipMemcpyHostToDevice));

        void* store_cb_data = store_cb_data_dev.data();
        fft_status
            = rocfft_execution_info_set_store_callback(info, &store_cb_host, &store_cb_data, 0);
        ASSERT_TRUE(fft_status == rocfft_status_success);
    }

    // Copy the input data to the GPU:
    for(int idx = 0; idx < gpu_input.size(); ++idx)
    {
        hip_status = hipMemcpy(ibuffer[idx].data(),
                               gpu_input[idx].data(),
                               gpu_input[idx].size(),
                               hipMemcpyHostToDevice);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    // Execute the transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)pibuffer.data(), // in buffers
                                (void**)pobuffer.data(), // out buffers
                                info); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

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
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
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
    const auto                             total_length = std::accumulate(params.length.begin(),
                                              params.length.end(),
                                              static_cast<size_t>(1),
                                              std::multiplies<size_t>());
    const double                           linf_cutoff
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

    params.validate();

    if(!params.valid(verbose))
    {
        if(verbose)
        {
            std::cout << "Invalid parameters, skip this test." << std::endl;
        }
        GTEST_SKIP();
    }

    if(ramgb > 0 && needed_ram(params, verbose) > ramgb * ONE_GiB)
    {
        if(verbose)
        {
            std::cout << "Problem exceeds memory limit; skipped [accuracy_test]." << std::endl;
        }
        GTEST_SKIP();
        return;
    }

    // Test also if the smalles problem size can fit in vram
    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();
    auto minimal_vram_footprint
        = params.nibuffer() * ibuffer_sizes[0]
          + (params.placement == rocfft_placement_inplace ? 0
                                                          : params.nobuffer() * obuffer_sizes[0]);
    size_t free   = 0;
    size_t total  = 0;
    auto   retval = hipMemGetInfo(&free, &total);
    ASSERT_TRUE(retval == hipSuccess);
    if(total < minimal_vram_footprint)
    {
        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped [accuracy_test]." << std::endl;
        }
        GTEST_SKIP() << "won't fit on device, even without work buffer";
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
