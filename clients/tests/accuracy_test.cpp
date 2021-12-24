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
#include "../rocfft_params.h"
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

void* get_load_callback_host(fft_array_type itype, fft_precision precision)
{
    void* load_callback_host = nullptr;
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_float2, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        case fft_precision_double:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_double2, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&load_callback_host, load_callback_dev_float, sizeof(void*)),
                hipSuccess);
            return load_callback_host;
        case fft_precision_double:
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

void* get_store_callback_host(fft_array_type otype, fft_precision precision)
{
    void* store_callback_host = nullptr;
    switch(otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&store_callback_host, store_callback_dev_float2, sizeof(void*)),
                hipSuccess);
            return store_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &store_callback_host, store_callback_dev_double2, sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_single:
            EXPECT_EQ(
                hipMemcpyFromSymbol(&store_callback_host, store_callback_dev_float, sizeof(void*)),
                hipSuccess);
            return store_callback_host;
        case fft_precision_double:
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
void apply_load_callback(const fft_params& params, fftw_data_t& input)
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
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float2*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
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
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
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
void apply_store_callback(const fft_params& params, fftw_data_t& output)
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
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = 2 * sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float2*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
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
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
                store_callback(output_begin, i, output_begin[i], &cbdata, nullptr);
            break;
        }
        case fft_precision_double:
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

// Test for comparison between FFTW and rocFFT.
TEST_P(accuracy_test, vs_fftw)
{
    rocfft_params params(GetParam());

    params.validate();

    if(!params.valid(verbose))
    {
        if(verbose)
        {
            std::cout << "Invalid parameters, skip this test." << std::endl;
        }
        GTEST_SKIP();
    }

    fft_vs_reference(params);
    SUCCEED();
}

// execute the GPU transform
static void execute_gpu_fft(rocfft_params&      params,
                            std::vector<void*>& pibuffer,
                            std::vector<void*>& pobuffer,
                            fftw_data_t&        gpu_output)
{
    gpubuf_t<callback_test_data> load_cb_data_dev;
    gpubuf_t<callback_test_data> store_cb_data_dev;
    if(params.run_callbacks)
    {
        void* load_cb_host = get_load_callback_host(params.itype, params.precision);

        callback_test_data load_cb_data_host;
        load_cb_data_host.scalar = params.load_cb_scalar;
        load_cb_data_host.base   = pibuffer.front();

        ASSERT_TRUE(hipSuccess == load_cb_data_dev.alloc(sizeof(callback_test_data)));
        ASSERT_TRUE(hipSuccess
                    == hipMemcpy(load_cb_data_dev.data(),
                                 &load_cb_data_host,
                                 sizeof(callback_test_data),
                                 hipMemcpyHostToDevice));

        void* store_cb_host = get_store_callback_host(params.otype, params.precision);

        callback_test_data store_cb_data_host;
        store_cb_data_host.scalar = params.store_cb_scalar;
        store_cb_data_host.base   = pobuffer.front();

        ASSERT_TRUE(hipSuccess == store_cb_data_dev.alloc(sizeof(callback_test_data)));
        ASSERT_TRUE(hipSuccess
                    == hipMemcpy(store_cb_data_dev.data(),
                                 &store_cb_data_host,
                                 sizeof(callback_test_data),
                                 hipMemcpyHostToDevice));
        auto fft_status = params.set_callbacks(
            load_cb_host, load_cb_data_dev.data(), store_cb_host, store_cb_data_dev.data());
        if(fft_status != fft_status_success)
            throw std::runtime_error("set callback failure");
    }

    // Execute the transform:
    auto fft_status = params.execute(pibuffer.data(), pobuffer.data());
    if(fft_status != fft_status_success)
        throw std::runtime_error("rocFFT plan execution failure");

    // copy GPU output back
    ASSERT_TRUE(!params.osize.empty()) << "Error: params osize is empty";
    gpu_output
        = allocate_host_buffer<fftwAllocator<char>>(params.precision, params.otype, params.osize);
    ASSERT_TRUE(!gpu_output.empty()) << "no output buffers";
    for(int idx = 0; idx < gpu_output.size(); ++idx)
    {
        ASSERT_TRUE(!gpu_output[idx].empty()) << "output buffer index " << idx << " is empty";
        auto hip_status = hipMemcpy(gpu_output[idx].data(),
                                    pobuffer.at(idx),
                                    gpu_output[idx].size(),
                                    hipMemcpyDeviceToHost);
        ASSERT_EQ(hip_status, hipSuccess) << "hipMemcpy failure";
    }
    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        params.print_obuffer(gpu_output);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        params.print_obuffer_flat(gpu_output);
    }
}

template <typename Tfloat>
static void execute_cpu_fft(rocfft_params&                               params,
                            fft_params&                                  contiguous_params,
                            typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan,
                            fftw_data_t&                                 cpu_input,
                            fftw_data_t&                                 cpu_output)
{
    // CPU output might not be allocated already for us, if FFTW never
    // needed an output buffer during planning
    if(cpu_output.empty())
        cpu_output = allocate_host_buffer<fftwAllocator<char>>(
            contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);

    // If this is either C2R or callbacks are enabled, the
    // input will be modified.  So we need to modify the copy instead.
    fftw_data_t  cpu_input_copy;
    fftw_data_t* input_ptr = &cpu_input;
    if(params.run_callbacks || contiguous_params.transform_type == fft_transform_type_real_inverse)
    {
        cpu_input_copy = cpu_input;
        input_ptr      = &cpu_input_copy;
    }

    // run FFTW (which may destroy CPU input)
    apply_load_callback(params, *input_ptr);
    fftw_run<Tfloat>(contiguous_params.transform_type,
                     cpu_plan,
                     input_ptr->front().data(),
                     cpu_output.front().data());
    // clean up
    fftw_destroy_plan_type(cpu_plan);
    // ask FFTW to fully clean up, since it tries to cache plan details
    fftw_cleanup();
    cpu_plan = nullptr;
    apply_store_callback(params, cpu_output);
}

// run CPU + rocFFT transform with the given params and compare
template <typename Tfloat>
static void fft_vs_reference_impl(rocfft_params& params)
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

    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();

    // Initial check of raw data footprint, not including any work
    // buffers - if that doesn't fit don't even bother going further
    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint))
    {
        if(verbose)
        {
            std::cout << "Problem raw data won't fit on device; skipped." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << vram_footprint << ") raw data too large for device";
        return;
    }

    // Create FFT plan - this will also allocate work buffer, but
    // will throw a specific exception if that step fails
    try
    {
        ASSERT_EQ(params.create_plan(), fft_status_success);
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        GTEST_SKIP() << "Problem size with work buffer (" << vram_footprint + params.workbuffersize
                     << ") too large for device";
    }

    // Recheck whether the raw data fits on the device, now that the
    // work buffer has been allocated (if required).
    if(verbose > 1)
    {
        size_t     free   = 0;
        size_t     total  = 0;
        hipError_t retval = hipMemGetInfo(&free, &total);
        if(retval != hipSuccess)
        {
            std::cerr << "hipMemGetInfo failed" << std::endl;
        }
        std::cout << "data footprint: " << vram_footprint << " (" << (double)vram_footprint
                  << ") workbuffer: " << params.workbuffersize << " ("
                  << (double)params.workbuffersize << ") free: " << free << " (" << (double)free
                  << ") total: " << total << " (" << (double)total << ")\n";
    }
    if(!vram_fits_problem(vram_footprint))
    {
        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << vram_footprint << ") too large for device";
        return;
    }

    fft_params contiguous_params;
    contiguous_params.length         = params.length;
    contiguous_params.precision      = params.precision;
    contiguous_params.placement      = fft_placement_notinplace;
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

    // helper function to convert double input/output to float
    // in-place so we don't need extra memory
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

    // Check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  Smaller batch runs can
    // compare against the larger data.
    fftw_data_t              cpu_input;
    fftw_data_t              cpu_output;
    std::shared_future<void> convert_cpu_output_precision;
    std::shared_future<void> convert_cpu_input_precision;
    bool                     run_fftw = true;
    if(last_cpu_fft_data.length == params.length
       && last_cpu_fft_data.transform_type == params.transform_type
       && last_cpu_fft_data.run_callbacks == params.run_callbacks)
    {
        if(last_cpu_fft_data.nbatch >= params.nbatch)
        {
            // use the cached input/output
            cpu_input.swap(last_cpu_fft_data.cpu_input);
            cpu_output.swap(last_cpu_fft_data.cpu_output);
            run_fftw = false;

            if(params.precision != last_cpu_fft_data.precision)
            {
                // Tests should be ordered so we do double first, then float.
                if(last_cpu_fft_data.precision == fft_precision_double)
                {
                    // convert the input/output to single-precision
                    convert_cpu_output_precision
                        = std::async(std::launch::async, [&]() { convert_to_single(cpu_output); });
                    convert_cpu_input_precision
                        = std::async(std::launch::async, [&]() { convert_to_single(cpu_input); });
                    last_cpu_fft_data.precision = fft_precision_single;
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
        }
        // If the last result has a smaller batch than the new
        // params, that might be a developer error - tests should be
        // ordered to generate the bigger batch first.  But if tests
        // got filtered or skipped due to insufficient memory, we
        // might never have tried to generate the bigger batch first.
        // So just fall through and redo the CPU FFT.
    }
    // Clear cache explicitly so that even if we didn't get a hit,
    // we're not uselessly holding on to cached cpu input/output
    last_cpu_fft_data = last_cpu_fft_cache();

    // Allocate CPU input
    if(run_fftw)
    {
        cpu_input = allocate_host_buffer<fftwAllocator<char>>(
            contiguous_params.precision, contiguous_params.itype, contiguous_params.isize);
    }

    // Create FFTW plan - this may write to input, but that's fine
    // since there's nothing in there right now
    typename fftw_trait<Tfloat>::fftw_plan_type cpu_plan = nullptr;
    if(run_fftw)
    {
        // Normally, we would want to defer allocation of CPU output
        // buffer until when we actually do the CPU FFT.  But if we're
        // using FFTW wisdom, FFTW needs an output buffer at plan
        // creation time.
        if(use_fftw_wisdom)
        {
            cpu_output = allocate_host_buffer<fftwAllocator<char>>(
                contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);
        }
        cpu_plan = fftw_plan_via_rocfft<Tfloat>(contiguous_params.length,
                                                contiguous_params.istride,
                                                contiguous_params.ostride,
                                                contiguous_params.nbatch,
                                                contiguous_params.idist,
                                                contiguous_params.odist,
                                                contiguous_params.transform_type,
                                                cpu_input,
                                                cpu_output);
    }

    // generate input
    if(run_fftw)
    {
        compute_input(contiguous_params, cpu_input);
        if(verbose > 3)
        {
            std::cout << "CPU input:\n";
            contiguous_params.print_ibuffer(cpu_input);
        }
    }

    // compute input norm
    std::shared_future<VectorNorms> cpu_input_norm = std::async(std::launch::async, [&]() {
        // in case the cached cpu input needed conversion, wait for it
        if(convert_cpu_input_precision.valid())
            convert_cpu_input_precision.get();

        auto input_norm = norm(cpu_input,
                               contiguous_params.ilength(),
                               contiguous_params.nbatch,
                               contiguous_params.precision,
                               contiguous_params.itype,
                               contiguous_params.istride,
                               contiguous_params.idist,
                               contiguous_params.ioffset);
        if(verbose > 2)
        {
            std::cout << "CPU Input Linf norm:  " << input_norm.l_inf << "\n";
            std::cout << "CPU Input L2 norm:    " << input_norm.l_2 << "\n";
        }
        return input_norm;
    });

    std::vector<gpubuf>  ibuffer(ibuffer_sizes.size());
    std::vector<void*>   pibuffer(ibuffer_sizes.size());
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    std::vector<void*>   pobuffer;

    // copy input to GPU
    //
    // limited scope for local variables
    {
        // in case the cached cpu input needed conversion, wait for it
        if(convert_cpu_input_precision.valid())
            convert_cpu_input_precision.get();

        // if GPU data layout differs from CPU layout, allocate temp host
        // buffer with desired layout and copy
        fftw_data_t* gpu_input = &cpu_input;
        fftw_data_t  temp_gpu_input;
        if(params.itype != contiguous_params.itype || params.istride != contiguous_params.istride
           || params.idist != contiguous_params.idist || params.isize != contiguous_params.isize)
        {
            temp_gpu_input = allocate_host_buffer<fftwAllocator<char>>(
                params.precision, params.itype, params.isize);
            copy_buffers(cpu_input,
                         temp_gpu_input,
                         params.ilength(),
                         params.nbatch,
                         params.precision,
                         contiguous_params.itype,
                         contiguous_params.istride,
                         contiguous_params.idist,
                         params.itype,
                         params.istride,
                         params.idist,
                         {0},
                         params.ioffset);
            gpu_input = &temp_gpu_input;
        }

        // allocate GPU input
        // GPU input and output buffers:
        for(unsigned int i = 0; i < ibuffer.size(); ++i)
        {
            auto hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
            if(hip_status != hipSuccess)
            {
                throw std::runtime_error("hipMalloc failure for input buffer " + std::to_string(i)
                                         + " size " + std::to_string(ibuffer_sizes[i]) + " "
                                         + params.str());
            }
            pibuffer[i] = ibuffer[i].data();
        }

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
                auto hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
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
                            std::cerr << "The system reports that there is enough space."
                                      << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "hipMemGetInfo also failed" << std::endl;
                    }
                    throw std::runtime_error("hipMalloc failure for output buffer "
                                             + std::to_string(i) + " size "
                                             + std::to_string(obuffer_sizes[i]) + " ("
                                             + std::to_string(static_cast<double>(obuffer_sizes[i]))
                                             + ") " + params.str());
                }
            }
        }
        pobuffer.resize(obuffer->size());
        for(unsigned int i = 0; i < obuffer->size(); ++i)
        {
            pobuffer[i] = obuffer->at(i).data();
        }

        // Copy input to GPU
        for(int idx = 0; idx < gpu_input->size(); ++idx)
        {
            auto hip_status = hipMemcpy(ibuffer[idx].data(),
                                        gpu_input->at(idx).data(),
                                        ibuffer_sizes[idx],
                                        hipMemcpyHostToDevice);
            if(hip_status != hipSuccess)
                throw std::runtime_error("hipMemcpy failure");
        }
    }

    // Run CPU transform, allocating output buffer
    //
    // NOTE: This must happen after input is copied to GPU and input
    // norm is computed, since the CPU FFT may overwrite the input.
    // We also want to make sure any temp buffer we created for
    // copying GPU input is gone before we allocate CPU output
    // buffer.
    VectorNorms              cpu_output_norm;
    std::shared_future<void> cpu_fft = std::async(std::launch::async, [&]() {
        // wait for input norm to finish, since we might overwrite input
        cpu_input_norm.get();

        if(run_fftw)
            execute_cpu_fft<Tfloat>(params, contiguous_params, cpu_plan, cpu_input, cpu_output);
        // in case the cached cpu output needed conversion, wait for it
        else if(convert_cpu_output_precision.valid())
            convert_cpu_output_precision.get();

        if(verbose > 3)
        {
            std::cout << "CPU output:\n";
            contiguous_params.print_obuffer(cpu_output);
        }

        cpu_output_norm = norm(cpu_output,
                               params.olength(),
                               params.nbatch,
                               params.precision,
                               contiguous_params.otype,
                               contiguous_params.ostride,
                               contiguous_params.odist,
                               contiguous_params.ooffset);
        if(verbose > 2)
        {
            std::cout << "CPU Output Linf norm: " << cpu_output_norm.l_inf << "\n";
            std::cout << "CPU Output L2 norm:   " << cpu_output_norm.l_2 << "\n";
        }
    });

    // execute GPU transform
    //
    // limited scope for local variables
    fftw_data_t gpu_output;
    execute_gpu_fft(params, pibuffer, pobuffer, gpu_output);

    // compute GPU output norm
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

    // compare output
    //
    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    // wait for cpu FFT so we can compute cutoff
    cpu_fft.get();
    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length = std::accumulate(params.length.begin(),
                                              params.length.end(),
                                              static_cast<size_t>(1),
                                              std::multiplies<size_t>());
    const double                           linf_cutoff
        = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);
    VectorNorms diff = distance(cpu_output,
                                gpu_output,
                                params.olength(),
                                params.nbatch,
                                params.precision,
                                contiguous_params.otype,
                                contiguous_params.ostride,
                                contiguous_params.odist,
                                params.otype,
                                params.ostride,
                                params.odist,
                                linf_failures,
                                linf_cutoff,
                                {0},
                                params.ooffset);

    ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_2));
    ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_inf));

    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));

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

    EXPECT_TRUE(diff.l_inf <= linf_cutoff)
        << "Linf test failed.  Linf:" << diff.l_inf
        << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.l_inf
        << "\tcutoff: " << linf_cutoff << params.str();

    EXPECT_TRUE(diff.l_2 / cpu_output_norm.l_2
                < sqrt(log2(total_length)) * type_epsilon(params.precision))
        << "L2 test failed. L2: " << diff.l_2
        << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.l_2
        << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
        << params.str();

    // store cpu output in cache
    last_cpu_fft_data.length         = params.length;
    last_cpu_fft_data.nbatch         = params.nbatch;
    last_cpu_fft_data.transform_type = params.transform_type;
    last_cpu_fft_data.run_callbacks  = params.run_callbacks;
    last_cpu_fft_data.precision      = params.precision;
    last_cpu_fft_data.cpu_output.swap(cpu_output);
    last_cpu_fft_data.cpu_input.swap(cpu_input);
}

void fft_vs_reference(rocfft_params& params)
{
    switch(params.precision)
    {
    case fft_precision_single:
        fft_vs_reference_impl<float>(params);
        break;
    case fft_precision_double:
        fft_vs_reference_impl<double>(params);
        break;
    }
}
