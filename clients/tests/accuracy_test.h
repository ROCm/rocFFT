// Copyright (C) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include <algorithm>
#include <future>
#include <iterator>
#include <vector>

#include "../../shared/gpubuf.h"
#include "../fft_params.h"
#include "fftw_transform.h"
#include "rocfft_against_fftw.h"
#include "test_params.h"

extern int    verbose;
extern size_t ramgb;

static const size_t ONE_GiB = 1 << 30;

typedef std::vector<std::vector<char, fftwAllocator<char>>> fftw_data_t;

typedef std::tuple<fft_transform_type, fft_result_placement, fft_array_type, fft_array_type>
    type_place_io_t;

// Estimate the amount of host memory needed.
inline size_t needed_ram(const fft_params& params, const int verbose)
{
    // We need at most 3 copies of the raw data: 2 are strictly
    // required (input + output) but we keep a third copy around to
    // save effort recomputing input for a smaller batch size or
    // precision.
    //
    // This calculation is assuming contiguous data - noncontiguous
    // temp buffers may be briefly required to mirror the data layout
    // on the GPU, but they're assumed to require a close enough
    // amount of space for the purposes of this estimate.
    size_t needed_ram = 3
                        * std::accumulate(params.length.begin(),
                                          params.length.end(),
                                          static_cast<size_t>(1),
                                          std::multiplies<size_t>());

    // Account for precision and data type:
    if(params.transform_type != fft_transform_type_real_forward
       && params.transform_type != fft_transform_type_real_inverse)
    {
        needed_ram *= 2;
    }
    switch(params.precision)
    {
    case fft_precision_single:
        needed_ram *= 4;
        break;
    case fft_precision_double:
        needed_ram *= 8;
        break;
    }

    needed_ram *= params.nbatch;

    if(verbose)
    {
        std::cout << "required host memory (GiB): " << needed_ram / ONE_GiB << std::endl;
    }

    return needed_ram;
}

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        return info.param.token();
    }
};

// Remember the results of the last FFT we computed with FFTW.  Tests
// are ordered so that later cases can often reuse this result.
struct last_cpu_fft_cache
{
    // keys to the cache
    std::vector<size_t> length;
    size_t              nbatch         = 0;
    fft_transform_type  transform_type = fft_transform_type_complex_forward;
    bool                run_callbacks  = false;
    fft_precision       precision      = fft_precision_single;

    // FFTW input/output
    fftw_data_t cpu_input;
    fftw_data_t cpu_output;
};
extern last_cpu_fft_cache last_cpu_fft_data;

const static std::vector<size_t> batch_range = {2, 1};

const static std::vector<fft_precision> precision_range
    = {fft_precision_double, fft_precision_single};
const static std::vector<fft_result_placement> place_range
    = {fft_placement_inplace, fft_placement_notinplace};
const static std::vector<fft_transform_type> trans_type_range = {fft_transform_type_complex_forward,
                                                                 fft_transform_type_complex_inverse,
                                                                 fft_transform_type_real_forward,
                                                                 fft_transform_type_real_inverse};

// Given a vector of vector of lengths, generate all unique permutations.
// Add an optional vector of ad-hoc lengths to the result.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths)
{
    std::vector<std::vector<size_t>> output;
    if(inlengths.size() == 0)
    {
        return output;
    }
    const size_t        dim = inlengths.size();
    std::vector<size_t> looplength(dim);
    for(int i = 0; i < dim; ++i)
    {
        looplength[i] = inlengths[i].size();
    }
    for(int idx = 0; idx < inlengths.size(); ++idx)
    {
        std::vector<size_t> index(dim);
        do
        {
            std::vector<size_t> length(dim);
            for(int i = 0; i < dim; ++i)
            {
                length[i] = inlengths[i][index[i]];
            }
            output.push_back(length);
        } while(increment_rowmajor(index, looplength));
    }
    // uniquify the result
    std::sort(output.begin(), output.end());
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return output;
}

// Return the valid rocFFT input and output types for a given transform type.
inline std::vector<std::pair<fft_array_type, fft_array_type>>
    iotypes(const fft_transform_type   transformType,
            const fft_result_placement place,
            const bool                 planar = true)
{
    std::vector<std::pair<fft_array_type, fft_array_type>> iotypes;
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_complex_interleaved, fft_array_type_complex_interleaved));
        if(planar)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_complex_planar, fft_array_type_complex_planar));
            if(place == fft_placement_notinplace)
            {
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_planar, fft_array_type_complex_interleaved));
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_interleaved, fft_array_type_complex_planar));
            }
        }
        break;
    case fft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_real, fft_array_type_hermitian_interleaved));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_real, fft_array_type_hermitian_planar));
        }
        break;
    case fft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_hermitian_interleaved, fft_array_type_real));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_hermitian_planar, fft_array_type_real));
        }
        break;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return iotypes;
}

// Generate all combinations of input/output types, from combinations of transform and placement
// types.
static std::vector<type_place_io_t>
    generate_types(fft_transform_type                       transform_type,
                   const std::vector<fft_result_placement>& place_range,
                   const bool                               planar)
{
    std::vector<type_place_io_t> ret;
    for(auto place : place_range)
    {
        for(auto iotype : iotypes(transform_type, place, planar))
        {
            ret.push_back(std::make_tuple(transform_type, place, iotype.first, iotype.second));
        }
    }
    return ret;
}

struct stride_generator
{
    struct stride_dist
    {
        stride_dist(const std::vector<size_t>& s, size_t d)
            : stride(s)
            , dist(d)
        {
        }
        std::vector<size_t> stride;
        size_t              dist;
    };

    // NOTE: allow for this ctor to be implicit, so it's less typing for a test writer
    //
    // cppcheck-suppress noExplicitConstructor
    stride_generator(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_list(stride_list_in)
    {
    }
    virtual std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                              size_t                     batch) const
    {
        std::vector<stride_dist> ret;
        for(const auto& s : stride_list)
            ret.emplace_back(s, 0);
        return ret;
    }
    std::vector<std::vector<size_t>> stride_list;
};

// Generate strides such that batch is essentially the innermost dimension
// e.g. given a batch-2 4x3x2 transform which logically looks like:
//
// batch0:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// batch1:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// we instead do stride-2 4x3x2 transform where first batch is the
// A's and second batch is the B's.
struct stride_generator_3D_inner_batch : public stride_generator
{
    explicit stride_generator_3D_inner_batch(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_generator(stride_list_in)
    {
    }
    std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                      size_t                     batch) const override
    {
        std::vector<stride_dist> ret = stride_generator::generate(lengths, batch);
        std::vector<size_t> strides{lengths[1] * lengths[2] * batch, lengths[2] * batch, batch};
        ret.emplace_back(strides, 1);
        return ret;
    }
};

// Create an array of parameters to pass to gtest.  Base generator
// that allows choosing transform type.
inline auto param_generator_base(const std::vector<fft_transform_type>&   type_range,
                                 const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 decltype(generate_types)                 types_generator,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,
                                 const bool                               planar        = true,
                                 const bool                               run_callbacks = false)
{

    std::vector<fft_params> params;

    // For any length, we compute double-precision CPU reference
    // for largest batch size first and reuse for smaller batch
    // sizes, then convert to single-precision.

    for(auto& transform_type : type_range)
    {
        for(const auto& lengths : v_lengths)
        {
            // try to ensure that we are given literal lengths, not
            // something to be passed to generate_lengths
            if(lengths.empty() || lengths.size() > 3)
            {
                assert(false);
                continue;
            }
            {
                for(const auto precision : precision_range)
                {
                    for(const auto batch : batch_range)
                    {
                        for(const auto& types :
                            types_generator(transform_type, place_range, planar))
                        {
                            for(const auto& istride_dist : istride.generate(lengths, batch))
                            {
                                for(const auto& ostride_dist : ostride.generate(lengths, batch))
                                {
                                    for(const auto& ioffset : ioffset_range)
                                    {
                                        for(const auto& ooffset : ooffset_range)
                                        {
                                            fft_params param;

                                            param.length         = lengths;
                                            param.istride        = istride_dist.stride;
                                            param.ostride        = ostride_dist.stride;
                                            param.nbatch         = batch;
                                            param.precision      = precision;
                                            param.transform_type = std::get<0>(types);
                                            param.placement      = std::get<1>(types);
                                            param.idist          = istride_dist.dist;
                                            param.odist          = ostride_dist.dist;
                                            param.itype          = std::get<2>(types);
                                            param.otype          = std::get<3>(types);
                                            param.ioffset        = ioffset;
                                            param.ooffset        = ooffset;

                                            if(run_callbacks)
                                            {
                                                // add a test if both input and output support callbacks
                                                if(param.itype != fft_array_type_complex_planar
                                                   && param.itype != fft_array_type_hermitian_planar
                                                   && param.otype != fft_array_type_complex_planar
                                                   && param.otype
                                                          != fft_array_type_hermitian_planar)
                                                {
                                                    param.run_callbacks = true;
                                                }
                                                else
                                                {
                                                    continue;
                                                }
                                            }
                                            if(param.valid(0))
                                            {
                                                params.push_back(param);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return params;
}

// Create an array of parameters to pass to gtest.  Default generator
// that picks all transform types.
inline auto param_generator(const std::vector<std::vector<size_t>>&  v_lengths,
                            const std::vector<fft_precision>&        precision_range,
                            const std::vector<size_t>&               batch_range,
                            const stride_generator&                  istride,
                            const stride_generator&                  ostride,
                            const std::vector<std::vector<size_t>>&  ioffset_range,
                            const std::vector<std::vector<size_t>>&  ooffset_range,
                            const std::vector<fft_result_placement>& place_range,
                            const bool                               planar,
                            const bool                               run_callbacks = false)
{
    return param_generator_base({fft_transform_type_complex_forward,
                                 fft_transform_type_complex_inverse,
                                 fft_transform_type_real_forward,
                                 fft_transform_type_real_inverse},
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,
                                planar,
                                run_callbacks);
}

// Create an array of parameters to pass to gtest.  Only tests complex-type transforms
inline auto param_generator_complex(const std::vector<std::vector<size_t>>&  v_lengths,
                                    const std::vector<fft_precision>&        precision_range,
                                    const std::vector<size_t>&               batch_range,
                                    const stride_generator&                  istride,
                                    const stride_generator&                  ostride,
                                    const std::vector<std::vector<size_t>>&  ioffset_range,
                                    const std::vector<std::vector<size_t>>&  ooffset_range,
                                    const std::vector<fft_result_placement>& place_range,
                                    const bool                               planar,
                                    const bool                               run_callbacks = false)
{
    return param_generator_base(
        {fft_transform_type_complex_forward, fft_transform_type_complex_inverse},
        v_lengths,
        precision_range,
        batch_range,
        generate_types,
        istride,
        ostride,
        ioffset_range,
        ooffset_range,
        place_range,
        planar,
        run_callbacks);
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_real(const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,
                                 const bool                               planar,
                                 const bool                               run_callbacks = false)
{
    return param_generator_base({fft_transform_type_real_forward, fft_transform_type_real_inverse},
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,
                                planar,
                                run_callbacks);
}

template <class Tcontainer>
auto param_generator_token(const Tcontainer& tokens)
{
    std::vector<fft_params> params;
    params.reserve(tokens.size());
    for(auto t : tokens)
    {
        params.push_back({});
        params.back().from_token(t);
    }
    return params;
}

struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
    // base address of input, to ensure that each callback gets an offset from that base
    void* base;
};

void* get_load_callback_host(fft_array_type itype, fft_precision precision);
void  apply_load_callback(const fft_params& params, fftw_data_t& input);
void  apply_store_callback(const fft_params& params, fftw_data_t& output);
void* get_store_callback_host(fft_array_type otype, fft_precision precision);

template <typename Tfloat>
inline void execute_cpu_fft(fft_params&                                  params,
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

// execute the GPU transform
template <class Tparams>
inline void execute_gpu_fft(Tparams&            params,
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

// run CPU + rocFFT transform with the given params and compare
template <class Tfloat, class Tparams>
inline void fft_vs_reference_impl(Tparams& params)
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

    // First try a quick estimation of vram footprint, to speed up skipping tests
    // that are too large to fit in the gpu (no plan created with the rocFFT backend)
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);

    if(!vram_fits_problem(raw_vram_footprint))
    {
        GTEST_SKIP() << "Raw problem size (" << raw_vram_footprint
                     << ") raw data too large for device";
    }

    if(verbose > 2)
    {
        std::cout << "Raw problem size: " << raw_vram_footprint << std::endl;
    }

    // If it passed the quick estimation test, go for the more
    // accurate calculation that actually creates the plan and
    // take into account the work buffer size
    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint))
    {
        if(verbose)
        {
            std::cout << "Problem raw data won't fit on device; skipped." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << vram_footprint << ") raw data too large for device";
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
        ASSERT_EQ(retval, hipSuccess) << "hipMemGetInfo failed with error " << retval;
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

    // Copy input to GPU
    //
    // Limited scope for local variables
    {
        // In case the cached cpu input needed conversion, wait for it
        if(convert_cpu_input_precision.valid())
            convert_cpu_input_precision.get();

        // If GPU data layout differs from CPU layout, allocate temp host
        // buffer with desired layout and copy
        fftw_data_t* gpu_input = &cpu_input;
        fftw_data_t  temp_gpu_input;
        if(params.itype != contiguous_params.itype || params.istride != contiguous_params.istride
           || params.idist != contiguous_params.idist || params.isize != contiguous_params.isize)
        {
            // allocation counts in elements, ibuffer_sizes is in bytes
            auto ibuffer_sizes_elems = ibuffer_sizes;
            for(auto& buf : ibuffer_sizes_elems)
                buf /= var_size<size_t>(params.precision, params.itype);

            temp_gpu_input = allocate_host_buffer<fftwAllocator<char>>(
                params.precision, params.itype, ibuffer_sizes_elems);
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

        // Allocate GPU input
        // GPU input and output buffers:
        for(unsigned int i = 0; i < ibuffer.size(); ++i)
        {
            auto hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
            ASSERT_EQ(hip_status, hipSuccess)
                << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i] << " "
                << params.str();
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
                    // Try and figure out why hip malloc failed.
                    size_t     free   = 0;
                    size_t     total  = 0;
                    hipError_t retval = hipMemGetInfo(&free, &total);
                    EXPECT_EQ(retval, hipSuccess) << "hipMemGetInfo failed with error " << retval;
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
                }
                ASSERT_EQ(hip_status, hipSuccess)
                    << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                    << " (" << static_cast<double>(obuffer_sizes[i]) << ") " << params.str();
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
            ASSERT_EQ(hip_status, hipSuccess) << "hipMemcpy failure with error " << hip_status;
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

    switch(params.precision)
    {
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_single = std::max(max_l2_eps_single,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_double = std::max(max_l2_eps_double,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

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

#endif
