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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include <algorithm>
#include <future>
#include <iterator>
#include <vector>

#include "../fft_params.h"
#include "fftw_transform.h"
#include "rocfft_against_fftw.h"

static const size_t ONE_GiB = 1 << 30;

typedef std::vector<std::vector<char, fftwAllocator<char>>> fftw_data_t;

typedef std::tuple<fft_transform_type, fft_result_placement, fft_array_type, fft_array_type>
    type_place_io_t;

// Estimate the amount of host memory needed.
inline size_t needed_ram(const fft_params& params, const int verbose)
{
    // Host input, output, and input copy: 3 buffers, all contiguous.
    // Also: FFTW dummy input, and an input copy for shared futures.  Total 5.
    size_t needed_ram = 5
                        * std::accumulate(params.length.begin(),
                                          params.length.end(),
                                          static_cast<size_t>(1),
                                          std::multiplies<size_t>());

    // GPU input buffer:
    needed_ram += std::inner_product(
        params.length.begin(), params.length.end(), params.istride.begin(), static_cast<size_t>(0));

    // GPU output buffer:
    needed_ram += std::inner_product(
        params.length.begin(), params.length.end(), params.ostride.begin(), static_cast<size_t>(0));

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
    struct cpu_fft_params
    {
        std::vector<size_t> length;
        size_t              nbatch         = 0;
        fft_precision       precision      = fft_precision_single;
        fft_transform_type  transform_type = fft_transform_type_complex_forward;

        // Input cpu parameters:
        std::vector<size_t> ilength;
        std::vector<size_t> istride;
        fft_array_type      itype = fft_array_type_complex_interleaved;
        size_t              idist = 0;

        // Output cpu parameters:
        std::vector<size_t> olength;
        std::vector<size_t> ostride;
        fft_array_type      otype = fft_array_type_complex_interleaved;
        size_t              odist = 0;

        std::shared_future<fftw_data_t> input;
        std::shared_future<VectorNorms> input_norm;
        std::shared_future<fftw_data_t> output;
        std::shared_future<VectorNorms> output_norm;

        cpu_fft_params()                      = default;
        cpu_fft_params(cpu_fft_params&&)      = default;
        cpu_fft_params(const cpu_fft_params&) = default;
        explicit cpu_fft_params(const fft_params& rocparams)
            : ilength(rocparams.length)
            , istride(rocparams.istride)
            , itype(rocparams.itype)
            , idist(rocparams.idist)
            , olength(rocparams.length)
            , ostride(rocparams.ostride)
            , otype(rocparams.otype)
            , odist(rocparams.odist)
        {
            std::reverse(std::begin(istride), std::end(istride));
            std::reverse(std::begin(ostride), std::end(ostride));

            const auto dim = rocparams.length.size();

            if(rocparams.transform_type == fft_transform_type_real_inverse)
            {
                ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
            }
            std::reverse(std::begin(ilength), std::end(ilength));

            if(rocparams.transform_type == fft_transform_type_real_forward)
            {
                olength[dim - 1] = olength[dim - 1] / 2 + 1;
            }
            std::reverse(std::begin(olength), std::end(olength));
        }
        cpu_fft_params& operator=(const cpu_fft_params&) = default;
        ~cpu_fft_params()                                = default;
    };
    static cpu_fft_params compute_cpu_fft(const fft_params& params);

    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        std::string ret;

        switch(info.param.transform_type)
        {
        case fft_transform_type_complex_forward:
            ret += "complex_forward_";
            break;
        case fft_transform_type_complex_inverse:
            ret += "complex_inverse_";
            break;
        case fft_transform_type_real_forward:
            ret += "real_forward_";
            break;
        case fft_transform_type_real_inverse:
            ret += "real_inverse_";
            break;
        }

        ret += "len_";

        for(auto n : info.param.length)
        {
            ret += std::to_string(n);
            ret += "_";
        }
        switch(info.param.precision)
        {
        case fft_precision_single:
            ret += "single_";
            break;
        case fft_precision_double:
            ret += "double_";
            break;
        }

        switch(info.param.placement)
        {
        case fft_placement_inplace:
            ret += "ip_";
            break;
        case fft_placement_notinplace:
            ret += "op_";
            break;
        }

        ret += "batch_";
        ret += std::to_string(info.param.nbatch);

        auto append_array_info = [&ret](const std::vector<size_t>& stride, fft_array_type type) {
            for(auto s : stride)
            {
                ret += std::to_string(s);
                ret += "_";
            }

            switch(type)
            {
            case fft_array_type_complex_interleaved:
                ret += "CI";
                break;
            case fft_array_type_complex_planar:
                ret += "CP";
                break;
            case fft_array_type_real:
                ret += "R";
                break;
            case fft_array_type_hermitian_interleaved:
                ret += "HI";
                break;
            case fft_array_type_hermitian_planar:
                ret += "HP";
                break;
            default:
                ret += "UN";
                break;
            }
        };

        ret += "_istride_";
        append_array_info(info.param.istride, info.param.itype);

        ret += "_ostride_";
        append_array_info(info.param.ostride, info.param.otype);

        ret += "_idist_";
        ret += std::to_string(info.param.idist);
        ret += "_odist_";
        ret += std::to_string(info.param.odist);

        ret += "_ioffset";
        for(auto n : info.param.ioffset)
        {
            ret += "_";
            ret += std::to_string(n);
        }

        ret += "_ooffset";
        for(auto n : info.param.ooffset)
        {
            ret += "_";
            ret += std::to_string(n);
        }

        if(info.param.run_callbacks)
            ret += "_CB";

        return ret;
    }
};

// Compute the rocFFT transform and verify the accuracy against the provided CPU data.
void rocfft_transform(fft_params& params, accuracy_test::cpu_fft_params& cpu, const size_t ramgb);

extern std::
    tuple<std::vector<size_t>, size_t, fft_transform_type, bool, accuracy_test::cpu_fft_params>
        last_cpu_fft;

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
    iotypes(const fft_transform_type transformType, const fft_result_placement place)
{
    std::vector<std::pair<fft_array_type, fft_array_type>> iotypes;
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_complex_interleaved, fft_array_type_complex_interleaved));
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_complex_planar, fft_array_type_complex_planar));
        if(place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_complex_planar, fft_array_type_complex_interleaved));
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_complex_interleaved, fft_array_type_complex_planar));
        }
        break;
    case fft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_real, fft_array_type_hermitian_interleaved));
        if(place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_real, fft_array_type_hermitian_planar));
        }
        break;
    case fft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_hermitian_interleaved, fft_array_type_real));
        if(place == fft_placement_notinplace)
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
                   const std::vector<fft_result_placement>& place_range)
{
    std::vector<type_place_io_t> ret;
    for(auto place : place_range)
    {
        for(auto iotype : iotypes(transform_type, place))
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
                                 bool                                     run_callbacks = false)
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
                        for(const auto& types : types_generator(transform_type, place_range))
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
                            const std::vector<fft_result_placement>& place_range)
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
                                place_range);
}

// Create an array of parameters to pass to gtest.  Only tests complex-type transforms
inline auto param_generator_complex(const std::vector<std::vector<size_t>>&  v_lengths,
                                    const std::vector<fft_precision>&        precision_range,
                                    const std::vector<size_t>&               batch_range,
                                    const stride_generator&                  istride,
                                    const stride_generator&                  ostride,
                                    const std::vector<std::vector<size_t>>&  ioffset_range,
                                    const std::vector<std::vector<size_t>>&  ooffset_range,
                                    const std::vector<fft_result_placement>& place_range)
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
        place_range);
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_real(const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range)
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
                                place_range);
}

#endif
