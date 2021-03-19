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

#include "../client_utils.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

typedef std::vector<std::vector<char, fftwAllocator<char>>> fftw_data_t;

typedef std::
    tuple<rocfft_transform_type, rocfft_result_placement, rocfft_array_type, rocfft_array_type>
        type_place_io_t;

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<rocfft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    struct cpu_fft_params
    {
        std::vector<size_t>   length;
        size_t                nbatch;
        rocfft_precision      precision;
        rocfft_transform_type transform_type;

        // Input cpu parameters:
        std::vector<size_t> ilength;
        std::vector<size_t> istride;
        rocfft_array_type   itype;
        size_t              idist;

        // Output cpu parameters:
        std::vector<size_t> olength;
        std::vector<size_t> ostride;
        rocfft_array_type   otype;
        size_t              odist;

        std::shared_future<fftw_data_t> input;
        std::shared_future<VectorNorms> input_norm;
        std::shared_future<fftw_data_t> output;
        std::shared_future<VectorNorms> output_norm;

        cpu_fft_params()                      = default;
        cpu_fft_params(cpu_fft_params&&)      = default;
        cpu_fft_params(const cpu_fft_params&) = default;
        cpu_fft_params(const rocfft_params& rocparams)
        {
            itype = rocparams.itype;
            otype = rocparams.otype;

            idist = rocparams.idist;
            odist = rocparams.odist;

            istride = rocparams.istride;
            std::reverse(std::begin(istride), std::end(istride));
            ostride = rocparams.ostride;
            std::reverse(std::begin(ostride), std::end(ostride));

            const auto dim = rocparams.length.size();

            ilength = rocparams.length;
            if(rocparams.transform_type == rocfft_transform_type_real_inverse)
            {
                ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
            }
            std::reverse(std::begin(ilength), std::end(ilength));

            olength = rocparams.length;
            if(rocparams.transform_type == rocfft_transform_type_real_forward)
            {
                olength[dim - 1] = olength[dim - 1] / 2 + 1;
            }
            std::reverse(std::begin(olength), std::end(olength));
        }
        cpu_fft_params& operator=(const cpu_fft_params&) = default;
        ~cpu_fft_params()                                = default;
    };
    static cpu_fft_params compute_cpu_fft(const rocfft_params& params);

    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        std::string ret;

        switch(info.param.transform_type)
        {
        case rocfft_transform_type_complex_forward:
            ret += "complex_forward_";
            break;
        case rocfft_transform_type_complex_inverse:
            ret += "complex_inverse_";
            break;
        case rocfft_transform_type_real_forward:
            ret += "real_forward_";
            break;
        case rocfft_transform_type_real_inverse:
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
        case rocfft_precision_single:
            ret += "single_";
            break;
        case rocfft_precision_double:
            ret += "double_";
            break;
        }

        switch(info.param.placement)
        {
        case rocfft_placement_inplace:
            ret += "ip_";
            break;
        case rocfft_placement_notinplace:
            ret += "op_";
            break;
        }

        ret += "batch_";
        ret += std::to_string(info.param.nbatch);

        auto append_array_info = [&ret](const std::vector<size_t>& stride, rocfft_array_type type) {
            for(auto s : stride)
            {
                ret += std::to_string(s);
                ret += "_";
            }

            switch(type)
            {
            case rocfft_array_type_complex_interleaved:
                ret += "CI";
                break;
            case rocfft_array_type_complex_planar:
                ret += "CP";
                break;
            case rocfft_array_type_real:
                ret += "R";
                break;
            case rocfft_array_type_hermitian_interleaved:
                ret += "HI";
                break;
            case rocfft_array_type_hermitian_planar:
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

        return ret;
    }
};

// Compute the rocFFT transform and verify the accuracy against the provided CPU data.
void rocfft_transform(const rocfft_params&                 params,
                      const accuracy_test::cpu_fft_params& cpu,
                      const size_t                         ramgb);

extern std::tuple<std::vector<size_t>, size_t, rocfft_transform_type, accuracy_test::cpu_fft_params>
    last_cpu_fft;

const static std::vector<size_t> batch_range = {2, 1};

const static std::vector<rocfft_precision> precision_range
    = {rocfft_precision_double, rocfft_precision_single};
const static std::vector<rocfft_result_placement> place_range
    = {rocfft_placement_inplace, rocfft_placement_notinplace};

// Given a vector of vector of lengths, generate all unique permutations.
// Add an optional vector of ad-hoc lengths to the result.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths,
                     const std::vector<std::vector<size_t>>& adhocLengths = {})
{
    std::vector<std::vector<size_t>> output = adhocLengths;
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
inline std::vector<std::pair<rocfft_array_type, rocfft_array_type>>
    iotypes(const rocfft_transform_type transformType, const rocfft_result_placement place)
{
    std::vector<std::pair<rocfft_array_type, rocfft_array_type>> iotypes;
    switch(transformType)
    {
    case rocfft_transform_type_complex_forward:
    case rocfft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved));
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_complex_planar, rocfft_array_type_complex_planar));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_complex_planar, rocfft_array_type_complex_interleaved));
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar));
        }
        break;
    case rocfft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_real, rocfft_array_type_hermitian_interleaved));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_real, rocfft_array_type_hermitian_planar));
        }
        break;
    case rocfft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_hermitian_interleaved, rocfft_array_type_real));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_hermitian_planar, rocfft_array_type_real));
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
    generate_types(rocfft_transform_type                       transformType,
                   const std::vector<rocfft_result_placement>& place_range)
{
    std::vector<type_place_io_t> ret;
    for(auto place : place_range)
    {
        for(auto iotype : iotypes(transformType, place))
        {
            ret.push_back(std::make_tuple(transformType, place, iotype.first, iotype.second));
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
    stride_generator_3D_inner_batch(const std::vector<std::vector<size_t>>& stride_list_in)
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
inline auto param_generator_base(const std::vector<rocfft_transform_type>&   type_range,
                                 const std::vector<std::vector<size_t>>&     v_lengths,
                                 const std::vector<rocfft_precision>&        precision_range,
                                 const std::vector<size_t>&                  batch_range,
                                 decltype(generate_types)                    types_generator,
                                 const stride_generator&                     istride,
                                 const stride_generator&                     ostride,
                                 const std::vector<std::vector<size_t>>&     ioffset_range,
                                 const std::vector<std::vector<size_t>>&     ooffset_range,
                                 const std::vector<rocfft_result_placement>& place_range)
{

    std::vector<rocfft_params> params;

    for(auto& transform_type : type_range)
    {
        // For any length, we compute double-precision CPU reference
        // for largest batch size first and reuse for smaller batch
        // sizes, then convert to single-precision.

        for(const auto& lengths : generate_lengths(v_lengths))
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
                                        rocfft_params param;

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

    return params;
}

// Create an array of parameters to pass to gtest.  Default generator
// that picks all transform types.
inline auto param_generator(const std::vector<std::vector<size_t>>&     v_lengths,
                            const std::vector<rocfft_precision>&        precision_range,
                            const std::vector<size_t>&                  batch_range,
                            const stride_generator&                     istride,
                            const stride_generator&                     ostride,
                            const std::vector<std::vector<size_t>>&     ioffset_range,
                            const std::vector<std::vector<size_t>>&     ooffset_range,
                            const std::vector<rocfft_result_placement>& place_range)
{
    return param_generator_base({rocfft_transform_type_complex_forward,
                                 rocfft_transform_type_complex_inverse,
                                 rocfft_transform_type_real_forward,
                                 rocfft_transform_type_real_inverse},
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
inline auto param_generator_complex(const std::vector<std::vector<size_t>>&     v_lengths,
                                    const std::vector<rocfft_precision>&        precision_range,
                                    const std::vector<size_t>&                  batch_range,
                                    const stride_generator&                     istride,
                                    const stride_generator&                     ostride,
                                    const std::vector<std::vector<size_t>>&     ioffset_range,
                                    const std::vector<std::vector<size_t>>&     ooffset_range,
                                    const std::vector<rocfft_result_placement>& place_range)
{
    return param_generator_base(
        {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse},
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
inline auto param_generator_real(const std::vector<std::vector<size_t>>&     v_lengths,
                                 const std::vector<rocfft_precision>&        precision_range,
                                 const std::vector<size_t>&                  batch_range,
                                 const stride_generator&                     istride,
                                 const stride_generator&                     ostride,
                                 const std::vector<std::vector<size_t>>&     ioffset_range,
                                 const std::vector<std::vector<size_t>>&     ooffset_range,
                                 const std::vector<rocfft_result_placement>& place_range)
{
    return param_generator_base(
        {rocfft_transform_type_real_forward, rocfft_transform_type_real_inverse},
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
