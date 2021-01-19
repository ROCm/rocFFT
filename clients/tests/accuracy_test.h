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
class accuracy_test : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, // length
                                                                 rocfft_precision,
                                                                 size_t, // batch
                                                                 std::vector<size_t>, // istride
                                                                 std::vector<size_t>, // ostride
                                                                 std::vector<size_t>, // ioffset
                                                                 std::vector<size_t>, // ooffset
                                                                 type_place_io_t>>
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
        // Dimension and transform type are expected to be in the test
        // suite name already.
        const std::vector<size_t>     length        = std::get<0>(info.param);
        const rocfft_precision        precision     = std::get<1>(info.param);
        const size_t                  nbatch        = std::get<2>(info.param);
        const std::vector<size_t>     istride       = std::get<3>(info.param);
        const std::vector<size_t>     ostride       = std::get<4>(info.param);
        const std::vector<size_t>     ioffset       = std::get<5>(info.param);
        const std::vector<size_t>     ooffset       = std::get<6>(info.param);
        type_place_io_t               type_place_io = std::get<7>(info.param);
        const rocfft_transform_type   type          = std::get<0>(type_place_io);
        const rocfft_result_placement place         = std::get<1>(type_place_io);
        const rocfft_array_type       itype         = std::get<2>(type_place_io);
        const rocfft_array_type       otype         = std::get<3>(type_place_io);

        std::string ret;

        switch(type)
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

        for(auto n : length)
        {
            ret += std::to_string(n);
            ret += "_";
        }
        switch(precision)
        {
        case rocfft_precision_single:
            ret += "single_";
            break;
        case rocfft_precision_double:
            ret += "double_";
            break;
        }

        switch(place)
        {
        case rocfft_placement_inplace:
            ret += "ip_";
            break;
        case rocfft_placement_notinplace:
            ret += "op_";
            break;
        }

        ret += "batch_";
        ret += std::to_string(nbatch);

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
        append_array_info(istride, itype);

        ret += "_ostride_";
        append_array_info(ostride, otype);
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
    return std::move(ret);
}

// Create an array of parameters to pass to gtest.
inline auto param_generator(const std::vector<std::vector<size_t>>&     v_lengths,
                            const std::vector<rocfft_precision>&        precision_range,
                            const std::vector<size_t>&                  batch_range,
                            const std::vector<std::vector<size_t>>&     istride_range,
                            const std::vector<std::vector<size_t>>&     ostride_range,
                            const std::vector<std::vector<size_t>>&     ioffset_range,
                            const std::vector<std::vector<size_t>>&     ooffset_range,
                            const std::vector<rocfft_result_placement>& place_range)
{

    std::vector<std::tuple<std::vector<size_t>,
                           rocfft_precision,
                           size_t,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           type_place_io_t>>
        params;

    for(auto& transform_type :
        std::vector<rocfft_transform_type>{rocfft_transform_type_complex_forward,
                                           rocfft_transform_type_complex_inverse,
                                           rocfft_transform_type_real_forward,
                                           rocfft_transform_type_real_inverse})
    {
        // Batch and precision must be in outer loop in order to ensure the correct CPU reference is
        // available.

        for(const auto precision : precision_range)
        {
            for(const auto batch : batch_range)
            {
                for(const auto& lengths : generate_lengths(v_lengths))
                {
                    for(const auto& types : generate_types(transform_type, place_range))
                    {
                        for(const auto& istride : istride_range)
                        {
                            for(const auto& ostride : ostride_range)
                            {
                                for(const auto& ioffset : ioffset_range)
                                {
                                    for(const auto& ooffset : ooffset_range)
                                    {
                                        decltype(params)::value_type param(lengths,
                                                                           precision,
                                                                           batch,
                                                                           istride,
                                                                           ostride,
                                                                           ioffset,
                                                                           ooffset,
                                                                           types);
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

    return params;
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_complex(const std::vector<std::vector<size_t>>&     v_lengths,
                                    const std::vector<rocfft_precision>&        precision_range,
                                    const std::vector<size_t>&                  batch_range,
                                    const std::vector<std::vector<size_t>>&     istride_range,
                                    const std::vector<std::vector<size_t>>&     ostride_range,
                                    const std::vector<std::vector<size_t>>&     ioffset_range,
                                    const std::vector<std::vector<size_t>>&     ooffset_range,
                                    const std::vector<rocfft_result_placement>& place_range)
{

    std::vector<std::tuple<std::vector<size_t>,
                           rocfft_precision,
                           size_t,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           type_place_io_t>>
        params;

    for(auto& transform_type : std::vector<rocfft_transform_type>{
            rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse})
    {
        // Batch and precision must be in outer loop in order to ensure the correct CPU reference is
        // available.

        for(const auto precision : precision_range)
        {
            for(const auto batch : batch_range)
            {
                for(const auto& lengths : generate_lengths(v_lengths))
                {
                    for(const auto& types : generate_types(transform_type, place_range))
                    {
                        for(const auto& istride : istride_range)
                        {
                            for(const auto& ostride : ostride_range)
                            {
                                for(const auto& ioffset : ioffset_range)
                                {
                                    for(const auto& ooffset : ooffset_range)
                                    {
                                        decltype(params)::value_type param(lengths,
                                                                           precision,
                                                                           batch,
                                                                           istride,
                                                                           ostride,
                                                                           ioffset,
                                                                           ooffset,
                                                                           types);
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

    return params;
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_real(const std::vector<std::vector<size_t>>&     v_lengths,
                                 const std::vector<rocfft_precision>&        precision_range,
                                 const std::vector<size_t>&                  batch_range,
                                 const std::vector<std::vector<size_t>>&     istride_range,
                                 const std::vector<std::vector<size_t>>&     ostride_range,
                                 const std::vector<std::vector<size_t>>&     ioffset_range,
                                 const std::vector<std::vector<size_t>>&     ooffset_range,
                                 const std::vector<rocfft_result_placement>& place_range)
{

    std::vector<std::tuple<std::vector<size_t>,
                           rocfft_precision,
                           size_t,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           std::vector<size_t>,
                           type_place_io_t>>
        params;

    for(auto& transform_type : std::vector<rocfft_transform_type>{
            rocfft_transform_type_real_forward, rocfft_transform_type_real_inverse})
    {
        // Batch and precision must be in outer loop in order to ensure the correct CPU reference is
        // available.

        for(const auto precision : precision_range)
        {
            for(const auto batch : batch_range)
            {
                for(const auto& lengths : generate_lengths(v_lengths))
                {
                    for(const auto& types : generate_types(transform_type, place_range))
                    {
                        for(const auto& istride : istride_range)
                        {
                            for(const auto& ostride : ostride_range)
                            {
                                for(const auto& ioffset : ioffset_range)
                                {
                                    for(const auto& ooffset : ooffset_range)
                                    {
                                        decltype(params)::value_type param(lengths,
                                                                           precision,
                                                                           batch,
                                                                           istride,
                                                                           ostride,
                                                                           ioffset,
                                                                           ooffset,
                                                                           types);
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

    return params;
}

#endif
