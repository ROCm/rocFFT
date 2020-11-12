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

#include "../client_utils.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"
#include <future>
#include <vector>

typedef std::vector<std::vector<char, fftwAllocator<char>>> fftw_data_t;

// Compute the rocFFT transform and verify the accuracy against the provided CPU data.
void rocfft_transform(const std::vector<size_t>&            length,
                      const std::vector<size_t>&            istride,
                      const std::vector<size_t>&            ostride,
                      const size_t                          nbatch,
                      const rocfft_precision                precision,
                      const rocfft_transform_type           transformType,
                      const rocfft_array_type               itype,
                      const rocfft_array_type               otype,
                      const rocfft_result_placement         place,
                      const std::vector<size_t>&            cpu_istride,
                      const std::vector<size_t>&            cpu_ostride,
                      const size_t                          cpu_idist,
                      const size_t                          cpu_odist,
                      const rocfft_array_type               cpu_itype,
                      const rocfft_array_type               cpu_otype,
                      const std::shared_future<fftw_data_t> cpu_input,
                      const std::shared_future<fftw_data_t> cpu_output,
                      const size_t                          ramgb,
                      const std::shared_future<VectorNorms> cpu_output_norm);

// Print the test parameters
inline void print_params(const std::vector<size_t>&    length,
                         const std::vector<size_t>&    istride,
                         const std::vector<size_t>&    ostride,
                         const size_t                  nbatch,
                         const rocfft_result_placement place,
                         const rocfft_precision        precision,
                         const rocfft_transform_type   transformType,
                         const rocfft_array_type       itype,
                         const rocfft_array_type       otype)
{
    std::cout << "length:";
    for(const auto& i : length)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "istride:";
    for(const auto& i : istride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "ostride:";
    for(const auto& i : ostride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "nbatch: " << nbatch << "\n";
    if(place == rocfft_placement_inplace)
        std::cout << "in-place\n";
    else
        std::cout << "out-of-place\n";
    if(precision == rocfft_precision_single)
        std::cout << "single-precision\n";
    else
        std::cout << "double-precision\n";
    switch(transformType)
    {
    case rocfft_transform_type_complex_forward:
        std::cout << "complex forward:\t";
        break;
    case rocfft_transform_type_complex_inverse:
        std::cout << "complex inverse:\t";
        break;
    case rocfft_transform_type_real_forward:
        std::cout << "real forward:\t";
        break;
    case rocfft_transform_type_real_inverse:
        std::cout << "real inverse:\t";
        break;
    }
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << " -> ";
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << std::endl;
}

typedef std::
    tuple<rocfft_transform_type, rocfft_result_placement, rocfft_array_type, rocfft_array_type>
        type_place_io_t;

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, // length
                                                                 rocfft_precision,
                                                                 size_t, // batch
                                                                 std::vector<size_t>, // istride
                                                                 std::vector<size_t>, // ostride
                                                                 type_place_io_t>>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    struct cpu_fft_data
    {

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

        cpu_fft_data()                    = default;
        cpu_fft_data(cpu_fft_data&&)      = default;
        cpu_fft_data(const cpu_fft_data&) = default;
        cpu_fft_data& operator=(const cpu_fft_data&) = default;
        ~cpu_fft_data()                              = default;
    };
    static cpu_fft_data compute_cpu_fft(const std::vector<size_t>& length,
                                        size_t                     nbatch,
                                        rocfft_precision           precision,
                                        rocfft_transform_type      transformType);
};

const static std::vector<size_t> batch_range = {2, 1};

const static std::vector<rocfft_precision> precision_range
    = {rocfft_precision_single, rocfft_precision_double};
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
        } while(increment_colmajor(index, looplength));
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

// generate all combinations of input/output types, from combinations
// of transform and placement types
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

#endif
